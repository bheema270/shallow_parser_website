#!/usr/bin/env python
# coding: utf-8
"""
inference_engine.py
===================
Telugu Shallow Parser V8 — Inference Engine for the web interface.

Loads:
  - metadata_v8.json   (label maps + char vocab)
  - best_model_v8.pt   (trained MuRIL_MTL_Hierarchical weights)

Architecture is identical to parallel_training.py (shallow_parser folder, V8).
No training, no dataset, no dataloader, no optimizer.
"""

import json
import unicodedata
import logging
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF

try:
    from indicnlp.tokenize import indic_tokenize, sentence_tokenize
    HAS_INDIC_NLP = True
except ImportError:
    HAS_INDIC_NLP = False

# Uvicorn configures this logger at INFO level, so diagnostics are visible in
# the terminal without changing global application logging.
logger = logging.getLogger("uvicorn.error")

# ======================================================
# CONFIG  (must match parallel_training.py V8 exactly)
# ======================================================
MAX_LEN             = 128
MAX_CHAR_LEN        = 50
MAX_SUFFIX_CHAR_LEN = 60
# Keep decoder work bounded for large documents. With early stopping enabled,
# windows can hold complete natural sentences without fragmentation.
MAX_INFERENCE_SUBWORDS = 100
MAX_INFERENCE_WORDS    = 40
CRF_DROPOUT         = 0.3
DEP_EMB_DIM         = 64
HEAD_DROPOUT        = 0.15
EMB_DROPOUT         = 0.10
CONF_THRESHOLD      = 0.30
LABEL_SMOOTHING     = 0.05     # present in loss_fns state dict, needed for model init

# These will be set after metadata is loaded
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
CHAR_VOCAB_SIZE = 0


# ======================================================
# CRF PACKING UTILITIES
# ======================================================
def pack_for_crf(emissions, tags, first_subword_mask):
    B        = emissions.size(0)
    num_tags = emissions.size(-1)
    word_counts = first_subword_mask.sum(dim=1)
    max_words   = word_counts.max().item()

    if max_words == 0:
        packed_emissions = emissions.new_zeros(B, 1, num_tags)
        length_mask      = torch.zeros(B, 1, dtype=torch.bool, device=emissions.device)
        packed_tags      = tags.new_zeros(B, 1) if tags is not None else None
        return packed_emissions, packed_tags, length_mask

    packed_emissions = emissions.new_zeros(B, max_words, num_tags)
    length_mask      = torch.zeros(B, max_words, dtype=torch.bool, device=emissions.device)
    packed_tags      = tags.new_zeros(B, max_words) if tags is not None else None

    for i in range(B):
        positions = first_subword_mask[i].nonzero(as_tuple=True)[0]
        n = positions.size(0)
        packed_emissions[i, :n] = emissions[i, positions]
        length_mask[i, :n]      = True
        if packed_tags is not None:
            packed_tags[i, :n] = tags[i, positions]

    return packed_emissions, packed_tags, length_mask


def unpack_crf_preds(decoded_paths, first_subword_mask, fill_value=-100):
    B       = first_subword_mask.size(0)
    SEQ_LEN = first_subword_mask.size(1)
    pred_tensor = torch.full((B, SEQ_LEN), fill_value, dtype=torch.long,
                             device=first_subword_mask.device)
    for i, path in enumerate(decoded_paths):
        positions = first_subword_mask[i].nonzero(as_tuple=True)[0]
        n = min(len(path), positions.size(0))
        pred_tensor[i, positions[:n]] = torch.tensor(path[:n], device=pred_tensor.device)
    return pred_tensor


# ======================================================
# COPY ATTENTION DECODER  (for Lemma)
# ======================================================
class CopyAttentionDecoder(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim=64, enc_hidden=128,
                 dec_hidden=256, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.dec_hidden      = dec_hidden
        self.char_vocab_size = char_vocab_size
        enc_dim = enc_hidden * 2

        self.char_emb     = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=PAD_ID)
        self.char_encoder = nn.GRU(char_emb_dim, enc_hidden, batch_first=True, bidirectional=True)
        self.bridge       = nn.Linear(encoder_dim, dec_hidden)
        self.context_proj = nn.Linear(encoder_dim, dec_hidden)
        self.decoder_gru  = nn.GRU(char_emb_dim + enc_dim + dec_hidden, dec_hidden, batch_first=True)
        self.attn_W       = nn.Linear(enc_dim,    dec_hidden, bias=False)
        self.attn_U       = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.attn_v       = nn.Linear(dec_hidden, 1,          bias=False)
        self.output_proj  = nn.Linear(dec_hidden + enc_dim, char_vocab_size)
        self.p_gen_linear = nn.Linear(dec_hidden + enc_dim + char_emb_dim, 1)
        self.dropout      = nn.Dropout(dropout)

    def _compute_attention(self, dec_hidden, enc_outputs, enc_proj, enc_mask):
        scores = self.attn_v(torch.tanh(
            enc_proj + self.attn_U(dec_hidden).unsqueeze(1)
        )).squeeze(-1)
        scores       = scores.masked_fill(~enc_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        context      = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights

    def forward(self, encoder_hidden, word_char_ids, word_char_lens,
                targets=None, max_len=None):
        if max_len is None:
            max_len = MAX_CHAR_LEN

        N   = encoder_hidden.size(0)
        dev = encoder_hidden.device

        char_emb_enc = self.char_emb(word_char_ids)
        safe_lens    = word_char_lens.clamp(min=1)
        packed       = nn.utils.rnn.pack_padded_sequence(
            char_emb_enc, safe_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        enc_outputs, _ = self.char_encoder(packed)
        enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_outputs, batch_first=True)

        actual_max = enc_outputs.size(1)
        src_len    = word_char_ids.size(1)
        if actual_max < src_len:
            pad_t       = torch.zeros(N, src_len - actual_max, enc_outputs.size(2), device=dev)
            enc_outputs = torch.cat([enc_outputs, pad_t], dim=1)

        enc_mask = torch.arange(src_len, device=dev).unsqueeze(0) < word_char_lens.unsqueeze(1)
        enc_proj = self.attn_W(enc_outputs)

        h         = self.bridge(encoder_hidden).unsqueeze(0)
        muril_ctx = self.context_proj(encoder_hidden)
        decode_len = (targets.size(1) - 1) if targets is not None else (max_len - 1)
        if targets is not None:
            inp_seq = targets[:, :-1]

        all_outputs = []
        prev_pred   = None
        done = torch.zeros(N, dtype=torch.bool, device=dev)
        for t in range(decode_len):
            if targets is not None:
                inp_char = inp_seq[:, t]
            else:
                inp_char = (torch.full((N,), SOS_ID, dtype=torch.long, device=dev)
                            if t == 0 else prev_pred)

            inp_emb                = self.char_emb(inp_char)
            attn_ctx, attn_weights = self._compute_attention(
                h.squeeze(0), enc_outputs, enc_proj, enc_mask
            )
            gru_input = torch.cat([inp_emb, attn_ctx, muril_ctx], dim=1).unsqueeze(1)
            out, h    = self.decoder_gru(gru_input, h)
            out       = self.dropout(out.squeeze(1))

            gen_logit  = self.output_proj(torch.cat([out, attn_ctx], dim=1))
            gen_dist   = F.softmax(gen_logit, dim=-1)
            p_gen      = torch.sigmoid(self.p_gen_linear(torch.cat([out, attn_ctx, inp_emb], dim=1)))
            copy_dist  = torch.zeros(N, self.char_vocab_size, device=dev)
            copy_dist.scatter_add_(1, word_char_ids[:, :src_len], attn_weights)
            final_dist = p_gen * gen_dist + (1 - p_gen) * copy_dist

            if targets is not None:
                all_outputs.append(final_dist)
            else:
                prev_pred = final_dist.argmax(-1)
                all_outputs.append(prev_pred)
                done |= (prev_pred == EOS_ID)
                if done.all():
                    break

        return torch.stack(all_outputs, dim=1)


# ======================================================
# SUFFIX SEQ2SEQ DECODER  (also reused for TAM)
# Returns (outputs, final_hidden) — a TUPLE
# ======================================================
class SuffixSeqDecoder(nn.Module):
    """GRU decoder with attention. Returns (stacked_outputs, final_hidden)."""

    def __init__(self, char_vocab_size, char_emb_dim=64, enc_hidden=128,
                 dec_hidden=256, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.dec_hidden      = dec_hidden
        self.char_vocab_size = char_vocab_size
        enc_dim = enc_hidden * 2

        self.char_emb     = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=PAD_ID)
        self.char_encoder = nn.GRU(char_emb_dim, enc_hidden, batch_first=True, bidirectional=True)
        self.bridge       = nn.Linear(encoder_dim, dec_hidden)
        self.context_proj = nn.Linear(encoder_dim, dec_hidden)
        self.decoder_gru  = nn.GRU(char_emb_dim + enc_dim + dec_hidden, dec_hidden, batch_first=True)
        self.attn_W       = nn.Linear(enc_dim,    dec_hidden, bias=False)
        self.attn_U       = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.attn_v       = nn.Linear(dec_hidden, 1,          bias=False)
        self.output_proj  = nn.Linear(dec_hidden + enc_dim, char_vocab_size)
        self.dropout      = nn.Dropout(dropout)

    def _compute_attention(self, dec_hidden, enc_outputs, enc_proj, enc_mask):
        scores = self.attn_v(torch.tanh(
            enc_proj + self.attn_U(dec_hidden).unsqueeze(1)
        )).squeeze(-1)
        scores       = scores.masked_fill(~enc_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        context      = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context

    def forward(self, encoder_hidden, word_char_ids, word_char_lens,
                targets=None, max_len=None):
        if max_len is None:
            max_len = MAX_SUFFIX_CHAR_LEN

        N   = encoder_hidden.size(0)
        dev = encoder_hidden.device

        char_emb_enc = self.char_emb(word_char_ids)
        safe_lens    = word_char_lens.clamp(min=1)
        packed       = nn.utils.rnn.pack_padded_sequence(
            char_emb_enc, safe_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        enc_outputs, _ = self.char_encoder(packed)
        enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_outputs, batch_first=True)

        actual_max = enc_outputs.size(1)
        src_len    = word_char_ids.size(1)
        if actual_max < src_len:
            pad_t       = torch.zeros(N, src_len - actual_max, enc_outputs.size(2), device=dev)
            enc_outputs = torch.cat([enc_outputs, pad_t], dim=1)

        enc_mask = torch.arange(src_len, device=dev).unsqueeze(0) < word_char_lens.unsqueeze(1)
        enc_proj = self.attn_W(enc_outputs)

        h         = self.bridge(encoder_hidden).unsqueeze(0)
        muril_ctx = self.context_proj(encoder_hidden)
        decode_len = (targets.size(1) - 1) if targets is not None else (max_len - 1)
        if targets is not None:
            inp_seq = targets[:, :-1]

        all_outputs = []
        prev_pred   = None
        done = torch.zeros(N, dtype=torch.bool, device=dev)
        for t in range(decode_len):
            if targets is not None:
                inp_char = inp_seq[:, t]
            else:
                inp_char = (torch.full((N,), SOS_ID, dtype=torch.long, device=dev)
                            if t == 0 else prev_pred)

            inp_emb   = self.char_emb(inp_char)
            attn_ctx  = self._compute_attention(h.squeeze(0), enc_outputs, enc_proj, enc_mask)
            gru_input = torch.cat([inp_emb, attn_ctx, muril_ctx], dim=1).unsqueeze(1)
            out, h    = self.decoder_gru(gru_input, h)
            out       = self.dropout(out.squeeze(1))
            logit     = self.output_proj(torch.cat([out, attn_ctx], dim=1))

            if targets is not None:
                all_outputs.append(logit)
            else:
                prev_pred = logit.argmax(-1)
                all_outputs.append(prev_pred)
                done |= (prev_pred == EOS_ID)
                if done.all():
                    break

        # Return outputs AND final decoder hidden state squeezed to [N, dec_hidden]
        return torch.stack(all_outputs, dim=1), h.squeeze(0)


# ======================================================
# FUSION HEAD  (V8 architecture)
# ======================================================
class FusionHead(nn.Module):
    """
    One cascading classification head.

    Design:
      1. MuRIL features `x` are ALWAYS the base signal (residual path).
      2. If this head has a dependency input (dep_dim > 0), it is first
         projected into the same H-dim space and ADDED as a residual.
      3. Order: residual add -> LayerNorm -> GELU -> Dropout -> Linear.
    """
    def __init__(self, H, dep_dim, num_classes, dropout=HEAD_DROPOUT):
        super().__init__()
        self.has_dep = dep_dim > 0
        if self.has_dep:
            self.dep_proj = nn.Linear(dep_dim, H)
            nn.init.xavier_uniform_(self.dep_proj.weight)
            nn.init.zeros_(self.dep_proj.bias)
        self.ln = nn.LayerNorm(H)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(H, num_classes)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, dep_emb=None):
        fused = x
        if self.has_dep and dep_emb is not None:
            fused = fused + self.dep_proj(dep_emb)   # residual, not raw concat
        fused = self.ln(fused)
        fused = self.dropout(F.gelu(fused))
        return self.out(fused)


# ======================================================
# LEXCAT-BASED FEATURE MASK (V9 / V1)
# ======================================================
LEXCAT_FEATURE_MASK = {
    'n':    {'person': True,  'num': True,  'gender': True,  'case': True},
    'nst':  {'person': True,  'num': True,  'gender': True,  'case': True},
    'adj':  {'person': True,  'num': True,  'gender': True,  'case': False},
    'v':    {'person': True,  'num': True,  'gender': True,  'case': False},
    'pn':   {'person': True,  'num': True,  'gender': True,  'case': True},
    'adv':  {'person': False, 'num': False, 'gender': False, 'case': False},
    'num':  {'person': False, 'num': False, 'gender': False, 'case': False},
    'avy':  {'person': False, 'num': False, 'gender': False, 'case': False},
    'punc': {'person': False, 'num': False, 'gender': False, 'case': False},
}
LEXCAT_DEFAULT_MASK = {'person': False, 'num': False, 'gender': False, 'case': False}


# ======================================================
# MAIN MODEL — MuRIL_MTL_Hierarchical (V8 & V9 compatible)
# ======================================================
class MuRIL_MTL_Hierarchical(nn.Module):
    def __init__(self, label2id, cls_tasks, char_vocab_size, muril_model="google/muril-base-cased", gender_dep_dim=64):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(muril_model)

        for layer in self.encoder.encoder.layer[:6]:
            for p in layer.parameters():
                p.requires_grad = False

        H = self.encoder.config.hidden_size  # 768
        E = DEP_EMB_DIM  # 64
        TAM_H = 256       # TAM decoder hidden size

        self.label2id = label2id
        self.gender_dep_dim = gender_dep_dim

        # Task embeddings — only for tasks consumed as a dependency
        dep_source_tasks = ["pos", "lexcat", "person"]
        self.task_embs = nn.ModuleDict({
            t: nn.Embedding(len(label2id[t]), E) for t in dep_source_tasks
        })

        # Learnable "neutral" vector used when upstream label is UNK / low-confidence
        self.neutral_emb = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(E)) for t in dep_source_tasks
        })

        # Learnable per-dependency trust strength (alpha)
        self.dep_alpha = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(0.0)) for t in dep_source_tasks
        })

        self.emb_dropout = nn.Dropout(EMB_DROPOUT)

        # Cascading Heads with FusionHead (supports V8: dep_dim=64 and V9/V1: dep_dim=128)
        self.heads = nn.ModuleDict({
            "pos":    FusionHead(H, dep_dim=0,              num_classes=len(label2id["pos"])),
            "lexcat": FusionHead(H, dep_dim=E,               num_classes=len(label2id["lexcat"])),
            "person": FusionHead(H, dep_dim=E,               num_classes=len(label2id["person"])),
            "num":    FusionHead(H, dep_dim=E,               num_classes=len(label2id["num"])),
            "gender": FusionHead(H, dep_dim=gender_dep_dim,  num_classes=len(label2id["gender"])),
            "case":   FusionHead(H, dep_dim=E + TAM_H,       num_classes=len(label2id["case"])),
        })

        # Loss functions — present in state dict, needed for loading
        self.loss_fns = nn.ModuleDict({
            t: nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
            for t in cls_tasks
        })

        # Uncertainty-weighted multi-task loss (Kendall) — present in state dict
        all_loss_keys = cls_tasks + ["chunk", "lemma", "suffix", "tam"]
        self.log_vars = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1)) for t in all_loss_keys
        })

        # Chunk CRF (Independent)
        self.chunk_dropout = nn.Dropout(CRF_DROPOUT)
        self.chunk_head    = nn.Linear(H, len(label2id["chunk"]))
        self.chunk_crf     = CRF(len(label2id["chunk"]), batch_first=True)

        # Seq2Seq Decoders
        self.lemma_decoder  = CopyAttentionDecoder(char_vocab_size, encoder_dim=H)
        self.suffix_decoder = SuffixSeqDecoder(char_vocab_size, encoder_dim=H)
        self.tam_decoder    = SuffixSeqDecoder(char_vocab_size, encoder_dim=H)

    def _soft_dependency_embedding_inference(self, task, logits):
        """Inference-only version of _soft_dependency_embedding (no teacher forcing)."""
        probs = F.softmax(logits, dim=-1)
        conf, pred_idx = probs.max(dim=-1)
        emb_matrix = self.task_embs[task].weight
        soft_emb = torch.matmul(probs, emb_matrix)

        neutral = self.neutral_emb[task].view(1, 1, -1).expand_as(soft_emb)

        # UNK suppression
        unk_id = self.label2id[task].get('[UNK]', None)
        unk_raw_id = self.label2id[task].get('UNK', None)
        if unk_id is not None:
            pred_is_unk = (pred_idx == unk_id).unsqueeze(-1)
            soft_emb = torch.where(pred_is_unk, neutral, soft_emb)
        if unk_raw_id is not None:
            pred_is_raw_unk = (pred_idx == unk_raw_id).unsqueeze(-1)
            soft_emb = torch.where(pred_is_raw_unk, neutral, soft_emb)

        # Low-confidence fallback
        low_conf = (conf < CONF_THRESHOLD).unsqueeze(-1)
        soft_emb = torch.where(low_conf, neutral, soft_emb)

        # Confidence gating
        gate = torch.sigmoid(self.dep_alpha[task]) * conf.unsqueeze(-1)
        dep_emb = self.emb_dropout(soft_emb * gate)
        return dep_emb

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                first_subword_mask=None, word_chars=None, word_char_lens=None,
                **kwargs):
        x = self.encoder(input_ids, attention_mask, token_type_ids).last_hidden_state

        preds = {}

        # 1. POS (independent)
        pos_logits   = self.heads["pos"](x)
        preds["pos"] = pos_logits.argmax(-1)

        # 2. Chunk CRF (independent)
        chunk_emissions = self.chunk_head(self.chunk_dropout(x))
        if first_subword_mask is not None:
            packed_emissions, _, length_mask = pack_for_crf(
                chunk_emissions, None, first_subword_mask
            )
            decoded_paths  = self.chunk_crf.decode(packed_emissions, mask=length_mask)
            preds["chunk"] = unpack_crf_preds(decoded_paths, first_subword_mask)
        else:
            preds["chunk"] = chunk_emissions.argmax(-1)

        # 3. Seq2Seq decoders
        lemma_preds  = None
        suffix_preds = None
        tam_preds    = None
        tam_repr     = None

        fsw = first_subword_mask.bool() if first_subword_mask is not None else None
        if fsw is not None and word_chars is not None:
            hidden_at_fsw = x[fsw]
            if hidden_at_fsw.size(0) > 0:
                word_chars_fsw = word_chars[fsw]
                word_lens_fsw  = word_char_lens[fsw]

                lemma_preds = self.lemma_decoder(
                    hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=None, max_len=25
                )
                suffix_out, _ = self.suffix_decoder(
                    hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=None, max_len=20
                )
                suffix_preds = suffix_out

                tam_out, tam_h = self.tam_decoder(
                    hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=None, max_len=20
                )
                tam_preds = tam_out
                tam_repr  = tam_h

        # 4. Full-sequence TAM repr for Case head
        full_tam_repr = torch.zeros(x.size(0), x.size(1), 256, device=x.device)
        if tam_repr is not None and fsw is not None:
            full_tam_repr[fsw] = tam_repr

        # 5. Cascading classification heads (V8: soft dependency embeddings)
        pos_emb       = self._soft_dependency_embedding_inference("pos", pos_logits)
        lexcat_logits = self.heads["lexcat"](x, pos_emb)
        preds["lexcat"] = lexcat_logits.argmax(-1)

        lexcat_emb    = self._soft_dependency_embedding_inference("lexcat", lexcat_logits)
        person_logits = self.heads["person"](x, lexcat_emb)
        num_logits    = self.heads["num"](x, lexcat_emb)

        person_emb    = self._soft_dependency_embedding_inference("person", person_logits)
        if self.gender_dep_dim == DEP_EMB_DIM * 2:
            gender_dep = torch.cat([person_emb, lexcat_emb], dim=-1)
        else:
            gender_dep = person_emb
        gender_logits = self.heads["gender"](x, gender_dep)

        case_dep      = torch.cat([lexcat_emb, full_tam_repr], dim=-1)
        case_logits   = self.heads["case"](x, case_dep)

        all_logits = {
            "pos": pos_logits, "lexcat": lexcat_logits, "person": person_logits,
            "num": num_logits, "gender": gender_logits, "case": case_logits
        }

        # V9 / V1: LexCat-based feature masking at inference
        if self.gender_dep_dim == DEP_EMB_DIM * 2:
            lexcat_pred_ids = preds["lexcat"]
            for feat in ['person', 'num', 'gender', 'case']:
                unk_id = self.label2id[feat].get('UNK', self.label2id[feat].get('[UNK]', 0))
                feat_logits = all_logits[feat]
                mask = torch.zeros_like(lexcat_pred_ids, dtype=torch.bool)
                for lc_label, lc_id in self.label2id["lexcat"].items():
                    feat_info = LEXCAT_FEATURE_MASK.get(lc_label, LEXCAT_DEFAULT_MASK)
                    if not feat_info.get(feat, False):
                        mask = mask | (lexcat_pred_ids == lc_id)
                if mask.any():
                    mask_expanded = mask.unsqueeze(-1).expand_as(feat_logits)
                    feat_logits = feat_logits.clone()
                    feat_logits[mask_expanded] = float('-inf')
                    feat_logits[:, :, unk_id] = feat_logits[:, :, unk_id].masked_fill(mask, 0.0)
                    all_logits[feat] = feat_logits

        preds["person"] = all_logits["person"].argmax(-1)
        preds["num"]    = all_logits["num"].argmax(-1)
        preds["gender"] = all_logits["gender"].argmax(-1)
        preds["case"]   = all_logits["case"].argmax(-1)

        return {
            "preds":        preds,
            "hidden":       x,
            "lemma_preds":  lemma_preds,
            "suffix_preds": suffix_preds,
            "tam_preds":    tam_preds,
        }


# ======================================================
# UNICODE HELPERS
# ======================================================
def clean_telugu(text: str) -> str:
    """Fix Telugu Unicode rendering issues."""
    text = unicodedata.normalize("NFC", text)
    result = []
    prev_ch = None
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Mc", "Mn") and ch == prev_ch:
            continue
        result.append(ch)
        prev_ch = ch
    return "".join(result)


def decode_seq_chars(ids, id2char_map):
    """Decode a list of char IDs → Telugu string, then clean Unicode."""
    chars = []
    for i in ids:
        if i == EOS_ID:
            break
        if i in (PAD_ID, SOS_ID):
            continue
        chars.append(id2char_map.get(i, "?"))
    raw = "".join(chars)
    return clean_telugu(raw)


# ======================================================
# INFERENCE ENGINE CLASS
# ======================================================
class InferenceEngine:
    """Wraps the Telugu Shallow Parser V8 model for inference."""

    def __init__(self, model_path: str, metadata_path: str, muril_model: str):
        """Load model and metadata at startup."""
        global PAD_ID, SOS_ID, EOS_ID, UNK_ID, CHAR_VOCAB_SIZE

        logger.info("Loading metadata from %s ...", metadata_path)
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        self.label2id  = metadata["label2id"]
        self.id2label  = metadata["id2label"]
        self.char2id   = metadata["char2id"]
        self.id2char   = metadata["id2char"]
        self.cls_tasks = metadata["cls_tasks"]
        self.all_tasks = metadata["all_tasks"]

        # JSON keys come back as strings — convert to int for lookups
        self.id2label = {task: {int(k): v for k, v in mapping.items()}
                         for task, mapping in self.id2label.items()}
        self.id2char  = {int(k): v for k, v in self.id2char.items()}

        # Char special token IDs
        PAD_ID = self.char2id["<PAD>"]
        SOS_ID = self.char2id["<SOS>"]
        EOS_ID = self.char2id["<EOS>"]
        UNK_ID = self.char2id["<UNK>"]
        CHAR_VOCAB_SIZE = len(self.char2id)

        logger.info("  Tasks           : %s", self.all_tasks)
        logger.info("  Char vocab size : %d", CHAR_VOCAB_SIZE)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("  Device          : %s", self.device)

        # Load weights and detect checkpoint architecture version
        logger.info("Loading model weights from %s ...", model_path)
        state = torch.load(model_path, map_location=self.device, weights_only=False)

        gender_dep_w = state.get("heads.gender.dep_proj.weight")
        gender_dep_dim = gender_dep_w.shape[1] if gender_dep_w is not None else 64
        logger.info("  Detected architecture gender dependency dim: %d (%s)",
                    gender_dep_dim, "V9/V1: Person+LexCat" if gender_dep_dim == 128 else "V8: Person")

        # Build model
        logger.info("Building MuRIL_MTL_Hierarchical model...")
        self.model = MuRIL_MTL_Hierarchical(
            label2id=self.label2id,
            cls_tasks=self.cls_tasks,
            char_vocab_size=CHAR_VOCAB_SIZE,
            muril_model=muril_model,
            gender_dep_dim=gender_dep_dim,
        ).to(self.device)

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            logger.info("  Ignored %d training-only keys: %s%s",
                        len(unexpected), unexpected[:3],
                        "..." if len(unexpected) > 3 else "")
        if missing:
            logger.warning("  Missing keys: %s", missing)

        self.model.eval()
        logger.info("Model loaded successfully.")

        # Tokenizer
        logger.info("Loading MuRIL tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(muril_model)
        logger.info("Tokenizer loaded.")

        self.loaded = True

    def _encode_word_chars(self, word):
        """Encode a word into character IDs."""
        ids = [self.char2id.get(c, UNK_ID) for c in word[:MAX_CHAR_LEN]]
        length = len(ids) or 1
        if not ids:
            ids = [UNK_ID]
        while len(ids) < MAX_CHAR_LEN:
            ids.append(PAD_ID)
        return ids, length

    def _preprocess(self, tokens):
        """Tokenize and create model inputs for a list of word tokens. Uses dynamic padding for speed."""
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        word_ids            = enc.word_ids()
        prev                = None
        word_chars_list     = []
        word_char_lens_list = []
        first_subword_mask  = []

        for wi in word_ids:
            if wi is None or wi == prev:
                word_chars_list.append([PAD_ID] * MAX_CHAR_LEN)
                word_char_lens_list.append(0)
                first_subword_mask.append(0)
            else:
                wc, wl = self._encode_word_chars(tokens[wi])
                word_chars_list.append(wc)
                word_char_lens_list.append(wl)
                first_subword_mask.append(1)
            prev = wi

        batch = {
            "input_ids":          enc["input_ids"],
            "attention_mask":     enc["attention_mask"],
            "token_type_ids":     enc["token_type_ids"],
            "word_chars":         torch.tensor([word_chars_list],     dtype=torch.long),
            "word_char_lens":     torch.tensor([word_char_lens_list], dtype=torch.long),
            "first_subword_mask": torch.tensor([first_subword_mask],  dtype=torch.bool),
        }
        return batch, word_ids

    def _token_windows(self, tokens: list[str]) -> list[list[str]]:
        """Split a sentence into word windows that fit MuRIL's input limit.

        Windows are built on word boundaries so a token is never lost simply
        because the sentence contains more than 128 wordpieces.
        """
        max_content_pieces = min(
            MAX_INFERENCE_SUBWORDS,
            MAX_LEN - self.tokenizer.num_special_tokens_to_add(pair=False),
        )
        windows: list[list[str]] = []
        current: list[str] = []
        current_pieces = 0

        for token in tokens:
            # add_special_tokens=False gives the actual WordPiece count for a word.
            piece_count = len(self.tokenizer(token, add_special_tokens=False)["input_ids"])
            piece_count = max(piece_count, 1)
            if current and (
                len(current) >= MAX_INFERENCE_WORDS
                or current_pieces + piece_count > max_content_pieces
            ):
                windows.append(current)
                current = []
                current_pieces = 0
            current.append(token)
            current_pieces += piece_count

        if current:
            windows.append(current)
        logger.info(
            "[parser] Window plan: tokens=%d windows=%d token_counts=%s",
            len(tokens), len(windows), [len(window) for window in windows],
        )
        return windows

    def tokenize_sentence(self, sentence: str) -> list[str]:
        """
        Tokenize Telugu text into linguistic tokens using IndicNLP.
        Falls back to comprehensive unicode regex if IndicNLP is unavailable.
        """
        if HAS_INDIC_NLP:
            try:
                tokens = indic_tokenize.trivial_tokenize(sentence, lang="te")
                cleaned = [t.strip() for t in tokens if t.strip()]
                if cleaned:
                    return cleaned
            except Exception as e:
                logger.warning("[parser] IndicNLP tokenize failed: %s, falling back to regex", e)

        pattern = re.compile(r'[\u0C00-\u0C7F]+|[A-Za-z0-9_]+|[^\s\w\u0C00-\u0C7F]')
        tokens = pattern.findall(sentence)
        return [t.strip() for t in tokens if t.strip()]

    TELUGU_ABBRS = (
        "డాక్టర్", "డా", "Dr", "dr", "DR",
        "ప్రొఫెసర్", "ప్రొ", "Prof", "prof",
        "మిస్టర్", "మిసెస్", "Mr", "Mrs", "Ms",
        "శ్రీ", "రూ", "Rs", "నెం", "సం", "కి.మీ",
    )

    def segment_sentences(self, text: str) -> list[str]:
        """
        Segment multi-sentence input using IndicNLP sentence tokenizer,
        with abbreviation protection (Dr., డా., రూ., etc.) and fallback to punctuation.
        """
        # Protect common abbreviations from being treated as sentence endings
        protected = text
        for abbr in self.TELUGU_ABBRS:
            pattern = r'(?i)\b(' + re.escape(abbr) + r')\.'
            protected = re.sub(pattern, r'\1<__DOT__>', protected)

        if HAS_INDIC_NLP:
            try:
                lines = [line.strip() for line in protected.splitlines() if line.strip()]
                all_sents = []
                for line in lines:
                    sents = sentence_tokenize.sentence_split(line, lang="te")
                    for s in sents:
                        s = s.replace("<__DOT__>", ".").strip()
                        if s:
                            all_sents.append(s)
                if all_sents:
                    return all_sents
            except Exception as e:
                logger.warning("[parser] IndicNLP sentence_split failed: %s, falling back to regex", e)

        raw_lines = [line.strip() for line in protected.splitlines() if line.strip()]
        sentences = []
        for line in raw_lines:
            parts = re.split(r'(?<=[.!?।])(?=\s|$|[^.!?।\s])', line)
            for part in parts:
                part = part.replace("<__DOT__>", ".").strip()
                if part:
                    sentences.append(part)
        return sentences if sentences else [text.strip()]

    def _predict_tokens(self, tokens: list[str]) -> list[dict]:
        """Run one already-size-bounded token window through the model."""
        if not tokens:
            return []

        started_at = time.perf_counter()
        batch, word_ids = self._preprocess(tokens)
        active_subwords = int(batch["attention_mask"].sum().item())
        logger.info(
            "[parser] Window inference start: tokens=%d subwords=%d device=%s",
            len(tokens), active_subwords, self.device,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # inference_mode avoids autograd bookkeeping and lowers peak memory for
        # the three character decoders used by every token.
        with torch.inference_mode():
            out = self.model(**batch)

        logger.info(
            "[parser] Window model pass complete: tokens=%d elapsed=%.2fs",
            len(tokens), time.perf_counter() - started_at,
        )

        preds        = out["preds"]
        lemma_preds  = out["lemma_preds"]
        suffix_preds = out["suffix_preds"]
        tam_preds    = out["tam_preds"]

        fsw_mask      = batch["first_subword_mask"][0].bool()
        fsw_positions = fsw_mask.nonzero(as_tuple=True)[0].tolist()

        # Classification predictions
        word_cls_preds = {}
        for t in self.all_tasks:
            pred_row = preds[t][0]
            word_cls_preds[t] = []
            for pos in fsw_positions:
                pid = pred_row[pos].item()
                word_cls_preds[t].append(self.id2label[t].get(pid, "[UNK]"))

        # Seq2seq predictions
        lemma_strings  = ([decode_seq_chars(ids, self.id2char) for ids in lemma_preds.tolist()]
                          if lemma_preds is not None else [])
        suffix_strings = ([decode_seq_chars(ids, self.id2char) for ids in suffix_preds.tolist()]
                          if suffix_preds is not None else [])
        tam_strings    = ([decode_seq_chars(ids, self.id2char) for ids in tam_preds.tolist()]
                          if tam_preds is not None else [])

        results = []
        for wi, tok in enumerate(tokens):
            if wi >= len(fsw_positions):
                break
            tok_clean = clean_telugu(tok)
            pos_val = word_cls_preds.get("pos", ["-"] * (wi + 1))[wi] if wi < len(word_cls_preds.get("pos", [])) else "-"
            lexcat_val = word_cls_preds.get("lexcat", ["-"] * (wi + 1))[wi] if wi < len(word_cls_preds.get("lexcat", [])) else "-"
            case_val = word_cls_preds.get("case", ["-"] * (wi + 1))[wi] if wi < len(word_cls_preds.get("case", [])) else "-"
            # Helper: check if a generated seq2seq string contains any UNK token
            def _has_unk(s):
                return any(u in s for u in {"<UNK>", "[UNK]", "UNK"})

            raw_lemma   = lemma_strings[wi]  if wi < len(lemma_strings)  else tok_clean
            suffix_val  = suffix_strings[wi] if wi < len(suffix_strings) else "-"
            tam_val     = tam_strings[wi]    if wi < len(tam_strings)    else "-"

            # Lemma: if UNK appears anywhere in generated string → fall back to token (identity lemma)
            lemma_val = tok_clean if _has_unk(raw_lemma) or not raw_lemma else raw_lemma

            # Linguistic rule: Case belongs strictly to nominals (nouns, pronouns, numbers)
            is_nominal = pos_val in {"NN", "NNP", "PRP", "NST", "DEM", "QC", "QF", "QO"} or lexcat_val in {"n", "pn", "nst", "num"}
            clean_case = case_val if is_nominal and case_val not in {"UNK", "[UNK]", "<UNK>"} else "_"

            # Suffix + TAM cleanup:
            # If punctuation/symbol OR token == lemma (no morphological change) → suffix="-", TAM="_"
            # If UNK appears anywhere in generated suffix/TAM string → suffix="-", TAM="_"
            # Otherwise pass model's prediction as-is
            is_punc = pos_val in {"SYM", "punc"} or lexcat_val in {"punc"} or (len(tok_clean) == 1 and not tok_clean.isalnum())
            if is_punc or tok_clean == lemma_val:
                clean_suffix = "-"
                clean_tam = "_"
            else:
                clean_suffix = "-" if _has_unk(suffix_val) or suffix_val in {"", "-"} else suffix_val
                clean_tam    = "_" if _has_unk(tam_val)    or tam_val    in {"", "-"}  else tam_val

            entry = {
                "id":     wi + 1,
                "token":  tok_clean,
                "lemma":  lemma_val,
                "pos":    pos_val,
                "lexcat": lexcat_val,
                "chunk":  word_cls_preds.get("chunk",  ["-"] * (wi + 1))[wi] if wi < len(word_cls_preds.get("chunk", []))  else "-",
                "person": word_cls_preds.get("person", ["-"] * (wi + 1))[wi] if wi < len(word_cls_preds.get("person", [])) else "-",
                "number": word_cls_preds.get("num",    ["-"] * (wi + 1))[wi] if wi < len(word_cls_preds.get("num", []))    else "-",
                "gender": word_cls_preds.get("gender", ["-"] * (wi + 1))[wi] if wi < len(word_cls_preds.get("gender", [])) else "-",
                "case":   clean_case,
                "suffix": clean_suffix,
                "tam":    clean_tam,
            }
            results.append(entry)

        logger.info(
            "[parser] Window decoded: input_tokens=%d output_tokens=%d elapsed=%.2fs",
            len(tokens), len(results), time.perf_counter() - started_at,
        )
        return results

    def predict(self, sentence: str) -> list[dict]:
        """Analyze a single sentence, using windows when it exceeds MuRIL's limit."""
        tokens = self.tokenize_sentence(sentence)
        logger.info("[parser] Sentence tokenized: tokens=%d characters=%d", len(tokens), len(sentence))
        results: list[dict] = []
        windows = self._token_windows(tokens)
        for window_index, window in enumerate(windows, start=1):
            logger.info("[parser] Processing window %d/%d", window_index, len(windows))
            window_results = self._predict_tokens(window)
            for token in window_results:
                token["id"] = len(results) + 1
                results.append(token)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return results

    def predict_document(self, text: str) -> list[dict]:
        """
        Analyze multi-sentence text.
        Returns a list of sentence dicts:
        [
          {
            "sentence": sentence_str,
            "tokens": [ ... ]
          },
          ...
        ]
        """
        started_at = time.perf_counter()
        sentences = self.segment_sentences(text)
        logger.info(
            "[parser] Document start: characters=%d sentences=%d",
            len(text), len(sentences),
        )
        doc_results = []
        for sentence_id, s in enumerate(sentences, start=1):
            if not s.strip():
                continue
            toks = self.predict(s)
            doc_results.append({
                "sentence_id": sentence_id,
                "sentence": s,
                "tokens": toks,
            })
        logger.info(
            "[parser] Document complete: sentences=%d tokens=%d elapsed=%.2fs",
            len(doc_results), sum(len(sentence["tokens"]) for sentence in doc_results),
            time.perf_counter() - started_at,
        )
        return doc_results
