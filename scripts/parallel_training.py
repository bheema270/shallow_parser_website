#!/usr/bin/env python
# coding: utf-8
"""
parallel_training_v6.py
=======================
Shallow Parser V6 — MuRIL Multi-Task Learning for Telugu

Changes from V5:
  - Suffix is now predicted by a separate Seq2Seq GRU decoder (no copy)
    instead of a 1858-class classifier. This handles rare suffix
    combinations via character-level generation.
  - Includes all bug fixes from V5 code review.
"""

# ======================================================
# IMPORTS & SEED
# ======================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from torchcrf import CRF
import numpy as np
import random
import math
import os
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
print("Imports and seed completed")

# ======================================================
# CONFIG
# ======================================================
DATA_FILE = "data/whole_dataset.conllu"
BATCH_SIZE = 8
EPOCHS = 15
LR = 2e-5
MAX_LEN = 128
MAX_CHAR_LEN = 50
MAX_SUFFIX_CHAR_LEN = 60
CRF_DROPOUT = 0.3

# ------------------------------------------------------
# V8 additions: dependency fusion / UNK handling / MTL balancing
# ------------------------------------------------------
DEP_EMB_DIM       = 64     # dependency embedding size (unchanged from V7)
HEAD_DROPOUT      = 0.15   # dropout inside each cascading head
EMB_DROPOUT       = 0.10   # dropout applied to dependency embeddings
LABEL_SMOOTHING   = 0.05   # smoothing for all classification heads
WEIGHT_DECAY      = 0.01   # AdamW weight decay (excluded for norm/bias params)
CONF_THRESHOLD    = 0.30   # below this confidence, treat an upstream pred as "don't know"
TF_PROB_START     = 1.0    # scheduled sampling: start at full teacher forcing
TF_PROB_MIN       = 0.5    # never anneal below this (keep some gold signal)

# Initialize DDP if running with torchrun
local_rank = int(os.environ.get("LOCAL_RANK", 0))
is_distributed = "LOCAL_RANK" in os.environ
if is_distributed:
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if local_rank == 0:
    print("Using device:", device)

# POS_FEATURE_MASK and DEFAULT_MASK removed to prevent artificial masking corruption


# ======================================================
# LOAD DATA & FILTER INVALID CLASSES
# ======================================================
def load_data(path):
    sentences, labels = [], []
    tokens, labs = [], []

    pos_skip   = {'ECH', 'INTJ', 'NNPP', 'NP'}
    chunk_skip = {'B-VGN', 'I-VGN'}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    for l in labs:
                        if l["pos"]   in pos_skip:   l["pos"]   = '[UNK]'
                        if l["chunk"] in chunk_skip: l["chunk"] = '[UNK]'
                    sentences.append(tokens)
                    labels.append(labs)
                    tokens, labs = [], []
                continue
            p = line.split("\t")
            if len(p) != 11:
                continue
            tokens.append(p[0])
            labs.append({
                "lemma": p[1], "lexcat": p[2], "pos": p[3],
                "gender": p[4], "num": p[5], "person": p[6],
                "case": p[7], "suffix": p[8], "tam": p[9], "chunk": p[10],
            })
    return sentences, labels

sentences, labels = load_data(DATA_FILE)
if local_rank == 0:
    print("Total valid sentences after processing:", len(sentences))
if len(sentences) == 0:
    raise ValueError(f"No valid sentences found in {DATA_FILE}.")

# ======================================================
# TRAIN / DEV / TEST SPLIT
# ======================================================
indices = list(range(len(sentences)))
random.shuffle(indices)

total_size = len(sentences)
train_size = int(0.8 * total_size)
dev_size   = int(0.1 * total_size)

train_indices = indices[:train_size]
dev_indices   = indices[train_size:train_size + dev_size]
test_indices  = indices[train_size + dev_size:]

train_sentences    = [sentences[i] for i in train_indices]
train_labels_list  = [labels[i]    for i in train_indices]
dev_sentences      = [sentences[i] for i in dev_indices]
dev_labels_list    = [labels[i]    for i in dev_indices]
test_sentences     = [sentences[i] for i in test_indices]
test_labels_list   = [labels[i]    for i in test_indices]

if local_rank == 0:
    print("Dataset sizes (Train/Dev/Test):",
          len(train_sentences), len(dev_sentences), len(test_sentences))

# ======================================================
# BUILD LABEL MAPS ON TRAIN SET ONLY
# ======================================================
cls_tasks = ["pos", "lexcat", "person", "num", "gender", "case"]
all_tasks = cls_tasks + ["chunk"]
# lemma, suffix, and tam are handled by seq2seq decoder

label_set = {t: set(['[UNK]']) for t in all_tasks}
for sent in train_labels_list:
    for token in sent:
        for t in all_tasks:
            label_set[t].add(token[t])

label2id = {t: {l: i for i, l in enumerate(sorted(label_set[t]))} for t in all_tasks}
id2label  = {t: {i: l for l, i in label2id[t].items()}             for t in all_tasks}

if local_rank == 0:
    print("Labels created (excl. lemma&suffix — handled by attention+copy seq2seq)")
    for t in all_tasks:
        print(f"  {t}: {len(label2id[t])} classes")

# ======================================================
# CHAR VOCABULARY FOR LEMMA SEQ2SEQ
# ======================================================
PAD_CHAR, SOS_CHAR, EOS_CHAR, UNK_CHAR = '<PAD>', '<SOS>', '<EOS>', '<UNK>'

char_set = set()
for sent_words, sent_labs in zip(train_sentences, train_labels_list):
    for word, lab in zip(sent_words, sent_labs):
        char_set.update(word)
        char_set.update(lab["lemma"])
        char_set.update(lab["suffix"])  # include suffix chars for seq2seq

char_vocab     = [PAD_CHAR, SOS_CHAR, EOS_CHAR, UNK_CHAR] + sorted(char_set)
char2id        = {c: i for i, c in enumerate(char_vocab)}
id2char        = {i: c for c, i in char2id.items()}
CHAR_VOCAB_SIZE = len(char_vocab)
PAD_ID = char2id[PAD_CHAR]
SOS_ID = char2id[SOS_CHAR]
EOS_ID = char2id[EOS_CHAR]
UNK_ID = char2id[UNK_CHAR]

if local_rank == 0:
    print(f"Char vocabulary: {CHAR_VOCAB_SIZE} tokens (incl. PAD/SOS/EOS/UNK)")
    
    # Save the label and character maps to a metadata JSON file for inference
    metadata = {
        "label2id": label2id,
        "id2label": id2label,
        "char2id": char2id,
        "id2char": id2char,
        "cls_tasks": cls_tasks,
        "all_tasks": all_tasks
    }
    with open("metadata_v8.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open("metadata_v7.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("Saved metadata to metadata_v8.json and metadata_v7.json")


def encode_word_chars(word):
    ids = [char2id.get(c, UNK_ID) for c in word[:MAX_CHAR_LEN]]
    length = len(ids) or 1
    if not ids: ids = [UNK_ID]
    while len(ids) < MAX_CHAR_LEN: ids.append(PAD_ID)
    return ids, length

def encode_lemma_chars(lemma_str):
    ids = [SOS_ID] + [char2id.get(c, UNK_ID) for c in lemma_str[:MAX_CHAR_LEN - 2]] + [EOS_ID]
    while len(ids) < MAX_CHAR_LEN: ids.append(PAD_ID)
    return ids

def encode_suffix_chars(suffix_str):
    ids = [SOS_ID] + [char2id.get(c, UNK_ID) for c in suffix_str[:MAX_SUFFIX_CHAR_LEN - 2]] + [EOS_ID]
    while len(ids) < MAX_SUFFIX_CHAR_LEN: ids.append(PAD_ID)
    return ids

def encode_tam_chars(tam_str):
    # TAM is usually short, we can use the same max length as suffix
    ids = [SOS_ID] + [char2id.get(c, UNK_ID) for c in tam_str[:MAX_SUFFIX_CHAR_LEN - 2]] + [EOS_ID]
    while len(ids) < MAX_SUFFIX_CHAR_LEN: ids.append(PAD_ID)
    return ids

def decode_lemma_chars(ids):
    chars = []
    for i in ids:
        if i == EOS_ID: break
        if i in (PAD_ID, SOS_ID): continue
        chars.append(id2char.get(i, '?'))
    return ''.join(chars)

def decode_suffix_chars(ids):
    chars = []
    for i in ids:
        if i == EOS_ID: break
        if i in (PAD_ID, SOS_ID): continue
        chars.append(id2char.get(i, '?'))
    return ''.join(chars)

def decode_tam_chars(ids):
    chars = []
    for i in ids:
        if i == EOS_ID: break
        if i in (PAD_ID, SOS_ID): continue
        chars.append(id2char.get(i, '?'))
    return ''.join(chars)

def decode_word_chars_fn(ids):
    chars = []
    for i in ids:
        if i == PAD_ID: break
        chars.append(id2char.get(i, '?'))
    return ''.join(chars)


# ======================================================
# LOG-SMOOTHED CLASS WEIGHTS
# ======================================================
task_weights = {}
for t in cls_tasks:
    counts = {l: 0 for l in label2id[t]}
    for sent in train_labels_list:
        for token in sent:
            counts[token[t]] += 1
    total_samples = sum(counts.values())
    weights = []
    for i in range(len(label2id[t])):
        lbl   = id2label[t][i]
        count = counts.get(lbl, 0)
        w = max(1.0, math.log(total_samples / count)) if count > 0 else 1.0
        if lbl in ['[UNK]', 'UNK']:
            w = min(w, 0.5)
        weights.append(w)
    task_weights[t] = torch.tensor(weights, dtype=torch.float).to(device)

if local_rank == 0:
    print("Log-smoothed class weights calculated.")

# ======================================================
# DATASET
# ======================================================
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")


class MTLDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels    = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labs   = self.labels[idx]

        enc = tokenizer(
            tokens, is_split_into_words=True,
            truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )

        word_ids         = enc.word_ids()
        prev             = None
        lemma_targets    = []
        suffix_targets   = []
        tam_targets      = []
        word_chars_list  = []
        word_char_lens   = []
        first_subword_mask = []

        for t in all_tasks:
            enc[f"labels_{t}"] = []

        for wi in word_ids:
            if wi is None or wi == prev:
                for t in all_tasks: enc[f"labels_{t}"].append(-100)
                lemma_targets.append([PAD_ID] * MAX_CHAR_LEN)
                suffix_targets.append([PAD_ID] * MAX_SUFFIX_CHAR_LEN)
                tam_targets.append([PAD_ID] * MAX_SUFFIX_CHAR_LEN)
                word_chars_list.append([PAD_ID] * MAX_CHAR_LEN)
                word_char_lens.append(0)
                first_subword_mask.append(0)
            else:
                for t in all_tasks:
                    enc[f"labels_{t}"].append(
                        label2id[t].get(labs[wi][t], label2id[t]['[UNK]'])
                    )
                lemma_targets.append(encode_lemma_chars(labs[wi]["lemma"]))
                suffix_targets.append(encode_suffix_chars(labs[wi]["suffix"]))
                tam_targets.append(encode_tam_chars(labs[wi]["tam"]))
                wc, wl = encode_word_chars(tokens[wi])
                word_chars_list.append(wc)
                word_char_lens.append(wl)
                first_subword_mask.append(1)
            prev = wi

        for t in all_tasks:
            enc[f"labels_{t}"] = torch.tensor(enc[f"labels_{t}"])

        enc["lemma_targets"]     = torch.tensor(lemma_targets,     dtype=torch.long)
        enc["suffix_targets"]    = torch.tensor(suffix_targets,    dtype=torch.long)
        enc["tam_targets"]       = torch.tensor(tam_targets,       dtype=torch.long)
        enc["word_chars"]        = torch.tensor(word_chars_list,   dtype=torch.long)
        enc["word_char_lens"]    = torch.tensor(word_char_lens,    dtype=torch.long)
        enc["first_subword_mask"]= torch.tensor(first_subword_mask,dtype=torch.bool)

        return {k: v.squeeze(0) for k, v in enc.items()}


train_set = MTLDataset(train_sentences, train_labels_list)
dev_set   = MTLDataset(dev_sentences,   dev_labels_list)
test_set  = MTLDataset(test_sentences,  test_labels_list)

train_sampler = DistributedSampler(train_set)              if is_distributed else None
dev_sampler   = DistributedSampler(dev_set,  shuffle=False) if is_distributed else None
test_sampler  = DistributedSampler(test_set, shuffle=False) if is_distributed else None

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=(train_sampler is None), sampler=train_sampler)
dev_loader   = DataLoader(dev_set,   batch_size=BATCH_SIZE, sampler=dev_sampler)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, sampler=test_sampler)

if local_rank == 0:
    print("DataLoader logic complete")

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
# ATTENTION + COPY SEQ2SEQ DECODER
# ======================================================
class CopyAttentionDecoder(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim=64, enc_hidden=128,
                 dec_hidden=256, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.dec_hidden     = dec_hidden
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
        scores      = scores.masked_fill(~enc_mask, float('-inf'))
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
        for t in range(decode_len):
            if targets is not None:
                inp_char = inp_seq[:, t]
            else:
                inp_char = torch.full((N,), SOS_ID, dtype=torch.long, device=dev) if t == 0 else prev_pred

            inp_emb          = self.char_emb(inp_char)
            attn_ctx, attn_weights = self._compute_attention(h.squeeze(0), enc_outputs, enc_proj, enc_mask)
            gru_input        = torch.cat([inp_emb, attn_ctx, muril_ctx], dim=1).unsqueeze(1)
            out, h           = self.decoder_gru(gru_input, h)
            out              = self.dropout(out.squeeze(1))

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

        return torch.stack(all_outputs, dim=1)


# ======================================================
# SUFFIX SEQ2SEQ DECODER (Attention, NO Copy)
# ======================================================
class SuffixSeqDecoder(nn.Module):
    """GRU decoder with attention over word chars but NO copy mechanism.
    Suffix tags are abstract morphological labels, not substrings of the
    surface word, so copy would not help."""

    def __init__(self, char_vocab_size, char_emb_dim=64, enc_hidden=128,
                 dec_hidden=256, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.dec_hidden      = dec_hidden
        self.char_vocab_size = char_vocab_size
        enc_dim = enc_hidden * 2   # bidirectional

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
        scores       = scores.masked_fill(~enc_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context      = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context

    def forward(self, encoder_hidden, word_char_ids, word_char_lens,
                targets=None, max_len=None):
        if max_len is None:
            max_len = MAX_SUFFIX_CHAR_LEN

        N   = encoder_hidden.size(0)
        dev = encoder_hidden.device

        # Encode word characters
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
        for t in range(decode_len):
            if targets is not None:
                inp_char = inp_seq[:, t]
            else:
                inp_char = torch.full((N,), SOS_ID, dtype=torch.long, device=dev) if t == 0 else prev_pred

            inp_emb   = self.char_emb(inp_char)
            attn_ctx  = self._compute_attention(h.squeeze(0), enc_outputs, enc_proj, enc_mask)
            gru_input = torch.cat([inp_emb, attn_ctx, muril_ctx], dim=1).unsqueeze(1)
            out, h    = self.decoder_gru(gru_input, h)
            out       = self.dropout(out.squeeze(1))
            logit     = self.output_proj(torch.cat([out, attn_ctx], dim=1))

            if targets is not None:
                all_outputs.append(logit)       # raw logits for CE loss
            else:
                prev_pred = logit.argmax(-1)
                all_outputs.append(prev_pred)

        # Return both the outputs and the final decoder hidden state
        # h is [1, N, dec_hidden]. We squeeze to [N, dec_hidden]
        return torch.stack(all_outputs, dim=1), h.squeeze(0)


# ======================================================
# FUSION HEAD
# ======================================================
class FusionHead(nn.Module):
    """
    One cascading classification head.

    Design:
      1. MuRIL features `x` are ALWAYS the base signal (residual path) —
         they are never diluted or gated away, ensuring MuRIL stays dominant.
      2. If this head has a dependency input (dep_dim > 0), it is first
         projected into the same H-dim space (`dep_proj`) and ADDED as a
         residual on top of x rather than concatenated.
      3. Order after fusion: residual add -> LayerNorm -> GELU -> Dropout ->
         Linear(classifier). Small Xavier init keeps early logits near uniform.
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
# MODEL
# ======================================================

class MuRIL_MTL_Hierarchical(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/muril-base-cased")

        for layer in self.encoder.encoder.layer[:6]:
            for p in layer.parameters():
                p.requires_grad = False

        H = self.encoder.config.hidden_size  # 768
        E = DEP_EMB_DIM  # 64
        TAM_H = 256       # TAM decoder hidden size

        self.label2id = label2id

        # Task embeddings — only for tasks consumed as a dependency
        # pos -> lexcat; lexcat -> person, num, case; person -> gender
        dep_source_tasks = ["pos", "lexcat", "person"]
        self.task_embs = nn.ModuleDict({
            t: nn.Embedding(len(label2id[t]), E) for t in dep_source_tasks
        })
        for t in dep_source_tasks:
            nn.init.normal_(self.task_embs[t].weight, mean=0.0, std=0.02)

        # Learnable "neutral" vector used when upstream label is UNK / low-confidence
        self.neutral_emb = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(E)) for t in dep_source_tasks
        })

        # Learnable per-dependency trust strength (alpha): sigmoid(0.0) = 0.5 moderate initial trust
        self.dep_alpha = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(0.0)) for t in dep_source_tasks
        })

        self.emb_dropout = nn.Dropout(EMB_DROPOUT)

        # Cascading Heads with FusionHead
        self.heads = nn.ModuleDict({
            "pos":    FusionHead(H, dep_dim=0,        num_classes=len(label2id["pos"])),
            "lexcat": FusionHead(H, dep_dim=E,         num_classes=len(label2id["lexcat"])),
            "person": FusionHead(H, dep_dim=E,         num_classes=len(label2id["person"])),
            "num":    FusionHead(H, dep_dim=E,         num_classes=len(label2id["num"])),
            "gender": FusionHead(H, dep_dim=E,         num_classes=len(label2id["gender"])),
            "case":   FusionHead(H, dep_dim=E + TAM_H, num_classes=len(label2id["case"])),
        })
        
        self.loss_fns = nn.ModuleDict({
            t: nn.CrossEntropyLoss(weight=task_weights[t], ignore_index=-100,
                                   label_smoothing=LABEL_SMOOTHING) for t in cls_tasks
        })

        # Uncertainty-weighted multi-task loss balancing (Kendall et al. 2018)
        all_loss_keys = cls_tasks + ["chunk", "lemma", "suffix", "tam"]
        self.log_vars = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1)) for t in all_loss_keys
        })

        # Chunk CRF (Independent)
        self.chunk_dropout = nn.Dropout(CRF_DROPOUT)
        self.chunk_head    = nn.Linear(H, len(label2id["chunk"]))
        self.chunk_crf     = CRF(len(label2id["chunk"]), batch_first=True)
        
        # Seq2Seq Decoders (Independent from H, but TAM provides context to Case)
        self.lemma_decoder  = CopyAttentionDecoder(CHAR_VOCAB_SIZE, encoder_dim=H)
        self.suffix_decoder = SuffixSeqDecoder(CHAR_VOCAB_SIZE, encoder_dim=H)
        self.tam_decoder    = SuffixSeqDecoder(CHAR_VOCAB_SIZE, encoder_dim=H)

        # Scheduled-sampling teacher forcing probability
        self.tf_prob = TF_PROB_START

    def masked_cls_loss(self, task, logits, target):
        unk_id = self.label2id[task].get('[UNK]', None)
        target = target.clone()
        if unk_id is not None:
            target[target == unk_id] = -100
        unk_raw_id = self.label2id[task].get('UNK', None)
        if unk_raw_id is not None:
            target[target == unk_raw_id] = -100
        valid_mask = (target != -100)
        if not valid_mask.any():
            return (logits * 0.0).sum()
        return self.loss_fns[task](logits.view(-1, logits.size(-1)), target.view(-1))

    def _soft_dependency_embedding(self, task, logits, gold_labels=None):
        probs = F.softmax(logits, dim=-1)
        conf, pred_idx = probs.max(dim=-1)
        emb_matrix = self.task_embs[task].weight
        soft_emb = torch.matmul(probs, emb_matrix)

        unk_id = self.label2id[task].get('[UNK]', None)
        unk_raw_id = self.label2id[task].get('UNK', None)
        neutral = self.neutral_emb[task].view(1, 1, -1).expand_as(soft_emb)

        if unk_id is not None:
            pred_is_unk = (pred_idx == unk_id).unsqueeze(-1)
            soft_emb = torch.where(pred_is_unk, neutral, soft_emb)
        if unk_raw_id is not None:
            pred_is_raw_unk = (pred_idx == unk_raw_id).unsqueeze(-1)
            soft_emb = torch.where(pred_is_raw_unk, neutral, soft_emb)

        low_conf = (conf < CONF_THRESHOLD).unsqueeze(-1)
        soft_emb = torch.where(low_conf, neutral, soft_emb)

        if self.training and gold_labels is not None:
            ignore = (gold_labels == -100)
            safe_gold = gold_labels.clone()
            safe_gold[ignore] = 0
            gold_emb = emb_matrix[safe_gold]
            if unk_id is not None:
                gold_is_unk = (safe_gold == unk_id)
                gold_emb = torch.where(gold_is_unk.unsqueeze(-1), neutral, gold_emb)
            if unk_raw_id is not None:
                gold_is_raw_unk = (safe_gold == unk_raw_id)
                gold_emb = torch.where(gold_is_raw_unk.unsqueeze(-1), neutral, gold_emb)
            gold_emb = torch.where(ignore.unsqueeze(-1), neutral, gold_emb)

            use_gold = (torch.rand(conf.shape, device=conf.device) < self.tf_prob).unsqueeze(-1)
            dep_emb = torch.where(use_gold, gold_emb, soft_emb)
            conf_for_gate = torch.where(use_gold.squeeze(-1), torch.ones_like(conf), conf)
        else:
            dep_emb = soft_emb
            conf_for_gate = conf

        gate = torch.sigmoid(self.dep_alpha[task]) * conf_for_gate.unsqueeze(-1)
        dep_emb = self.emb_dropout(dep_emb * gate)
        return dep_emb

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                first_subword_mask=None, word_chars=None, word_char_lens=None,
                lemma_targets=None, suffix_targets=None, tam_targets=None, **labels):
        x = self.encoder(input_ids, attention_mask, token_type_ids).last_hidden_state

        losses, preds = {}, {}

        # 1. POS Head (independent from other heads) & Chunk Head
        pos_logits = self.heads["pos"](x)
        preds["pos"] = pos_logits.argmax(-1)
        
        chunk_emissions = self.chunk_head(self.chunk_dropout(x))
        if first_subword_mask is not None:
            chunk_tags = labels.get("labels_chunk", None)
            packed_emissions, packed_tags, length_mask = pack_for_crf(
                chunk_emissions, chunk_tags, first_subword_mask
            )
            if packed_tags is not None:
                safe_tags = packed_tags.clone()
                safe_tags[~length_mask] = 0
                losses["chunk"] = -self.chunk_crf(
                    packed_emissions, safe_tags, mask=length_mask, reduction='mean'
                )
            decoded_paths = self.chunk_crf.decode(packed_emissions, mask=length_mask)
            preds["chunk"] = unpack_crf_preds(decoded_paths, first_subword_mask)
        else:
            preds["chunk"] = chunk_emissions.argmax(-1)

        # 2. Sequence Decoders (run on first subwords only)
        fsw            = first_subword_mask.bool() if first_subword_mask is not None else None
        tam_repr       = None
        
        if fsw is not None and word_chars is not None:
            hidden_at_fsw  = x[fsw]
            if hidden_at_fsw.size(0) > 0:
                word_chars_fsw = word_chars[fsw]
                word_lens_fsw  = word_char_lens[fsw]

                # Lemma seq2seq loss
                if lemma_targets is not None:
                    targets_fsw    = lemma_targets[fsw]
                    final_dists    = self.lemma_decoder(
                        hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=targets_fsw
                    )
                    log_probs = torch.log(torch.clamp(final_dists, min=1e-8))
                    gold      = targets_fsw[:, 1:]
                    if (gold != PAD_ID).any():
                        losses["lemma"] = F.nll_loss(
                            log_probs.reshape(-1, CHAR_VOCAB_SIZE),
                            gold.reshape(-1), ignore_index=PAD_ID
                        )
                    else:
                        losses["lemma"] = (final_dists * 0.0).sum()

                # Suffix seq2seq loss
                if suffix_targets is not None:
                    suffix_tgt_fsw = suffix_targets[fsw]
                    suffix_logits, _ = self.suffix_decoder(
                        hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=suffix_tgt_fsw
                    )
                    suffix_gold = suffix_tgt_fsw[:, 1:]
                    if (suffix_gold != PAD_ID).any():
                        losses["suffix"] = F.cross_entropy(
                            suffix_logits.reshape(-1, CHAR_VOCAB_SIZE),
                            suffix_gold.reshape(-1), ignore_index=PAD_ID
                        )
                    else:
                        losses["suffix"] = (suffix_logits * 0.0).sum()
                    
                # TAM seq2seq loss & representation
                tam_tgt_fsw = tam_targets[fsw] if tam_targets is not None else None
                tam_logits, tam_h = self.tam_decoder(
                    hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=tam_tgt_fsw
                )
                tam_repr = tam_h # This is [N_words, 256]
                
                if tam_tgt_fsw is not None:
                    tam_gold = tam_tgt_fsw[:, 1:]
                    if (tam_gold != PAD_ID).any():
                        losses["tam"] = F.cross_entropy(
                            tam_logits.reshape(-1, CHAR_VOCAB_SIZE),
                            tam_gold.reshape(-1), ignore_index=PAD_ID
                        )
                    else:
                        losses["tam"] = (tam_logits * 0.0).sum()
                    
        # 3. Cascading Classification Hierarchy
        full_tam_repr = torch.zeros(x.size(0), x.size(1), 256, device=x.device)
        if tam_repr is not None and fsw is not None:
            full_tam_repr[fsw] = tam_repr

        # LexCat <- MuRIL (residual) + gated soft POS embedding
        pos_gold = labels.get("labels_pos", None)
        pos_emb = self._soft_dependency_embedding("pos", pos_logits, pos_gold)
        lexcat_logits = self.heads["lexcat"](x, pos_emb)
        preds["lexcat"] = lexcat_logits.argmax(-1)

        # Person <- MuRIL (residual) + gated soft LexCat embedding
        lexcat_gold = labels.get("labels_lexcat", None)
        lexcat_emb = self._soft_dependency_embedding("lexcat", lexcat_logits, lexcat_gold)
        person_logits = self.heads["person"](x, lexcat_emb)

        # Num <- MuRIL (residual) + gated soft LexCat embedding
        num_logits = self.heads["num"](x, lexcat_emb)

        # Gender <- MuRIL (residual) + gated soft Person embedding
        person_gold = labels.get("labels_person", None)
        person_emb = self._soft_dependency_embedding("person", person_logits, person_gold)
        gender_logits = self.heads["gender"](x, person_emb)

        # Case <- MuRIL (residual) + gated soft LexCat embedding + TAM decoder hidden state
        case_dep = torch.cat([lexcat_emb, full_tam_repr], dim=-1)
        case_logits = self.heads["case"](x, case_dep)

        preds["person"] = person_logits.argmax(-1)
        preds["num"]    = num_logits.argmax(-1)
        preds["gender"] = gender_logits.argmax(-1)
        preds["case"]   = case_logits.argmax(-1)

        all_logits = {
            "pos": pos_logits, "lexcat": lexcat_logits, "person": person_logits,
            "num": num_logits, "gender": gender_logits, "case": case_logits
        }

        # UNK-masked classification losses: gold==UNK is treated like padding (-100)
        for t in cls_tasks:
            if f"labels_{t}" in labels:
                losses[t] = self.masked_cls_loss(t, all_logits[t], labels[f"labels_{t}"])

        # Uncertainty-weighted multi-task loss (Kendall et al., 2018)
        if losses:
            total_loss = sum(
                torch.exp(-self.log_vars[t]) * l + self.log_vars[t]
                for t, l in losses.items()
            ).squeeze()
        else:
            total_loss = torch.tensor(0.0, device=x.device)

        return {"loss": total_loss, "losses": losses, "preds": preds, "hidden": x}

model = MuRIL_MTL_Hierarchical().to(device)
if is_distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

# Weight decay everywhere except LayerNorm, bias, and uncertainty/gating scalars
no_decay = ["bias", "ln.weight", "LayerNorm.weight", "layer_norm.weight", "log_vars", "dep_alpha", "neutral_emb"]
decay_params, no_decay_params = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay_params if any(nd in n for nd in no_decay) else decay_params).append(p)

optimizer = AdamW(
    [
        {"params": decay_params,    "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=LR,
)

total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

if local_rank == 0:
    print("Model, Optimizer, and Scheduler ready")

# ======================================================
# TRAIN
# ======================================================
best_dev      = float("inf")
patience      = 5
wait          = 0
all_loss_keys = cls_tasks + ["chunk", "lemma", "suffix", "tam"]

for epoch in range(EPOCHS):
    if is_distributed:
        train_sampler.set_epoch(epoch)

    # Linearly anneal scheduled sampling teacher forcing probability
    tf_prob = TF_PROB_START - (TF_PROB_START - TF_PROB_MIN) * (epoch / max(1, EPOCHS - 1))
    (model.module if hasattr(model, "module") else model).tf_prob = tf_prob

    model.train()
    train_loss        = 0
    train_task_losses = {t: 0.0 for t in all_loss_keys}

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out   = model(**batch)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += out["loss"].item()
        for t, l in out["losses"].items():
            train_task_losses[t] += l.item()

    model.eval()
    dev_loss        = 0
    dev_task_losses = {t: 0.0 for t in all_loss_keys}

    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            dev_loss += out["loss"].item()
            for t, l in out["losses"].items():
                dev_task_losses[t] += l.item()

    n_train     = len(train_loader)
    n_dev       = len(dev_loader)
    train_loss /= n_train
    dev_loss   /= n_dev

    # Synchronize dev loss across all DDP ranks for consistent early stopping
    if is_distributed:
        dev_loss_tensor = torch.tensor(dev_loss, device=device)
        dist.all_reduce(dev_loss_tensor, op=dist.ReduceOp.SUM)
        dev_loss = dev_loss_tensor.item() / dist.get_world_size()

    loss_parts = " | ".join(
        f"{t}: {train_task_losses[t]/n_train:.3f}/{dev_task_losses[t]/n_dev:.3f}"
        for t in all_loss_keys
    )
    if local_rank == 0:
        print(f"Epoch {epoch+1} | Train {train_loss:.4f} | Dev {dev_loss:.4f} | TF_prob {tf_prob:.3f} | {loss_parts}")

    if dev_loss < best_dev:
        best_dev = dev_loss
        wait     = 0
        if local_rank == 0:
            torch.save(
                model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "best_model_v8.pt"
            )
    else:
        wait += 1
        if wait >= patience:
            if local_rank == 0:
                print("Early stopping triggered")
            break

if local_rank == 0:
    print("Training complete")

if is_distributed:
    dist.barrier()

# ======================================================
# TEST
# ======================================================
if local_rank == 0:
    # ── FIX: unwrap DDP before loading checkpoint ──────────────────────────
    # Checkpoint was saved with model.module.state_dict() → keys have NO
    # "module." prefix. Loading into the DDP wrapper (which expects "module.*")
    # causes the RuntimeError seen in the previous run.
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(torch.load("best_model_v8.pt", map_location=device))
    raw_model.eval()
    # ───────────────────────────────────────────────────────────────────────

    gold = {t: [] for t in all_tasks}
    pred = {t: [] for t in all_tasks}
    lemma_gold, lemma_pred, lemma_words = [], [], []
    suffix_gold, suffix_pred = [], []
    tam_gold, tam_pred = [], []

    # FIX: Use non-distributed DataLoader so rank 0 evaluates the FULL test set
    full_test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for batch in full_test_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            out = raw_model(**batch_on_device)

            for t in all_tasks:
                p = out["preds"][t]
                g = batch_on_device[f"labels_{t}"]
                for pi, gi in zip(p, g):
                    for p_tok, g_tok in zip(pi, gi):
                        if g_tok != -100:
                            gold[t].append(id2label[t][g_tok.item()])
                            pred[t].append(id2label[t][p_tok.item()])

            fsw            = batch_on_device["first_subword_mask"].bool()
            hidden_at_fsw  = out["hidden"][fsw]
            word_chars_fsw = batch_on_device["word_chars"][fsw]
            word_lens_fsw  = batch_on_device["word_char_lens"][fsw]
            gold_lemma_ids = batch_on_device["lemma_targets"][fsw]
            gold_suffix_ids = batch_on_device["suffix_targets"][fsw]
            gold_tam_ids = batch_on_device["tam_targets"][fsw]

            if hidden_at_fsw.size(0) > 0:
                # Lemma predictions
                pred_lemma_ids = raw_model.lemma_decoder(
                    hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=None
                )
                for gi, pi, wi in zip(gold_lemma_ids, pred_lemma_ids, word_chars_fsw):
                    lemma_gold.append(decode_lemma_chars(gi.tolist()))
                    lemma_pred.append(decode_lemma_chars(pi.tolist()))
                    lemma_words.append(decode_word_chars_fn(wi.tolist()))

                # Suffix predictions
                pred_suffix_ids, _ = raw_model.suffix_decoder(
                    hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=None
                )
                for si_g, si_p in zip(gold_suffix_ids, pred_suffix_ids):
                    suffix_gold.append(decode_suffix_chars(si_g.tolist()))
                    suffix_pred.append(decode_suffix_chars(si_p.tolist()))
                    
                # TAM predictions
                pred_tam_ids, _ = raw_model.tam_decoder(
                    hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=None
                )
                for ti_g, ti_p in zip(gold_tam_ids, pred_tam_ids):
                    tam_gold.append(decode_tam_chars(ti_g.tolist()))
                    tam_pred.append(decode_tam_chars(ti_p.tolist()))

    print("\n================ TEST TOKEN LEVEL RESULTS ================\n")
    for t in all_tasks:
        print(f"\n===== {t.upper()} =====")
        print(classification_report(gold[t], pred[t], zero_division=0))

    lemma_exact = sum(g == p for g, p in zip(lemma_gold, lemma_pred))
    lemma_total = len(lemma_gold)
    lemma_acc   = lemma_exact / lemma_total if lemma_total > 0 else 0.0
    print(f"\n===== LEMMA (Attention + Copy Seq2Seq) =====")
    print(f"Exact Match Accuracy: {lemma_acc:.4f}  ({lemma_exact}/{lemma_total})")

    suffix_exact = sum(g == p for g, p in zip(suffix_gold, suffix_pred))
    suffix_total = len(suffix_gold)
    suffix_acc   = suffix_exact / suffix_total if suffix_total > 0 else 0.0
    print(f"\n===== SUFFIX (Attention Seq2Seq, no copy) =====")
    print(f"Exact Match Accuracy: {suffix_acc:.4f}  ({suffix_exact}/{suffix_total})")

    print("\nSample lemma predictions (first 30):")
    for i, (w, g, p) in enumerate(zip(lemma_words[:30], lemma_gold[:30], lemma_pred[:30])):
        marker = "[OK]" if g == p else "[X]"
        print(f"  {marker}  word='{w}'  gold='{g}'  pred='{p}'")

    print("\nSample suffix predictions (first 30):")
    for i, (g, p) in enumerate(zip(suffix_gold[:30], suffix_pred[:30])):
        marker = "[OK]" if g == p else "[X]"
        print(f"  {marker}  gold='{g}'  pred='{p}'")

    print("\n================ SUMMARY METRICS (TEST) ================\n")
    summary = {}
    for t in all_tasks:
        rpt = classification_report(gold[t], pred[t], zero_division=0, output_dict=True)
        summary[t] = {
            "accuracy":          rpt["accuracy"],
            "macro_f1":          rpt["macro avg"]["f1-score"],
            "weighted_f1":       rpt["weighted avg"]["f1-score"],
            "macro_precision":   rpt["macro avg"]["precision"],
            "macro_recall":      rpt["macro avg"]["recall"],
        }

    for t, m in summary.items():
        print(
            f"{t.upper():8s} | "
            f"Acc: {m['accuracy']:.4f} | "
            f"Macro P/R/F1: {m['macro_precision']:.4f} / {m['macro_recall']:.4f} / {m['macro_f1']:.4f} | "
            f"Weighted F1: {m['weighted_f1']:.4f}"
        )
    print(f"{'LEMMA':8s} | Exact Match Acc: {lemma_acc:.4f}")
    print(f"{'SUFFIX':8s} | Exact Match Acc: {suffix_acc:.4f}")
    
    tam_exact = sum(g == p for g, p in zip(tam_gold, tam_pred))
    tam_total = len(tam_gold)
    tam_acc   = tam_exact / tam_total if tam_total > 0 else 0.0
    print(f"{'TAM':8s} | Exact Match Acc: {tam_acc:.4f}")

if is_distributed:
    dist.destroy_process_group()
