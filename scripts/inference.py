#!/usr/bin/env python
# coding: utf-8
"""
inference.py
============
Telugu Shallow Parser V6 — Inference Only

Loads:
  - metadata_v6.json   (label maps + char vocab)
  - best_model_v6.pt   (trained MuRIL_MTL weights)

Architecture is identical to parallel_training.py.
No training, no dataset, no dataloader, no optimizer.
"""

# ======================================================
# IMPORTS
# ======================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
import json
import sys
import unicodedata
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ======================================================
# CONFIG  (must match parallel_training.py exactly)
# ======================================================
MAX_LEN             = 128
MAX_CHAR_LEN        = 50
MAX_SUFFIX_CHAR_LEN = 60
CRF_DROPOUT         = 0.3

METADATA_PATH = "metadata_v8.json"
MODEL_PATH    = "best_model_v8.pt"

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# STEP 1 — LOAD METADATA
# ======================================================
print("Loading metadata...")
with open(METADATA_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

label2id  = metadata["label2id"]   # {task: {label: id}}
id2label  = metadata["id2label"]   # {task: {str(id): label}}
char2id   = metadata["char2id"]    # {char: id}
id2char   = metadata["id2char"]    # {str(id): char}
cls_tasks = metadata["cls_tasks"]  # ["lexcat","pos","gender","num","person","case","tam"]
all_tasks = metadata["all_tasks"]  # cls_tasks + ["chunk"]

# id2label keys come back from JSON as strings — convert to int for lookups
id2label = {task: {int(k): v for k, v in mapping.items()}
            for task, mapping in id2label.items()}
id2char  = {int(k): v for k, v in id2char.items()}

# Char special token IDs
PAD_CHAR, SOS_CHAR, EOS_CHAR, UNK_CHAR = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
PAD_ID = char2id[PAD_CHAR]
SOS_ID = char2id[SOS_CHAR]
EOS_ID = char2id[EOS_CHAR]
UNK_ID = char2id[UNK_CHAR]

CHAR_VOCAB_SIZE = len(char2id)

print(f"  Tasks           : {all_tasks}")
print(f"  Char vocab size : {CHAR_VOCAB_SIZE}")
for t in all_tasks:
    print(f"  {t:10s}: {len(label2id[t])} classes")

# ======================================================
# CHAR ENCODING / DECODING HELPERS
# (identical to parallel_training.py)
# ======================================================
def encode_word_chars(word):
    ids = [char2id.get(c, UNK_ID) for c in word[:MAX_CHAR_LEN]]
    length = len(ids) or 1
    if not ids:
        ids = [UNK_ID]
    while len(ids) < MAX_CHAR_LEN:
        ids.append(PAD_ID)
    return ids, length


def clean_telugu(text: str) -> str:
    """
    Fix Telugu Unicode rendering issues:
    1. NFC-normalize  — composes combining sequences correctly
    2. Strip duplicate consecutive combining marks (ుు → ు)
       caused by the greedy seq2seq decoder over-predicting.
    """
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


def decode_lemma_chars(ids):
    chars = []
    for i in ids:
        if i == EOS_ID:
            break
        if i in (PAD_ID, SOS_ID):
            continue
        chars.append(id2char.get(i, "?"))
    return clean_telugu("".join(chars))


def decode_suffix_chars(ids):
    chars = []
    for i in ids:
        if i == EOS_ID:
            break
        if i in (PAD_ID, SOS_ID):
            continue
        chars.append(id2char.get(i, "?"))
    return clean_telugu("".join(chars))


# ======================================================
# CRF PACKING UTILITIES
# (identical to parallel_training.py)
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
# COPY ATTENTION DECODER
# (identical to parallel_training.py)
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

        return torch.stack(all_outputs, dim=1)


# ======================================================
# SUFFIX SEQ2SEQ DECODER (Attention, NO Copy)
# (identical to parallel_training.py)
# ======================================================
class SuffixSeqDecoder(nn.Module):
    """GRU decoder with attention over word chars but NO copy mechanism."""

    def __init__(self, char_vocab_size, char_emb_dim=64, enc_hidden=128,
                 dec_hidden=256, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.dec_hidden      = dec_hidden
        self.char_vocab_size = char_vocab_size
        enc_dim = enc_hidden * 2  # bidirectional

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

        return torch.stack(all_outputs, dim=1)


# ======================================================
# MAIN MODEL  (identical to parallel_training.py)
# ======================================================
class MuRIL_MTL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/muril-base-cased")

        # Freeze bottom 6 layers — same as training
        for layer in self.encoder.encoder.layer[:6]:
            for p in layer.parameters():
                p.requires_grad = False

        H = self.encoder.config.hidden_size  # 768

        self.heads = nn.ModuleDict({
            t: nn.Linear(H, len(label2id[t])) for t in cls_tasks
        })
        # loss_fns are part of the saved state_dict — must be defined here
        # so the keys match exactly. Weights don't matter at inference time.
        self.loss_fns = nn.ModuleDict({
            t: nn.CrossEntropyLoss(ignore_index=-100) for t in cls_tasks
        })
        self.chunk_dropout = nn.Dropout(CRF_DROPOUT)
        self.chunk_head    = nn.Linear(H, len(label2id["chunk"]))
        self.chunk_crf     = CRF(len(label2id["chunk"]), batch_first=True)
        self.lemma_decoder  = CopyAttentionDecoder(CHAR_VOCAB_SIZE, encoder_dim=H)
        self.suffix_decoder = SuffixSeqDecoder(CHAR_VOCAB_SIZE, encoder_dim=H)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                first_subword_mask=None, word_chars=None, word_char_lens=None,
                **kwargs):
        x = self.encoder(input_ids, attention_mask, token_type_ids).last_hidden_state

        preds = {}

        # ── Classification heads ──────────────────────────────────────
        for t in cls_tasks:
            logit    = self.heads[t](x)
            preds[t] = logit.argmax(-1)

        # ── Chunk CRF ─────────────────────────────────────────────────
        chunk_emissions = self.chunk_head(self.chunk_dropout(x))
        if first_subword_mask is not None:
            packed_emissions, _, length_mask = pack_for_crf(
                chunk_emissions, None, first_subword_mask
            )
            decoded_paths  = self.chunk_crf.decode(packed_emissions, mask=length_mask)
            preds["chunk"] = unpack_crf_preds(decoded_paths, first_subword_mask)
        else:
            preds["chunk"] = chunk_emissions.argmax(-1)

        # ── Lemma & Suffix seq2seq ────────────────────────────────────
        lemma_preds  = None
        suffix_preds = None
        if first_subword_mask is not None and word_chars is not None:
            fsw           = first_subword_mask.bool()
            hidden_at_fsw = x[fsw]
            if hidden_at_fsw.size(0) > 0:
                word_chars_fsw = word_chars[fsw]
                word_lens_fsw  = word_char_lens[fsw]

                lemma_preds = self.lemma_decoder(
                    hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=None
                )
                suffix_preds = self.suffix_decoder(
                    hidden_at_fsw, word_chars_fsw, word_lens_fsw, targets=None
                )

        return {
            "preds":        preds,
            "hidden":       x,
            "lemma_preds":  lemma_preds,
            "suffix_preds": suffix_preds,
        }


# ======================================================
# STEP 2 — BUILD & LOAD MODEL
# ======================================================
print("Loading model...")
model = MuRIL_MTL().to(device)
state = torch.load(MODEL_PATH, map_location=device)
# strict=False: safely ignores loss_fns class-weight buffers saved during training.
missing, unexpected = model.load_state_dict(state, strict=False)
if unexpected:
    print(f"  [info] Ignored {len(unexpected)} training-only keys: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
if missing:
    print(f"  [WARNING] Missing keys: {missing}")
model.eval()
print("Model loaded successfully.")

# ======================================================
# TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

# ======================================================
# PREPROCESSING
# mirrors MTLDataset.__getitem__ — inference only, no labels
# ======================================================
def preprocess(tokens):
    """
    Given a list of Telugu word tokens, return a batch dict
    ready to be passed to model.forward().
    """
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    word_ids           = enc.word_ids()
    prev               = None
    word_chars_list    = []
    word_char_lens_list = []
    first_subword_mask = []

    for wi in word_ids:
        if wi is None or wi == prev:
            word_chars_list.append([PAD_ID] * MAX_CHAR_LEN)
            word_char_lens_list.append(0)
            first_subword_mask.append(0)
        else:
            wc, wl = encode_word_chars(tokens[wi])
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


# ======================================================
# INFERENCE FOR ONE SENTENCE
# ======================================================
def predict(sentence: str):
    """
    Tokenise the sentence on whitespace, run the model,
    and return a list of dicts — one dict per word token.
    """
    tokens = sentence.strip().split()
    if not tokens:
        return []

    batch, word_ids = preprocess(tokens)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    preds        = out["preds"]         # {task: (1, MAX_LEN) tensor}
    lemma_preds  = out["lemma_preds"]   # (num_first_subwords, decode_len) or None
    suffix_preds = out["suffix_preds"]  # same

    # First-subword positions in the flat subword sequence
    fsw_mask      = batch["first_subword_mask"][0].bool()          # (MAX_LEN,)
    fsw_positions = fsw_mask.nonzero(as_tuple=True)[0].tolist()    # list of ints

    # Map word index -> label string for every classification task
    word_cls_preds = {}
    for t in all_tasks:
        pred_row = preds[t][0]   # (MAX_LEN,)
        word_cls_preds[t] = []
        for pos in fsw_positions:
            pid = pred_row[pos].item()
            word_cls_preds[t].append(id2label[t].get(pid, "[UNK]"))

    # Decode lemma and suffix character sequences
    lemma_strings  = []
    suffix_strings = []
    if lemma_preds is not None:
        for ids in lemma_preds.tolist():
            lemma_strings.append(decode_lemma_chars(ids))
    if suffix_preds is not None:
        for ids in suffix_preds.tolist():
            suffix_strings.append(decode_suffix_chars(ids))

    # Build one result dict per word token
    results = []
    for wi, tok in enumerate(tokens):
        if wi >= len(fsw_positions):
            # Sentence was truncated at MAX_LEN subwords
            break
        entry = {
            "token":  clean_telugu(tok),
            "lemma":  lemma_strings[wi]  if wi < len(lemma_strings)  else "-",
            "suffix": suffix_strings[wi] if wi < len(suffix_strings) else "-",
        }
        for t in all_tasks:
            entry[t] = word_cls_preds[t][wi] if wi < len(word_cls_preds[t]) else "[UNK]"
        results.append(entry)

    return results


# ======================================================
# DISPLAY  — uses rich.Table for proper Indic script rendering
# ======================================================
COLUMNS = ["token", "lemma", "lexcat", "pos", "gender", "num",
           "person", "case", "suffix", "tam", "chunk"]
HEADERS = ["TOKEN", "LEMMA", "LEXCAT", "POS", "GENDER", "NUM",
           "PERSON", "CASE", "SUFFIX", "TAM", "CHUNK"]

HEADER_STYLES = [
    "bold cyan",    # TOKEN
    "bold green",   # LEMMA
    "yellow",       # LEXCAT
    "magenta",      # POS
    "blue",         # GENDER
    "blue",         # NUM
    "blue",         # PERSON
    "blue",         # CASE
    "green",        # SUFFIX
    "green",        # TAM
    "bold red",     # CHUNK
]


def print_results(sentence: str, results: list):
    console.print()
    console.rule("[bold yellow]Sentence[/bold yellow]")
    console.print(sentence, style="bold white")

    console.print()
    console.rule("[bold yellow]Predictions[/bold yellow]")

    if not results:
        console.print("  (no tokens to display)", style="dim")
        console.rule()
        return

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        header_style="bold",
        expand=False,
        padding=(0, 1),
    )

    for hdr, style in zip(HEADERS, HEADER_STYLES):
        table.add_column(hdr, style=style, no_wrap=False, overflow="fold")

    for row in results:
        table.add_row(*[str(row.get(col, "-")) for col in COLUMNS])

    console.print(table)
    console.rule()


# ======================================================
# INTERACTIVE LOOP
# ======================================================
console.print("\n[bold green]Ready for inference.[/bold green]")
console.print('Type a Telugu sentence and press Enter.')
console.print('Type [bold]"exit"[/bold] or [bold]"quit"[/bold] to stop.\n')

while True:
    try:
        sentence = input("Enter Telugu sentence: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Exiting.[/yellow]")
        sys.exit(0)

    if sentence.lower() in {"exit", "quit"}:
        console.print("[yellow]Exiting.[/yellow]")
        sys.exit(0)

    if not sentence:
        continue

    results = predict(sentence)
    print_results(sentence, results)
