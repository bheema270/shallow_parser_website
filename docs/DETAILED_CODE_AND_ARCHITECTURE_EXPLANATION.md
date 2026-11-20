# Telugu Shallow Parser (V8): Architecture, Design & Code Walkthrough

This document provides a comprehensive, step-by-step narrative and technical explanation of the **Telugu Shallow Parser (V8 Hierarchical Multi-Task Architecture)**. It explains the linguistic motivation, the precise execution flow, each mathematical and architectural choice, every algorithm implemented, and how edge cases (especially unknown tokens `[UNK]`) are handled.

---

## 1. The Story & Motivation: Why a Hierarchical Architecture?

### 1.1 The Linguistic Nature of Telugu
Telugu is a Dravidian language characterized by **rich agglutinative morphology** and relatively flexible word order:
- A single Telugu word often consists of a lexical root (lemma) joined with multiple suffixes marking **case**, **tense**, **aspect**, **mood (TAM)**, **person**, **number**, and **gender (PNG)**.
- For example, in the word *వెళ్ళాడు* (*veḷḷāḍu* - "he went"):
  - Root (Lemma): *వెళ్లు* (*veḷḷu* - "go")
  - Lexical Category (LexCat): Verb (*v*)
  - Part of Speech (POS): Main Finite Verb (*VM*)
  - Person: 3rd (*3*)
  - Number: Singular (*sg*)
  - Gender: Masculine (*m*)
  - Suffix / TAM: Past tense (*ఆ* / *ాడు*)

### 1.2 The Dilemma: Flat vs. Hierarchical
1. **Flat Baseline (V6)**: All morphological tags were predicted independently straight from MuRIL representations ($H = 768$). While robust, it disregarded fundamental linguistic dependency rules (e.g., predicting Person/Number for words that aren't nouns/verbs).
2. **Naive Hierarchical Approach (V7)**: Passing hard `argmax` decisions between heads caused **cascading error propagation** (if POS is mispredicted, LexCat is corrupted, which then corrupts Person, Number, Gender, and Case). Furthermore, artificial logit masking exploded validation loss.
3. **The V8 Resolution**: V8 preserves the strict linguistic hierarchy while solving error propagation through **probabilistic soft expectation embeddings**, **confidence gating**, **residual fusion**, **UNK suppression**, and **uncertainty-weighted loss balancing**.

---

## 2. The Execution Flow: The Journey of a Telugu Sentence

When a Telugu sentence (e.g., `"రాముడు బడికి వెళ్ళాడు"`) enters the shallow parser, it progresses through five sequential stages:

```
[Input Telugu Sentence]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 0: Preprocessing & Multi-Granular Tokenization   │
│  - MuRIL WordPiece subwords (max length 128)           │
│  - Word-level character grids (max length 50/60)       │
│  - First-subword boolean mask                          │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Contextual Representation via MuRIL Encoder   │
│  - Base: google/muril-base-cased                       │
│  - Layers 1-6: Frozen (prevents catastrophic forgetting)│
│  - Output: Contextual word embeddings (768 dims)       │
└────────────────────────────────────────────────────────┘
          │
          ├─────────────────────────┬─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌───────────────────────────┐
│ STAGE 2A: Chunk  │      │ STAGE 2B: Lemma  │      │ STAGE 2C: Suffix & TAM    │
│ Linear + CRF     │      │ CopyAttention    │      │ SuffixSeqDecoder (GRU)    │
│ (BIO tag paths)  │      │ Decoder (Seq2Seq)│      │ Suffix chars + TAM chars  │
└──────────────────┘      └──────────────────┘      │ Extract TAM hidden state  │
                                                    │ (256-dim dense vector)    │
                                                    └─────────────┬─────────────┘
                                                                  │
          ┌───────────────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Cascading Hierarchical Classification Chain                          │
│                                                                               │
│  [MuRIL Context] ──► (1) POS Head ──► POS Logits                              │
│                            │                                                  │
│               Soft Gated Emb (POS)                                            │
│                            ▼                                                  │
│  [MuRIL Context] ──► (2) LexCat Head ──► LexCat Logits                        │
│                            │                                                  │
│               Soft Gated Emb (LexCat)                                         │
│               ┌────────────┴────────────┐                                     │
│               ▼                         ▼                                     │
│  [MuRIL] ──► (3) Person Head    [MuRIL] ──► (4) Number Head                   │
│     │            │                                                            │
│     │  Soft Gated Emb (Person)                                                │
│     │            ▼                                                            │
│     └──────► (5) Gender Head                                                  │
│                                                                               │
│  [MuRIL Context] + Soft Emb (LexCat) + TAM Hidden (256-dim) ──► (6) Case Head │
└───────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 4: Unicode NFC Normalization & Output Formatting │
│  - Clean duplicate diacritics                          │
│  - Format as Tab-separated CoNLL-U or JSON records     │
└────────────────────────────────────────────────────────┘
```

---

## 3. Deep Dive into Every Method & Algorithm

### 3.1 Preprocessing: Multi-Granular Alignment
Telugu words are morphologically rich, meaning word-level tokens, subwords, and characters must all stay synchronized:
- **MuRIL Tokenizer**: Segments words into subwords. Because evaluation happens at the word level, we construct a `first_subword_mask`. All token-level classification decisions and seq2seq decoders are aligned strictly to the **first subword** of each Telugu word.
- **Character Encoding (`_encode_word_chars`)**: Maps individual Unicode characters of each word to integer indices for character-level decoders, padding or truncating to `MAX_CHAR_LEN = 50`.

---

### 3.2 Sequence Decoders: Generating Lemma, Suffix, and TAM

#### Algorithm 1: Pointer-Generator Copy Mechanism (`CopyAttentionDecoder` for Lemma)
Lemmas in Telugu often share most of their characters with the surface word (e.g., *బడికి* $\rightarrow$ *బడి*). A standard decoder might misspell rare characters.
- **Encoder**: Bidirectional GRU ($128 \times 2 = 256$ dimensions) encodes word characters.
- **Attention**: Computes Bahdanau additive attention over the word's character representations:
  $$\text{score}(h_{dec}, h_{enc}) = v^T \tanh(W h_{enc} + U h_{dec})$$
- **Generation vs. Copy Probability ($p_{gen}$)**:
  $$p_{gen} = \sigma(W_c [h_{dec}; \text{context}; \text{emb}_{char}])$$
- **Final Output Distribution**:
  $$P(c) = p_{gen} P_{vocab}(c) + (1 - p_{gen}) \sum_{i: c_i = c} \text{attn\_weights}_i$$
  If a character exists in the source word, the model can directly copy it with zero vocabulary generation error.

#### Algorithm 2: Attentive Character Generation (`SuffixSeqDecoder` for Suffix & TAM)
Suffixes and TAM markers are grammatical abstractions (e.g., `<UNK>i`, `ఆ`, `0`). 
- Uses an attentive GRU decoder to generate the character sequence.
- **Dual Output**: For TAM, `forward()` returns both the predicted character sequence **and** the final decoder hidden state $h \in \mathbb{R}^{256}$. This 256-dimensional vector contains a dense summary of tense, aspect, and mood, which is directly piped into the downstream **Case** head.

---

### 3.3 The Hierarchical Cascading Pipeline

#### The Soft Dependency Embedding (`_soft_dependency_embedding`)
In standard cascades, one passes `emb(argmax(upstream_logits))`. This causes two critical flaws:
1. It is non-differentiable.
2. If upstream accuracy is 90%, 10% of downstream inputs receive completely false information with 100% confidence.

In V8, we use **expectation embeddings**:
1. Take softmax probabilities over upstream classes:
   $$p = \text{Softmax}(\text{logits}) \in \mathbb{R}^{C}$$
2. Compute the expected embedding as a weighted sum over the embedding matrix $W_{emb} \in \mathbb{R}^{C \times E}$:
   $$e_{soft} = p \cdot W_{emb} \in \mathbb{R}^{E}$$
   When the upstream model is uncertain between two tags, downstream heads receive a smooth mixture of both hypotheses rather than an arbitrary wrong choice.

#### Confidence Gating & Dynamic Trust (`dep_alpha`)
Even with soft embeddings, if the model has no idea what the upstream tag is, passing noisy features can hurt downstream predictions.
- We compute confidence: $\text{conf} = \max(p)$.
- If $\text{conf} < 0.30$ (`CONF_THRESHOLD`), the signal is treated as uninformative.
- Each dependency connection has a learnable trust parameter $\alpha$:
  $$\text{gate} = \sigma(\alpha) \cdot \text{conf}$$
  $$e_{final} = \text{Dropout}(e_{soft} \cdot \text{gate})$$
  The network automatically learns how much downstream heads should rely on upstream predictions.

---

### 3.4 How Unknown Tokens (`[UNK]`) Are Handled

Unknown tokens (`[UNK]` or `UNK`) can distort both training gradients and evaluation metrics if left unhandled. V8 applies a three-layer defense:

#### 1. Neutral Embedding Vector (`neutral_emb`) for Dependency Signals
- If the upstream head predicts `[UNK]` or `UNK`, or if prediction confidence falls below 30%, the model does not look up an embedding.
- Instead, it substitutes a dedicated, learnable parameter vector:
  $$\text{neutral\_emb} \in \mathbb{R}^{64}$$
- This guarantees that an unknown tag at the POS level acts as a zero-bias neutral prior, allowing the downstream head to rely purely on MuRIL's base representation rather than being corrupted by random weights.

#### 2. Loss Masking on Gold `[UNK]` (`masked_cls_loss`)
In the Telugu corpus, certain tokens are labeled `[UNK]` where annotators were uncertain.
- In standard Cross-Entropy, the model expends gradient capacity learning to classify words into `[UNK]`.
- In V8, `masked_cls_loss` identifies target labels matching `[UNK]` or `UNK` and replaces them with `-100` (`ignore_index`):
  ```python
  if unk_id is not None:
      target[target == unk_id] = -100
  ```
  These tokens produce **zero loss and zero gradient**, preventing the model from treating `[UNK]` as an easy shortcut class.

#### 3. UNK Class-Weight Damping
When class weighting is applied to counteract class imbalance:
- Rare classes receive higher weights. Because `[UNK]` is infrequent in some features, a standard inverse-frequency formula would assign it a massive weight.
- V8 dampens UNK weights:
  ```python
  for unk_key in ['[UNK]', 'UNK']:
      if unk_key in label2id[t]:
          task_weights[t][uid] = min(task_weights[t][uid], 0.5)
  ```
  This prevents rare unknown tokens from dominating the loss surface.

---

### 3.5 The `FusionHead` Architecture
In V7, upstream features were concatenated directly into a single linear layer: $\text{Linear}(H + E, C)$. This caused the large 768-dim MuRIL vector and the 64-dim dependency vector to compete, often diluting the MuRIL features.

In V8, every cascading head is an explicit `FusionHead`:
1. **MuRIL Residual Highway**: MuRIL contextual vector $x \in \mathbb{R}^{768}$ is the primary backbone.
2. **Projected Dependency Addition**: The dependency embedding $e \in \mathbb{R}^{E}$ is projected to 768 dimensions via a learned projection layer $W_{proj}$ with Xavier initialization and added as a residual:
   $$x_{fused} = x + W_{proj}(e)$$
3. **Stabilization**:
   $$h = \text{Dropout}(\text{GELU}(\text{LayerNorm}(x_{fused})))$$
   $$\text{logits} = W_{out} h$$
This ensures MuRIL representation is never degraded, while dependency information acts as an informative inductive bias.

---

### 3.6 Multi-Task Loss Balancing (Kendall Uncertainty Formulation)
Training 10 distinct tasks (6 classification, 1 CRF, 3 Seq2Seq) with static loss weights is notoriously unstable.

V8 implements homoscedastic uncertainty weighting (Kendall et al., 2018):
- Each task $t$ has a learnable parameter $s_t = \log(\sigma_t^2)$ (`log_vars[t]`):
  $$\mathcal{L}_{total} = \sum_{t} \left( \frac{1}{2 \exp(s_t)} \mathcal{L}_t + \frac{1}{2} s_t \right)$$
- If a task is noisy or difficult early in training, the model increases $s_t$, automatically downweighting that task's gradient while incurring a regularization penalty $\frac{1}{2} s_t$.
- As training stabilizes, the model lowers $s_t$ to refine precision across all tasks simultaneously without manual tuning.

---

### 3.7 Output Unicode Cleaning (`clean_telugu`)
Indic complex scripts render with combining vowel marks (*matras*) and viramas (*halant*). Greedy decoders can occasionally predict duplicate consecutive combining marks (e.g., `ుు` instead of `ు`).
- `clean_telugu` applies Unicode **NFC normalization**.
- It iterates through Unicode character categories, dropping consecutive duplicate spacing combining marks (`Mc`) and non-spacing marks (`Mn`), ensuring that output strings render correctly in standard browsers and terminal fonts.

---

## 4. Summary Table of Tasks & Flows

| Order | Task | Mechanism | Dependency Input | Target Type |
|---|---|---|---|---|
| **Independent** | **Chunk** | Linear + Linear-Chain CRF | None (MuRIL $H$) | Sequence BIO tags |
| **Independent** | **Lemma** | GRU + Pointer-Generator Copy | None (MuRIL $H$ + Chars) | Character Sequence |
| **Independent** | **Suffix** | Attentive GRU Decoder | None (MuRIL $H$ + Chars) | Character Sequence |
| **Independent** | **TAM** | Attentive GRU Decoder | None (MuRIL $H$ + Chars) | Character Sequence + Hidden state $h \in \mathbb{R}^{256}$ |
| **Stage 1** | **POS** | FusionHead (dep=0) | None (MuRIL $H$) | Classification (24 tags) |
| **Stage 2** | **LexCat** | FusionHead (dep=64) | Soft Gated POS Embedding | Classification (10 tags) |
| **Stage 3** | **Person** | FusionHead (dep=64) | Soft Gated LexCat Embedding | Classification (6 tags) |
| **Stage 3** | **Number** | FusionHead (dep=64) | Soft Gated LexCat Embedding | Classification (5 tags) |
| **Stage 4** | **Gender** | FusionHead (dep=64) | Soft Gated Person Embedding | Classification (8 tags) |
| **Stage 5** | **Case** | FusionHead (dep=320) | Soft Gated LexCat ($64$) + TAM Hidden ($256$) | Classification (4 tags) |

---

## 5. Summary
By combining:
1. **Residual feature fusion** (`FusionHead`)
2. **Soft, confidence-gated dependency embeddings**
3. **Learnable UNK suppression & loss masking**
4. **Kendall uncertainty loss balancing**
5. **Character-level copy and attentive decoders**

The V8 architecture preserves the linguistically grounded cascading flow while achieving maximum stability, high accuracy, and clean deployment readiness for the Telugu Shallow Parser web interface.
