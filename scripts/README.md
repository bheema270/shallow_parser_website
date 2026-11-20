# Telugu Shallow Parser — Scripts Guide

This directory contains standalone CLI inference and deep learning training scripts for the Telugu Shallow Parser project, built on Google's multilingual BERT variant (**MuRIL** — Multilingual Representations for Indian Languages).

---

## Script Overview & Model Mapping

| Script Name | Target Model | Dataset Used | Key Architecture & Training Details |
| :--- | :--- | :--- | :--- |
| **`inference.py`** | V8 / V1 / V9 Checkpoints | None (Inference only) | Standalone terminal inference; loads `.pt` weights and `.json` metadata; formats predictions into rich tables. |
| **`parallel_training.py`** | **Model V8** | 213,576 raw sentences (`v_8_dataset.conllu`) | Baseline multi-task learning; linear classification heads; CRF chunking; Copy-Attention GRU decoders for Lemma/Suffix; masked UNK loss. |
| **`parallel_training_v9.py`** | **Model V9** | 121,000 deduplicated sentences (`v_9_dataset.conllu`) | Hierarchical multi-task learning; Dual Dependency Fusion (Person + LexCat → Gender; LexCat + TAM → Case); trained on all tokens including UNK. |
| **`parallel_training_v10.py`** | **Model V10** | Any Telugu CoNLL-U dataset | All V9 hierarchical architecture enhancements, but with **UNK loss masking restored** (`target == UNK` is ignored in loss) to prevent bad UNK labels on nouns. |

---

## 1. Standalone CLI Inference (`inference.py`)

`inference.py` runs fast, standalone inference directly in your terminal without needing the web server or frontend.

### What it does:
* Loads a PyTorch model checkpoint (e.g. `best_model_v8.pt`, `best_model_v1.pt`, or `best_model_v9.pt`) and its matching metadata file (`metadata_v*.json`).
* Segments input text into sentences and tokens.
* Runs the multi-task hierarchical MuRIL network.
* Uses the Copy-Attention GRU decoder for Lemma generation and Suffix Seq2Seq decoder.
* Outputs a formatted **Rich ASCII Table** directly in your terminal with all linguistic columns:
  `ID`, `Token`, `Lemma`, `POS`, `LexCat`, `Chunk`, `Person`, `Number`, `Gender`, `Case`, `Suffix`, `TAM`.

### How to configure & run:
1. Open `inference.py` and set your checkpoint and metadata paths:
   ```python
   METADATA_PATH = "metadata_v8.json"
   MODEL_PATH    = "best_model_v8.pt"
   ```
2. Run the script:
   ```bash
   python3 inference.py
   ```
3. By default, it runs the demo sentence:
   ```
   రాముడు బడికి వెళ్ళాడు.
   ```
   You can also pass custom sentences or pipe input to analyze full text blocks.

---

## 2. Baseline Model Training (`parallel_training.py` — V8)

This is the script used to train **Model V8**, the baseline model of this project.

### Architecture Highlights:
* **Encoder**: `google/muril-base-cased` (768-dimensional contextual subword representations).
* **Word Alignment**: First-subword pooling aligns BERT subwords with Telugu word tokens.
* **Classification Heads**: Separate linear projection heads for `pos`, `lexcat`, `gender`, `num`, `person`, `case`, `tam`.
* **Chunk Head**: Linear emission layer followed by a Linear-Chain **Conditional Random Field (CRF)**.
* **Character-Level Decoders**:
  * Copy-Attention GRU decoder for **Lemma** generation.
  * Seq2Seq GRU decoder for **Suffix** prediction.
* **Loss Strategy**:
  * Multi-task Cross-Entropy with log-smoothed class weights.
  * **UNK tokens were masked out** in `masked_cls_loss` (`mask = (labels != unk_id)`), so the model never learned to output `UNK`.

---

## 3. Hierarchical Full Model Training (`parallel_training_v9.py` — V9)

This script trains **Model V9** on the 121K deduplicated dataset (`v_9_dataset.conllu`).

### Improvements introduced in V9:
1. **Dual Dependency Feature Fusion**:
   * In Dravidian grammar, Gender agreement depends strongly on both Person and Lexical Category.
   * V9 projects both Person and LexCat soft embeddings into the Gender head input ($64 + 64 = 128$ dimensions), improving morpho-syntactic consistency.
2. **Context-Aware Case**:
   * Case projection receives context from both LexCat and TAM decoder states.
3. **Loss Balancing**:
   * Uses Kendall et al. (2018) uncertainty-weighted multi-task loss with learnable homoscedastic variance parameters (`log_vars`).
4. **UNK Training**:
   * In V9, `masked_cls_loss` trained on all tokens including `UNK` to represent the "not applicable" class.

---

## 4. Retraining Script with UNK Masking (`parallel_training_v10.py` — V10)

This is the updated, production-ready training script. It combines **all architectural strengths of V9** with the **clean loss masking of V8**.

### Key Differences in V10:
* **UNK Loss Masking Restored**:
  ```python
  def masked_cls_loss(self, task, logits, target):
      # V10: Tokens with target == UNK or [UNK] are masked to -100 (ignored in loss)
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
  ```
* **Why this is better**:
  Even if the dataset has nouns labeled as `Person = UNK`, V10 skips those unannotated labels and trains **only on true grammatical tags** (`1`, `2`, `3`). The model will **never predict UNK for nouns**.
* **Model Checkpoints**: Saves to `metadata_v10.json` and `best_model_v10.pt`.

---

## How to Train (HPC / Multi-GPU Execution)

Both `parallel_training_v9.py` and `parallel_training_v10.py` use PyTorch Distributed Data Parallel (`torch.distributed`).

### 1. Local / Single Server (4 GPUs):
```bash
torchrun --nproc_per_node=4 parallel_training_v10.py
```

### 2. ADA HPC Cluster / SLURM Batch Job:
Create or edit your SLURM submission script (`run_ada.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=telugu_v10
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=run10_%j.log

source activate myenv
torchrun --nproc_per_node=4 parallel_training_v10.py
```
Submit the job:
```bash
sbatch run_ada.sh
```
Monitor live training logs:
```bash
tail -f run10_<job_id>.log
```
