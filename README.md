# Telugu Shallow Parser & Morphological Analyzer Project

This folder contains the complete trained models, research documentation, standalone scripts, and the full-stack web application for the Telugu Shallow Parser project.

---

## 📂 Folder Contents

```
final_output_oc1/
├── README.md                              # This file (project overview & model guide)
│
├── screenshot-perfect-pixel-5695-main/    # Full Web Application (Frontend + Backend)
│   ├── backend/
│   │   ├── models/                        # Available trained models (.pt & .json)
│   │   │   ├── best_model_v9.pt           # Finalized production model (Ma'am architecture, 121K)
│   │   │   ├── metadata_v9.json
│   │   │   ├── best_model_v1.pt           # In-person verified subset model (32K sentences)
│   │   │   ├── metadata_v1.json
│   │   │   ├── best_model_v8.pt           # Legacy baseline model (213K sentences)
│   │   │   └── metadata_v8.json
│   │   ├── main.py                        # FastAPI REST API with multi-model support
│   │   ├── inference_engine.py            # High-performance MuRIL inference & IndicNLP
│   │   ├── model_registry.py              # Automatic model discovery
│   │   ├── formatter.py                   # CoNLL-U format generator
│   │   └── requirements.txt               # Complete Python package list
│   ├── src/                               # React + Vite + TypeScript web interface
│   ├── README.md                          # Detailed web application setup instructions
│   └── USER_GUIDE.md                      # Complete user guide and annotation manual
│
├── docs/                                  # Research documentation and evaluation logs
│   ├── evaluation/                        # Training & evaluation run logs for each model
│   │   ├── run_muril_v5_v9.txt            # V9 evaluation log (121K deduplicated sentences)
│   │   ├── run_muril_v5_v1.txt            # V1 evaluation log (32K verified subset)
│   │   └── run_muril_v5_v8.txt            # V8 evaluation log (213K legacy sentences)
│   ├── COMPLETE_BEGINNER_TO_EXPERT_CODE_STORY.md
│   ├── DETAILED_CODE_AND_ARCHITECTURE_EXPLANATION.md
│   └── presentation_btp_update.tex
│
└── scripts/                               # Standalone CLI and training scripts
    ├── inference.py                       # CLI inference tool
    ├── parallel_training.py               # Original training code
    └── parallel_training_v9.py            # Enhanced training code
```

---

## 🧠 Models Overview & Evaluation Guide

The platform integrates three distinct models which can be selected dynamically from the website dropdown:

### 1. `MuRIL MTL V9` — Finalized Production Model (Recommended)
- **Status**: The finalized model adopting the hierarchical multi-task architecture advised by our project advisor (Ma'am).
- **Architecture Highlights**:
  - **Dual Dependency for Gender**: Gender head receives fused representations from **both Person ($64\text{d}$) and LexCat ($64\text{d}$)** ($128\text{d}$ total), resolving ambiguity.
  - **Active UNK Training**: Removes UNK loss masking, allowing the model to properly learn "not applicable" cases.
  - **Inference-Time LexCat Feature Masking**: Non-inflected word types (punctuation, symbols, adverbs) are linguistically masked to UNK.
- **Dataset**: Trained on the clean **deduplicated corpus of 121,823 sentences** (`whole_dataset.conllu`).
- **Evaluation Log**: [`docs/evaluation/run_muril_v5_v9.txt`](file:///home/bheemasiddartha/Desktop/sem6/BTP/6_lakhs/final_output_oc1/docs/evaluation/run_muril_v5_v9.txt)

### 2. `MuRIL MTL V1` — In-Person Verified Subset Model
- **Status**: Curated subset model using the exact same Ma'am architecture as V9.
- **Dataset**: Trained on a focused combined dataset of **32,006 sentences** (`all_combined_set.conllu` across Sets 1, 2, 4, 5, 6, 7).
- **Key Advantage**: All annotations were verified and checked in-person for consistency and high linguistic fidelity.
- **Evaluation Log**: [`docs/evaluation/run_muril_v5_v1.txt`](file:///home/bheemasiddartha/Desktop/sem6/BTP/6_lakhs/final_output_oc1/docs/evaluation/run_muril_v5_v1.txt)

### 3. `MuRIL MTL V8` — Baseline / Legacy Dependency Model
- **Status**: The primary finalized model before incorporating Ma'am's architectural revisions.
- **Architecture**: Single dependency hierarchy where Gender depended only on Person ($64\text{d}$), and UNK tokens were masked out of loss computation during training.
- **Dataset**: Trained on the original raw dataset of **213,576 sentences** (`whole_dataset_past.conllu`).
- **Characteristics**: Strong on POS (98.39%), Chunking (97.04%), Lemma (96.38%), and TAM (98.03%), but suffered from lower accuracy on Person and Case due to UNK masking (which was fixed in V9 and V1).
- **Evaluation Log**: [`docs/evaluation/run_muril_v5_v8.txt`](file:///home/bheemasiddartha/Desktop/sem6/BTP/6_lakhs/final_output_oc1/docs/evaluation/run_muril_v5_v8.txt)

---

### Side-by-Side Test Set Evaluation Metrics

| Task / Feature | V9 (Finalized 121K) | V1 (Verified 32K) | V8 (Baseline 213K) |
| :--- | :---: | :---: | :---: |
| **POS Tagging** | **97.35%** | 95.59% | **98.39%** |
| **LexCat (Lexical Category)** | **98.65%** | 97.88% | **99.18%** |
| **Person** | **99.38%** | 96.51% | 19.13% |
| **Number** | **98.73%** | 97.07% | 60.62% |
| **Gender** | **97.18%** | 95.41% | 52.43% |
| **Case** | **96.52%** | 94.48% | 0.30% |
| **Chunking (CRF)** | **94.41%** | 91.22% | **97.04%** |
| **Lemma (Exact Match)** | **93.47%** | 83.71% | **96.38%** |
| **Suffix (Exact Match)** | **96.58%** | 91.65% | **98.14%** |
| **TAM (Exact Match)** | **96.06%** | 85.84% | **98.03%** |

---

## 📜 Scripts Directory Guide (`scripts/`)

The `scripts/` directory contains standalone scripts for model training and CLI evaluation:

| Script File | Purpose & Role | Model Generated | Dataset Used |
| :--- | :--- | :--- | :--- |
| **`parallel_training_v9.py`** | **Modern Training Script (Ma'am Architecture)**<br>Implements dual dependency for Gender ($E+E=128\text{d}$), active UNK training, and inference-time LexCat feature masking. | **`best_model_v9.pt`**<br>&<br>**`best_model_v1.pt`**<br>*(same script code)* | • **V9**: `whole_dataset.conllu` (121,823 deduplicated sentences)<br>• **V1**: `all_combined_set.conllu` (32,006 in-person verified sentences) |
| **`parallel_training.py`** | **Legacy Training Script (Baseline Model)**<br>Original hierarchical multi-task script where Gender depended only on Person ($E=64\text{d}$) and UNK loss masking was active. | **`best_model_v8.pt`** | `whole_dataset_past.conllu` (213,576 raw sentences) |
| **`inference.py`** | **Standalone CLI Inference Tool**<br>Direct terminal evaluation tool that runs predictions on Telugu sentences and prints styled ASCII/Rich tables without starting the web server. | — *(Evaluates using `best_model_v8.pt`)* | — |

> **Important Note on V9 vs. V1 Generation**:
> Both **`best_model_v9.pt`** and **`best_model_v1.pt`** were trained using the **exact same training code (`parallel_training_v9.py`)**. The only difference between them is the dataset passed to `DATA_FILE`:
> - **V9** was trained on the full deduplicated dataset (`whole_dataset.conllu` — 121,823 sentences).
> - **V1** was trained on the human/in-person verified combined dataset (`all_combined_set.conllu` — 32,006 sentences).

---

## 🚀 How to Run the Website (Quick Start)

### 1. Setup & Run the Backend

```bash
cd screenshot-perfect-pixel-5695-main/backend

# Create and activate a Python virtual environment:
python3 -m venv venv
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt

# Start the server:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be live at `http://127.0.0.1:8000`.

### 2. Setup & Run the Frontend

In a second terminal window:
```bash
cd screenshot-perfect-pixel-5695-main

# Install frontend dependencies:
npm install

# Start development server:
npm run dev
```

Open **`http://localhost:5173/shallow_parser`** in your browser.
*(Accessing the root URL `http://localhost:5173/` automatically redirects to `/shallow_parser`)*

---

## 🔮 Adding New Models in the Future

Whenever a new model is trained (for example, `best_model_v1.pt` trained on all combined datasets):
1. Copy `best_model_v1.pt` and `metadata_v1.json` into:
   `screenshot-perfect-pixel-5695-main/backend/models/`
2. Restart or reload the website.
3. The new model will automatically appear in the **Model** dropdown selector!
