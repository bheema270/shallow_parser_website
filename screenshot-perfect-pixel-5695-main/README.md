# Telugu Shallow Parser & Morphological Analyzer (Web Application)

An end-to-end web platform and REST API for high-performance shallow parsing and morphological analysis of Telugu sentences and documents, powered by a fine-tuned **MuRIL Multi-Task Learning (MTL) Hierarchical model**.

---

## 📌 Project Overview

This application provides linguistic annotation for Telugu text:
- **POS Tagging** (Part-of-Speech: NNP, NN, VM, etc.)
- **Chunking / Phrase Boundary Identification** (B-NP, I-NP, B-VGF, etc. via CRF)
- **Morphological Analysis**:
  - Lemma generation (via Copy-Attention Seq2Seq Decoder)
  - Suffix extraction (via Attention GRU Decoder)
  - TAM (Tense-Aspect-Modality) prediction (via Attention GRU Decoder)
  - Person, Number, Gender, Case (Nominal & Oblique)
- **Export Format**: Standard **CoNLL-U** tab-separated format with one-click copy and file download.
- **Dynamic Multi-Model Support**: Automatically detects and lets users switch between different trained models (`v9`, `v1`, `v8`) directly from the web interface.

---

## 🧠 Available Models & Evaluation Guide

The platform includes **three distinct trained models**, allowing users and researchers to choose the best model for their needs based on evaluation metrics:

| Model ID | Model Name | Role / Status | Dataset Used | Sentence Count | Recommended Use |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`v9`** | **MuRIL MTL V9** | **Finalized Production Model** *(Advisor / Ma'am Architecture)* | `whole_dataset.conllu` *(Deduplicated)* | **121,823** | **Default & Recommended**: Best overall accuracy across all morphological features (Person, Gender, Case, Number). |
| **`v1`** | **MuRIL MTL V1** | **Curated & Verified Subset Model** *(Ma'am Architecture)* | `all_combined_set.conllu` *(Sets 1-7)* | **32,006** | **High-Precision Subset**: Uses the new architecture trained on high-quality, in-person verified annotations. |
| **`v8`** | **MuRIL MTL V8** | **Baseline / Legacy Dependency Model** *(Pre-Ma'am Architecture)* | `whole_dataset_past.conllu` *(Raw / Pre-dedup)* | **213,576** | **Legacy Baseline**: Previous finalized model before Ma'am's architecture changes; strong on POS, Chunk, Lemma & TAM. |

---

### Detailed Model Architecture & Training Differences

#### 1. `MuRIL MTL V9` — Finalized Production Model (Recommended)
- **Architecture**: Incorporates the complete updated hierarchical architecture advised by our project guide (Ma'am):
  - **Dual Dependency for Gender**: Gender classification head receives fused embeddings from **both Person ($64\text{d}$) and LexCat ($64\text{d}$)** ($64+64=128\text{d}$), eliminating gender ambiguity.
  - **UNK Loss Masking Removed**: Model actively learns all valid "not applicable" / UNK labels instead of masking them out during training.
  - **Inference-Time LexCat Feature Masking**: Non-inflected word classes (punctuation, conjunctions, adverbs) are linguistically constrained to UNK.
- **Training Data**: Full deduplicated corpus of **121,823 sentences** (reduced from the original 213K duplicates).
- **Evaluation Log**: `docs/evaluation/run_muril_v5_v9.txt`

#### 2. `MuRIL MTL V1` — In-Person Verified Subset Model
- **Architecture**: Identical code and architecture as V9 (Dual dependency, UNK training, LexCat masking).
- **Training Data**: Specially curated multi-set corpus of **32,006 sentences** (`all_combined_set.conllu` combining Sets 1, 2, 4, 5, 6, 7).
- **Key Advantage**: Annotations in this dataset were thoroughly reviewed and verified in-person, providing a cleaner, highly reliable training distribution.
- **Evaluation Log**: `docs/evaluation/run_muril_v5_v1.txt`

#### 3. `MuRIL MTL V8` — Baseline / Legacy Dependency Model
- **Architecture**: The original finalized model developed before incorporating Ma'am's architectural recommendations:
  - Gender head depended only on Person ($64\text{d}$).
  - UNK tokens were masked out of loss calculations during training.
- **Training Data**: Original raw corpus of **213,576 sentences** (`whole_dataset_past.conllu`).
- **Characteristics**: Exceptional performance on POS (98.39%), Chunking (97.04%), Lemma (96.38%), and TAM (98.03%), but suffered from low Person (19.13%) and Case (0.30%) due to UNK masking (which was solved in V9 and V1).
- **Evaluation Log**: `docs/evaluation/run_muril_v5_v8.txt`

---

### Side-by-Side Test Set Evaluation Metrics

Extracted directly from the respective evaluation logs in `docs/evaluation/`:

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

## 🛠️ System Prerequisites

Before starting, ensure your system has:
1. **Python**: Version `3.10` or higher
2. **Node.js**: Version `18.x` or higher (with `npm`)

---

## 🚀 Quick Setup & Installation

If you received this project as a zip file, extract it and follow these steps:

### 1. Backend Setup (FastAPI & PyTorch)

Navigate to the `backend/` folder:
```bash
cd backend
```

Create and activate a Python virtual environment:
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt / PowerShell):
python -m venv venv
venv\Scripts\activate
```

Install all required Python packages with a single command:
```bash
pip install -r requirements.txt
```

> **Note**: If you want PyTorch with GPU (CUDA) acceleration, install the PyTorch build matching your CUDA version from [pytorch.org](https://pytorch.org/), otherwise it runs automatically on CPU.

---

### 2. Frontend Setup (React & Vite)

Open a new terminal window, navigate to the web project root:
```bash
cd screenshot-perfect-pixel-5695-main
```

Install frontend dependencies:
```bash
npm install
```

---

## 🖥️ Running the Application

### Step 1: Start the Backend Server

From the `backend/` directory (with virtual environment activated):
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
- The backend API will start at: `http://127.0.0.1:8000`
- Interactive API documentation (Swagger): `http://127.0.0.1:8000/docs`

### Step 2: Start the Frontend UI

From the project root directory (`screenshot-perfect-pixel-5695-main/`):
```bash
npm run dev
```
- The web interface will open at: **`http://localhost:5173/shallow_parser`**
- *(Note: Accessing the root URL `http://localhost:5173/` automatically redirects to `/shallow_parser`)*

---

## 🚀 Production Deployment with PM2

When deploying on a shared server, VM, or production environment using **pm2**:

### 1. Build the Frontend
```bash
npm run build
```

### 2. Start with PM2
```bash
# Start Backend
pm2 start "venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000" --name "shallow-parser-api" --cwd backend

# Start Frontend (or Vite preview / SSR server)
pm2 start "npm run preview -- --port 5173 --host 0.0.0.0" --name "shallow-parser-web"

# Save pm2 process list to auto-restart on system reboot
pm2 save
```

Once running:
- Open the application at: **`http://<server-ip-or-domain>:5173/shallow_parser`**
- All API and UI traffic properly resolves with the `/shallow_parser` path.

---

## 📁 How to Add a New Model in the Future

The system features **automatic model discovery**. To add a new trained model:

1. Place your PyTorch checkpoint and metadata JSON in `backend/models/`:
   - Checkpoint: `best_model_v1.pt` (or `<name>.pt`)
   - Metadata: `metadata_v1.json` (or `<name>.json`)
2. That's it!
   - The backend automatically detects the model upon startup or page reload.
   - The frontend dropdown will immediately list **`MuRIL MTL V1`** alongside existing models.
   - When selected, inference runs specifically on that model.

---

## 📂 Project Structure

```
screenshot-perfect-pixel-5695-main/
├── backend/
│   ├── models/                  # Stores trained .pt checkpoints and metadata .json
│   │   ├── best_model_v9.pt     # Finalized production model (Ma'am architecture, 121K)
│   │   ├── metadata_v9.json
│   │   ├── best_model_v1.pt     # In-person verified subset model (32K sentences)
│   │   ├── metadata_v1.json
│   │   ├── best_model_v8.pt     # Legacy baseline model (213K sentences)
│   │   └── metadata_v8.json
│   ├── config.py                # Server, CORS, and paths configuration
│   ├── model_registry.py        # Automatic discovery and registration of models
│   ├── inference_engine.py      # MuRIL model architecture, decoders, and IndicNLP tokenizer
│   ├── formatter.py             # Formats output into CoNLL-U format
│   ├── main.py                  # FastAPI REST endpoints (/api/analyze, /api/models, etc.)
│   └── requirements.txt         # Python dependencies
├── src/
│   ├── components/parser/       # Header, InputSection, OutputSection
│   ├── hooks/                   # useAnalyze (API polling + cancellation), useModels
│   ├── config.ts                # API URL and table columns configuration
│   ├── routes/index.tsx         # Main page layout and state management
│   └── types/parser.ts          # TypeScript interfaces
├── package.json                 # Frontend dependencies and npm scripts
└── README.md                    # Project documentation
```
