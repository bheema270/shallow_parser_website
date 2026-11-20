# User & Architecture Guide — Telugu Shallow Parser

This document explains what this website does, how to use all its features, how output annotations work, and how edge cases and multiple models are handled.

---

## 1. What is this Website?

The **Telugu Shallow Parser** is an NLP tool that analyzes Telugu text at both the morphological (word-internal structure) and syntactic (phrase-level structure) levels.

Unlike traditional pipeline tools where errors compound at each stage, this website is powered by a **Multi-Task Learning (MTL) Hierarchical Neural Network** based on Google's multilingual BERT model (**MuRIL**). It predicts Part-of-Speech, phrase chunks, root lemmas, case markers, and TAM suffixes jointly in a single pass.

---

## 2. Web Interface Features & How to Use

### 2.1 Entering Input Text
- You can enter a **single sentence** or a **full paragraph / document**.
- To test quickly, click the **"Try example"** link to load a sample Telugu sentence:
  ```
  రాముడు బడికి వెళ్ళాడు
  ```
- **Keyboard Shortcut**: Press `Ctrl + Enter` (or `Cmd + Enter` on Mac) inside the text box to trigger analysis immediately.

### 2.2 Selecting the Model
- Use the **Model** dropdown in the toolbar to pick which trained model checkpoint executes your analysis:
  - `MuRIL MTL V8 (Default)`
  - Future models added to `backend/models/` (e.g. `MuRIL MTL V1`) appear here automatically.
- The output table displays a badge confirming which model produced the results.

### 2.3 Output Format
- Currently defaults to **CoNLL-U** (Universal Dependencies tab-separated format).
- Displays both an interactive **Formatted Table** view and a **Raw Output** view.

### 2.4 Navigation by Sentence
- When analyzing a multi-sentence paragraph, a dropdown menu appears:
  - **Whole document**: Shows all tokens sequentially across the entire text.
  - **Individual Sentences**: Each option displays the sentence number, a preview of the first 1–2 words, and the token count:
    ```
    Sentence 1: "రాముడు అడవికి..." (4 tokens)
    Sentence 2: "సీత లక్ష్మణుడు..." (5 tokens)
    ```

### 2.5 Halting / Cancelling an Analysis
- If you submit a very long document or wish to stop execution, an active **"Halt / Cancel"** button appears beside the loading spinner.
- Clicking it immediately notifies the server to stop computation and aborts waiting without freezing the webpage.

### 2.6 Exporting Results
- **Copy All / Copy Sentence**: Copies the CoNLL-U text to your clipboard.
- **Download Complete CoNLL-U**: Downloads a `.conllu` file named `complete-output.conllu` (or `sentence-<id>.conllu` if viewing an individual sentence).

---

## 3. CoNLL-U Annotation Columns Explained

The output table provides 12 linguistic attributes for each token:

| Column | Name | Description | Example |
| :--- | :--- | :--- | :--- |
| **ID** | Token Index | Index of the word within the sentence (starts at 1) | `1`, `2` |
| **Token** | Word Form | The surface Telugu token as it appears in the text | `రాముడు`, `బడికి` |
| **Lemma** | Base Root | Root form predicted by the seq2seq copy-decoder | `రాముడు`, `బడి` |
| **POS** | Part-of-Speech | Coarse grammatical category (NN, NNP, VM, VAUX, etc.) | `NNP`, `NN`, `VM` |
| **LexCat** | Lexical Category | Fine-grained lexical class (n, v, adj, adv, punc, etc.) | `n`, `v` |
| **Chunk** | Phrase Chunk | BIO-tagged syntactic chunk from the CRF layer | `B-NP`, `B-VGF` |
| **Person** | Person | Grammatical person: `1` (first), `2` (second), `3` (third) | `3` |
| **Number** | Number | Grammatical number: `sg` (singular), `pl` (plural) | `sg` |
| **Gender** | Gender | Gender marker: `m` (masculine), `f` (feminine), `n` (neuter) | `m` |
| **Case** | Case Marker | `d` (direct / nominative) or `o` (oblique / governed by postposition) | `d`, `o` |
| **Suffix** | Inflexion Suffix | Suffix characters attached to the root | `-`, `కి` |
| **TAM** | Tense-Aspect-Modality | Predicted verbal inflection (e.g., `_`, `past`, `pres`, etc.) | `_` |

---

## 4. Error Handling & Edge Cases

The system is built to handle edge cases gracefully:
- **Empty Input**: The "Analyze" button is disabled; pressing `Ctrl + Enter` displays a prompt to enter text.
- **Backend Offline**: If the FastAPI server is not running or unreachable, an alert banner states: *"Unable to reach the server. Please ensure the backend is running."*
- **Model Loading Failure**: If a corrupted checkpoint or missing metadata file is selected, the server returns a detailed error message and stops the spinner immediately.
- **Very Long Paragraphs**: IndicNLP segments sentences on natural boundaries, preventing token overflow while processing sequential sentence windows efficiently.
