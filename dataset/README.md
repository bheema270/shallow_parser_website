# Telugu Shallow Parser Datasets

This folder contains the three primary CoNLL-U format datasets used across the project lifecycle:

| File Name | Size | Sentence Count | Trained Model | Description |
| :--- | :---: | :---: | :---: | :--- |
| **`v_9_dataset.conllu`** | **59 MB** | **121,823** | **`best_model_v9.pt`** | **Deduplicated Production Dataset**: The original 213K raw corpus with all exact duplicate sentences removed. Used to train the finalized production V9 model. |
| **`v_1_dataset.conllu`** | **31 MB** | **32,006** | **`best_model_v1.pt`** | **In-Person Verified Combined Dataset**: Curated multi-set dataset combining Sets 1, 2, 4, 5, 6, and 7, where linguistic annotations were verified in-person for consistency. |
| **`v_8_dataset.conllu`** | **90 MB** | **213,576** | **`best_model_v8.pt`** | **Original Raw Baseline Dataset**: The complete raw legacy Telugu corpus (containing natural sentence repetitions) used to train the baseline V8 model. |

---

### Format Specification

All datasets follow the standard 11-column tab-separated CoNLL-U format:
1. `Token` (Surface word)
2. `Lemma` (Root word)
3. `LexCat` (Lexical category: n, v, adj, avy, etc.)
4. `POS` (Part of Speech tag: NN, VM, NNP, etc.)
5. `Gender` (m, f, n, any, UNK)
6. `Number` (sg, pl, any, UNK)
7. `Person` (1, 2, 3, any, UNK)
8. `Case` (d for direct/nominative, o for oblique, UNK)
9. `Suffix` (Inflectional suffix characters)
10. `TAM` (Tense-Aspect-Modality marker)
11. `Chunk` (IOB chunk tag: B-NP, I-NP, B-VGF, etc.)
