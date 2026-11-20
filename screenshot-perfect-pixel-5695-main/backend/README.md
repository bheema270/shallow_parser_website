# Telugu Shallow Parser — Backend

FastAPI backend serving the Telugu Shallow Parser model (`best_model_v8.pt`).

## Running the Backend

Using your project virtual environment:

```bash
# From the backend directory:
/home/bheemasiddartha/Desktop/sem6/BTP/6_lakhs/shallow_parser_past/venv_parser/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or using an activated environment:

```bash
source /home/bheemasiddartha/Desktop/sem6/BTP/6_lakhs/shallow_parser_past/venv_parser/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Model & Metadata
The backend automatically resolves:
- `best_model_v8.pt` (V8 MuRIL MTL Hierarchical model checkpoint)
- `metadata_v8.json` (Label maps, character vocabularies, and task definitions)

You can override paths via environment variables if desired:
- `MODEL_PATH`: Custom path to model checkpoint (.pt)
- `METADATA_PATH`: Custom path to metadata (.json)
- `MURIL_MODEL`: Hugging Face model identifier (defaults to `google/muril-base-cased`)
- `FRONTEND_DIST`: Directory of built frontend assets for production single-port serving

