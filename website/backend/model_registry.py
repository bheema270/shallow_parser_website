import os
import glob
import re
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("uvicorn.error")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "models_config.json")


def get_model_specs() -> List[dict]:
    """
    Load model specifications from models_config.json in backend/.
    Preserves the exact order and labels defined in models_config.json.
    Falls back to dynamic discovery if config is missing.
    """
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config_models = json.load(f)
            models = []
            for item in config_models:
                m_file = item.get("model_file", f"best_model_{item['id']}.pt")
                meta_file = item.get("metadata_file", f"metadata_{item['id']}.json")
                m_path = os.path.join(MODELS_DIR, m_file)
                meta_path = os.path.join(MODELS_DIR, meta_file)
                models.append({
                    "id": item["id"],
                    "label": item.get("label", item["id"]),
                    "description": item.get("description", ""),
                    "model_file": m_file,
                    "metadata_file": meta_file,
                    "model_path": os.path.abspath(m_path),
                    "metadata_path": os.path.abspath(meta_path),
                    "is_default": bool(item.get("is_default", False)),
                })
            if models and not any(m["is_default"] for m in models):
                models[0]["is_default"] = True
            return models
        except Exception as e:
            logger.warning("[model_registry] Failed to read models_config.json: %s", e)

    if not os.path.isdir(MODELS_DIR):
        return []

    pt_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
    json_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.json")))

    models = []
    for pt in pt_files:
        pt_base = os.path.basename(pt)
        stem = os.path.splitext(pt_base)[0]

        # Extract version number if present (e.g. best_model_v8 -> v8, best_model_v1 -> v1)
        m = re.search(r'v(\d+)', stem, re.IGNORECASE)
        if m:
            version = f"v{m.group(1)}"
            model_id = version
            label = f"MuRIL MTL {version.upper()}"
        else:
            clean_name = stem.replace("best_model_", "").replace("model_", "")
            model_id = clean_name
            label = clean_name.replace("_", " ").title()

        matching_json = None
        cand_names = [
            f"metadata_{model_id}.json",
            f"{stem}.json",
            f"metadata_{stem}.json",
            f"{model_id}.json",
            "metadata.json"
        ]
        for c in cand_names:
            full_c = os.path.join(MODELS_DIR, c)
            if os.path.isfile(full_c):
                matching_json = full_c
                break

        if not matching_json and m:
            v_tag = f"v{m.group(1)}".lower()
            for jf in json_files:
                if v_tag in os.path.basename(jf).lower():
                    matching_json = jf
                    break

        if not matching_json and len(json_files) == 1:
            matching_json = json_files[0]

        if matching_json:
            models.append({
                "id": model_id,
                "label": label,
                "model_file": pt_base,
                "metadata_file": os.path.basename(matching_json),
                "model_path": os.path.abspath(pt),
                "metadata_path": os.path.abspath(matching_json),
                "is_default": False
            })

    if models:
        default_found = False
        for mod in models:
            if mod["id"] == "v8":
                mod["is_default"] = True
                default_found = True
                break
        if not default_found:
            models[0]["is_default"] = True

    return models


def get_model_by_id(model_id: str) -> Optional[dict]:
    specs = get_model_specs()
    for s in specs:
        if s["id"].lower() == model_id.lower():
            return s
    return None


def get_default_model_id() -> str:
    specs = get_model_specs()
    for s in specs:
        if s.get("is_default"):
            return s["id"]
    return specs[0]["id"] if specs else "v8"
