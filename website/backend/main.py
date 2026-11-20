import os
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from formatter import FORMATTERS
from inference_engine import InferenceEngine
from model_registry import get_model_specs, get_model_by_id, get_default_model_id

logger = logging.getLogger("uvicorn.error")
analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parser-analysis")
analysis_jobs: dict[str, dict] = {}
analysis_jobs_lock = Lock()

# Model engines cache
engines: dict[str, InferenceEngine] = {}
engines_lock = Lock()


def get_or_load_engine(model_id: str | None = None) -> tuple[InferenceEngine, dict]:
    """Retrieve an already loaded engine or load it on-demand from backend/models/."""
    target_id = model_id or get_default_model_id()
    spec = get_model_by_id(target_id)
    if not spec:
        available = [s["id"] for s in get_model_specs()]
        raise ValueError(f"Model '{target_id}' not found. Available models: {available}")

    with engines_lock:
        if target_id in engines:
            return engines[target_id], spec

        logger.info("[startup] Loading model '%s' from %s", target_id, spec["model_path"])
        eng = InferenceEngine(spec["model_path"], spec["metadata_path"], config.MURIL_MODEL)
        engines[target_id] = eng
        return eng, spec


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load default model on startup so the first user query is fast
    default_id = get_default_model_id()
    try:
        get_or_load_engine(default_id)
        logger.info("[startup] Default model '%s' loaded successfully.", default_id)
    except Exception as e:
        logger.error("[startup] Default model '%s' failed to load: %s", default_id, e)
    yield


app = FastAPI(title="Telugu Shallow Parser", lifespan=lifespan)

@app.middleware("http")
async def log_request(request, call_next):
    """Log essential request metadata; suppress high-frequency status polling and static asset logs."""
    path = request.url.path
    is_polling_or_static = (
        path.startswith("/assets")
        or (path.startswith("/api/analyze/") and request.method == "GET")
    )

    if is_polling_or_static:
        return await call_next(request)

    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[api] Unhandled request failure: method=%s path=%s", request.method, path)
        raise

    logger.info(
        "[api] %s %s -> status=%d elapsed=%.2fs",
        request.method, path, response.status_code, time.perf_counter() - started_at,
    )
    return response

# Credentialed CORS must never be paired with a wildcard origin. This API uses
# no cookie or authorization credentials, so it can safely support a literal
# "*" only when the operator explicitly sets it in CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    format: str = "conllu"
    model_id: str | None = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "loaded_models": list(engines.keys()),
        "available_models": [s["id"] for s in get_model_specs()],
    }


@app.get("/api/models")
def list_models():
    """List all models discovered in backend/models/."""
    specs = get_model_specs()
    return {
        "models": [
            {
                "id": s["id"],
                "label": s["label"],
                "is_default": s.get("is_default", False),
                "model_file": s["model_file"],
            }
            for s in specs
        ],
        "default_model": get_default_model_id(),
    }


@app.get("/api/formats")
def formats():
    return {"formats": [{"id": k, "label": f.label, "extension": f.extension} for k, f in FORMATTERS.items()]}


def run_analysis_job(job_id: str, text: str, output_format: str, model_id: str | None) -> None:
    started_at = time.perf_counter()
    formatter = FORMATTERS[output_format]
    with analysis_jobs_lock:
        analysis_jobs[job_id]["status"] = "running"

    try:
        # Check cancellation before loading model
        with analysis_jobs_lock:
            if analysis_jobs[job_id].get("status") == "cancelled":
                logger.info("[api] Job %s was cancelled before starting", job_id)
                return

        eng, spec = get_or_load_engine(model_id)
        logger.info(
            "[api] Analysis job started: job_id=%s model=%s characters=%d format=%s",
            job_id, spec["id"], len(text), output_format,
        )

        sentences = eng.segment_sentences(text)
        doc_sentences = []

        for sentence_id, s in enumerate(sentences, start=1):
            if not s.strip():
                continue
            # Check cancellation between sentences for fast halt
            with analysis_jobs_lock:
                if analysis_jobs[job_id].get("status") == "cancelled":
                    logger.info("[api] Job %s cancelled at sentence %d/%d", job_id, sentence_id, len(sentences))
                    return

            toks = eng.predict(s)
            doc_sentences.append({
                "sentence_id": sentence_id,
                "sentence": s,
                "tokens": toks,
            })

        # The document table uses global IDs; each sentence retains its local IDs.
        all_tokens = []
        global_id = 1
        for s_idx, sent_obj in enumerate(doc_sentences):
            for t in sent_obj["tokens"]:
                t_copy = dict(t)
                all_tokens.append({**t_copy, "id": global_id, "sent_index": s_idx})
                global_id += 1
            sent_obj["raw_output"] = formatter.format(sent_obj["tokens"], sent_obj["sentence"])

        # Format full document raw output
        if hasattr(formatter, "format_document"):
            full_raw_output = formatter.format_document(doc_sentences)
        else:
            full_raw_output = "\n\n".join(s["raw_output"] for s in doc_sentences)

    except Exception as e:
        logger.exception("[api] Analysis job failed: job_id=%s elapsed=%.2fs", job_id, time.perf_counter() - started_at)
        with analysis_jobs_lock:
            if analysis_jobs[job_id].get("status") != "cancelled":
                analysis_jobs[job_id].update(status="failed", error=f"Analysis failed: {e}")
        return

    # Final check if job was cancelled during formatting
    with analysis_jobs_lock:
        if analysis_jobs[job_id].get("status") == "cancelled":
            return

    logger.info(
        "[api] Analysis job complete: job_id=%s model=%s sentences=%d tokens=%d elapsed=%.2fs",
        job_id, spec["id"], len(doc_sentences), len(all_tokens), time.perf_counter() - started_at,
    )

    result = {
        "tokens": all_tokens,
        "raw_output": full_raw_output,
        "format": output_format,
        "sentence": text,
        "sentences": doc_sentences,
        "model_used": spec["id"],
        "model_name": spec["label"],
    }
    with analysis_jobs_lock:
        analysis_jobs[job_id].update(status="complete", result=result)


@app.post("/api/analyze", status_code=202)
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Input text is required.")
    if req.format not in FORMATTERS:
        raise HTTPException(400, f"Unsupported format: {req.format}")

    # Validate model exists in registry
    model_id = req.model_id or get_default_model_id()
    if not get_model_by_id(model_id):
        available = [s["id"] for s in get_model_specs()]
        raise HTTPException(400, f"Unknown model '{model_id}'. Available models: {available}")

    job_id = uuid.uuid4().hex
    with analysis_jobs_lock:
        analysis_jobs[job_id] = {"status": "queued"}
    analysis_executor.submit(run_analysis_job, job_id, text, req.format, model_id)
    logger.info("[api] Analysis queued: job_id=%s model=%s characters=%d", job_id, model_id, len(text))
    return {"job_id": job_id, "status": "queued", "model_id": model_id}


@app.get("/api/analyze/{job_id}")
def analysis_status(job_id: str):
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Analysis job not found or server restarted.")
        if job["status"] == "complete":
            return {"status": "complete", "result": job["result"]}
        if job["status"] == "failed":
            return {"status": "failed", "detail": job["error"]}
        if job["status"] == "cancelled":
            return {"status": "cancelled", "detail": "Analysis was cancelled."}
        return {"status": job["status"]}


@app.post("/api/analyze/{job_id}/cancel")
def cancel_analysis(job_id: str):
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Analysis job not found.")
        job["status"] = "cancelled"
    logger.info("[api] Analysis cancelled: job_id=%s", job_id)
    return {"status": "cancelled", "job_id": job_id}


@app.get("/")
def root_redirect():
    """Redirect root access directly to /shallow_parser."""
    return RedirectResponse(url="/shallow_parser")


@app.get("/shallow_parser")
def serve_shallow_parser():
    """Serve the shallow parser frontend application."""
    index_file = os.path.join(config.FRONTEND_DIST, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {"message": "Telugu Shallow Parser API. Frontend build not found. Run 'npm run build'."}


# Production: serve the built frontend from the same process.
if os.path.isdir(config.FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

