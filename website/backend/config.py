import os
import re
import socket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_path(env_var: str, filename: str) -> str:
    explicit = os.getenv(env_var)
    if explicit and os.path.exists(explicit):
        return os.path.abspath(explicit)
    
    candidates = [
        os.path.join(BASE_DIR, "models", filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "..", filename),
        os.path.join(BASE_DIR, "..", "..", filename),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return os.path.join(BASE_DIR, "models", filename)

MODEL_PATH = _resolve_path("MODEL_PATH", "best_model_v8.pt")
METADATA_PATH = _resolve_path("METADATA_PATH", "metadata_v8.json")
MURIL_MODEL = os.getenv("MURIL_MODEL", "google/muril-base-cased")

# Candidate locations for built frontend
def _resolve_frontend_dist() -> str:
    explicit = os.getenv("FRONTEND_DIST")
    if explicit and os.path.isdir(explicit):
        return os.path.abspath(explicit)
    candidates = [
        os.path.join(BASE_DIR, "..", "dist", "client"),
        os.path.join(BASE_DIR, "..", "dist"),
        os.path.join(BASE_DIR, "..", "build"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.join(BASE_DIR, "..", "dist", "client")

FRONTEND_DIST = _resolve_frontend_dist()

# Local Vite can move to another port when its default port is occupied. Keep
# these common values for explicit deployments and match every localhost port
# during development. Credentials remain disabled in the API middleware.
_DEFAULT_CORS_ORIGINS = ",".join((
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]
_LOCAL_DEV_HOSTS = ("localhost", "127.0.0.1", socket.gethostname())
_LOCAL_DEV_HOST_PATTERN = "|".join(re.escape(host) for host in _LOCAL_DEV_HOSTS)
CORS_ORIGIN_REGEX = rf"^https?://({_LOCAL_DEV_HOST_PATTERN})(?::\d+)?$"

