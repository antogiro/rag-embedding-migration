from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = DIR / "artifacts"

DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
GOLDEN_QUERIES_PATH = DATA_DIR / "golden_queries.jsonl"

MODELS = {
    "v1": "intfloat/multilingual-e5-base",
    "v2": "BAAI/bge-m3",
}

TOP_K = 5