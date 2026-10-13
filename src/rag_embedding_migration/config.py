from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
GOLDEN_QUERIES_PATH = DATA_DIR / "golden_queries.jsonl"

MODELS = {
    "v1": "intfloat/multilingual-e5-base",
    "v2": "BAAI/bge-m3",
    "v2-bad": "BAAI/bge-m3",
}

INCOMPLETE_INDEX_VERSIONS = {
    "v2-bad": {
        "docs/release-notes.md#v2-4-postgres",
    }
}

TOP_K = 5