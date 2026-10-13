from __future__ import annotations

import argparse
import json

from rag_embedding_migration.config import ARTIFACTS_DIR, DOCUMENTS_PATH, INCOMPLETE_INDEX_VERSIONS, MODELS
from rag_embedding_migration.corpus import load_jsonl
from rag_embedding_migration.embeddings import embed_documents, load_model
from rag_embedding_migration.index import LocalVectorIndex

def build_index(version: str) -> None:
    documents = load_jsonl(DOCUMENTS_PATH)
    model_name = MODELS[version]

    excluded_ids = INCOMPLETE_INDEX_VERSIONS.get(version, set())
    documents = [
        document
        for document in documents
        if document["id"] not in excluded_ids
    ]

    model = load_model(model_name)

    vectors = embed_documents(
        model,
        model_name,
        [document["text"] for document in documents],
    )

    output_dir = ARTIFACTS_DIR / f"index-{version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    LocalVectorIndex(documents, vectors).save(output_dir)

    manifest = {
        "index_version": version,
        "embedding_model": model_name,
        "document_count": len(documents),
        "vector_dimension": int(vectors.shape[1]),
        "excluded_chunk_ids": sorted(excluded_ids),
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=sorted(MODELS),
        help="Nur einen Index bauen; ohne Argument werden alle gebaut.",
    )
    args = parser.parse_args()

    versions = [args.version] if args.version else sorted(MODELS)
    for version in versions:
        build_index(version)

if __name__ == "__main__":
    main()

