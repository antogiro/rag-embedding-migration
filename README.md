# RAG Embedding Model Migration: Dual-Index Evaluation & CI-Gates

Begleitendes Referenzprojekt zum **Golem+**-Artikel *„Vektorräume sind nicht abwärtskompatibel – Warum der Modellwechsel in RAG-Systemen wie eine heikle Datenbankmigration behandelt werden muss“*.

Dieses Repository demonstriert ein Zero-Downtime-Migrationsverfahren beim Wechsel von Embedding-Modellen in produktiven RAG-Systemen:

1. **Baseline-Index (`v1`):** `intfloat/multilingual-e5-base` (768 Dimensionen)
2. **Kandidaten-Index (`v2`):** `BAAI/bge-m3` (1024 Dimensionen)
3. **Fehlerhafter Reindex (`v2-bad`):** Simuliert einen vergessenen Chunk (*Silent Corpus Drop*) und zeigt den daraus entstehenden *Silent Near-Miss*.

## Projektstruktur

- `data/corpus.json` & `data/golden_queries.json` – Reproduzierbarer Dokumentations-Korpus inkl. geschäftskritischer Testfragen (`critical: true`)
- `src/evaluation.py` – Retrieval-Engine, Metriken (`Recall@5`, `MRR@5`, `nDCG@5`, `Critical Pass Rate`) & Report-Generator
- `src/router.py` – Referenzimplementierung des `DualIndexRouter` für unterbrechungsfreie Cutover- und Rollback-Operationen
- `tests/test_retrieval_regression.py` – Automatisierte CI-Gates mit `pytest` (Release-Blocker bei Regressionen)
- `notebooks/rag_embedding_migration.ipynb` – Interaktives Jupyter/Colab-Notebook zur Reproduktion aller Tabellen

## Schnellstart (Lokal)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Evaluation ausführen & Reports erzeugen
python -m src.evaluation

# CI-Regressionstests gegen den validen Kandidaten (v2) ausführen
pytest -v
