# RAG Embedding Migration: Ein Praxisleitfaden

Ein kleines Praxisprojekt, das zeigt, wie man in produktiven RAG-Systemen ein Embedding-Modell sicher migriert: Reindexing, Dual-Index-Betrieb, Qualitätsvergleich und automatische Regressionserkennung in der CI.

## Warum dieses Projekt?

RAG ist 2026 nicht mehr nur ein Experiment, sondern Teil kritischer produktiver Infrastruktur. Gerade bei Embedding-Modellen entsteht jedoch oft ein Problem:

- Ein Modell wird abgekündigt oder veraltet
- Das Reindexing kostet Zeit und Infrastruktur
- Eine Migration kann die Retrieval-Qualität verschlechtern
- Ohne Test-Harness ist ein Rollback schwer zu planen

Dieses Projekt zeigt die wichtigsten Schritte einer Embedding-Migration in einer einfachen, nachvollziehbaren Struktur.

## Quick Start

### 1) Repository in Codespaces öffnen

```bash
git clone https://github.com/antogiro/rag-embedding-migration.git
cd rag-embedding-migration
```

### 2) Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 3) Baseline-Index erzeugen

```bash
python scripts/index_baseline.py
```

Das erzeugt einen ersten Index mit dem Baseline-Modell und zeigt die wichtigsten Kennzahlen an:

- Anzahl Dokumente
- Embedding-Dimension
- Indexierungsdauer
- Indexgröße

### 4) Migration und Vergleich starten

```bash
python scripts/reindex_new_model.py
python scripts/evaluate_migration.py
```

## Projektstruktur

```text
rag-embedding-migration/
├── config/
│   ├── baseline_model.yaml
│   └── new_model.yaml
├── data/
│   ├── sample_documents.txt
│   ├── indexes/
│   └── test_queries.jsonl
├── eval_harness/
│   ├── ci_validator.py
│   └── metrics.py
├── scripts/
│   ├── index_baseline.py
│   ├── reindex_new_model.py
│   ├── evaluate_migration.py
│   └── dual_index_demo.py
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── eval_regression.yml
```

## Kernkonzepte

### 1. Reindexing
Ein Modellwechsel bedeutet meistens: Alle Dokumente müssen neu embeddet werden. Das ist nicht nur technisch, sondern auch ein Kosten- und Betriebsproblem.

### 2. Dual-Index-Betrieb
Beim Wechsel auf ein neues Embedding-Modell ist ein Dual-Index-Setup sinnvoll:

- Altes Modell weiter aktiv
- Neues Modell parallel testen
- Qualitätsvergleich mit einem Fixdatensatz
- Stufenweise Umschaltung

### 3. Qualitätsvalidierung
Beim Modellwechsel müssen Retrieval-Metriken überprüft werden:

- MRR
- NDCG@K
- Recall@K

Diese Metriken helfen dabei, Regressionen früh zu erkennen.

### 4. CI-Regressionstest
Ein einfaches Eval-Harness kann in GitHub Actions in der CI laufen:

- Testqueries mit Referenz-Ergebnissen
- Qualitätsvergleich Baseline vs. neu
- Abbruch bei regressiver Qualität

## Konfiguration

Die Modelle werden in `config/` definiert.

### Baseline-Modell
```yaml
model_name: "sentence-transformers/all-MiniLM-L6-v2"
dimension: 384
pooling_method: "mean"
```

### Neues Modell
```yaml
model_name: "BAAI/bge-small-en-v1.5"
dimension: 384
pooling_method: "cls"
```

## Beispielmodell-Strategie

| Strategie | Beschreibung | Risiko | Empfehlung |
| --- | --- | --- | --- |
| Big Bang | Alles auf einmal umstellen | Hoch | Nur für kleine / sichere Umgebungen |
| Dual-Index | Alt und neu parallel betreiben | Mittel | Gute Standardstrategie |
| Canary | Nur ein Teil des Traffics auf neues Modell | Niedrig | Ideal für Produktion |

## Nächste Schritte

Das Projekt ist bewusst klein und erweiterbar gebaut. Die nächsten logischen Schritte sind:

1. Test-Queries und Referenz-Ergebnisse hinzufügen
2. Reindexing-Scripts für neues Modell ergänzen
3. Qualitätsmetriken in `eval_harness/metrics.py` erweitern
4. GitHub Actions Workflow für CI-Regression einbauen
5. Artikel-Text mit den praktischen Ergebnissen ergänzen

## Für den Artikel

Dieses Repo ist als Einstieg für einen Praxisartikel gedacht, der alle relevanten Probleme rund um Embedding-Migration behandelt:

- hidden costs von produktiven RAG-Systemen
- Modellwechsel als Engineering-Problem
- Qualität vor und nach der Migration
- Reindexing als Operations-Aufgabe
- Rollback-Strategien und Sicherheitsnetze

## License

MIT
