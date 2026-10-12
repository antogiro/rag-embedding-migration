# RAG-Embedding-Migration

Ein minimales, lokal laufendes Referenzprojekt, um den Wechsel eines Embedding-Modells in einem produktionsnahen RAG-System kontrolliert zu testen.

Der Wechsel eines Embedding-Modells ist keine einfache Konfigurationsänderung: Vektoren aus unterschiedlichen Modellen sind nicht gegenseitig suchbar. Ein neuer Index muss vollständig aus den Quelldokumenten aufgebaut werden, parallel zum alten Index laufen und gegen einen festen Testdatensatz geprüft werden, bevor er Produktionsverkehr übernimmt.

Dieses Projekt zeigt genau diesen Migrationspfad:

- Zwei versionierte Vektorindizes für zwei Embedding-Modelle
- Ein Golden Set aus Suchfragen und relevanten Dokument-Chunk-IDs
- Retrieval-Metriken: Recall@5, MRR@5 und nDCG@5
- Ein Vergleichsreport für alten und neuen Index
- Automatische Regressionstests, die eine Qualitätsverschlechterung blockieren
- Lokale Ausführung ohne Cloud-API, ohne API-Key und ohne Vektordatenbank-Dienst

## Warum dieses Projekt?

Ein neues Embedding-Modell kann auf einem allgemeinen Benchmark besser abschneiden und trotzdem im eigenen RAG-System schlechtere Treffer liefern. Benchmarks helfen bei der Vorauswahl, ersetzen aber keine Evaluation auf dem eigenen Dokumentkorpus, den eigenen Abkürzungen, Sprachen, Produktversionen und typischen Nutzerfragen.

Ein Embedding-Modellwechsel ist daher eine Datenmigration:

```text
Alter Zustand
query --Modell A--> Vektor A --Suche--> Index A

Kandidat
query --Modell B--> Vektor B --Suche--> Index B
```

Ein Index B kann erst nach einem bestandenen Qualitätsvergleich freigegeben werden. Solange der Vergleich läuft, bleibt Index A die Rückfalloption.

## Architektur

```text
data/documents.jsonl
        │
        ├── Modell v1: intfloat/multilingual-e5-base
        │       └── artifacts/index-v1
        │
        └── Modell v2: BAAI/bge-m3
                └── artifacts/index-v2

data/golden_queries.jsonl
        │
        ├── Evaluation gegen index-v1
        │       └── artifacts/evaluation-v1.json
        │
        └── Evaluation gegen index-v2
                └── artifacts/evaluation-v2.json

artifacts/evaluation-v1.json
artifacts/evaluation-v2.json
        │
        └── compare.py
                └── Vergleichsreport und Regressionen

tests/test_retrieval_regression.py
        └── CI-Qualitäts-Gate
```

## Voraussetzungen

- Python 3.11 oder neuer
- [uv](https://docs.astral.sh/uv/)
- Keine GPU erforderlich
- Keine Cloud-API oder kein API-Key erforderlich

`uv` verwaltet die virtuelle Umgebung und Abhängigkeiten. Ein Aufruf von `uv run` stellt sicher, dass das Projekt-Environment mit der Lockdatei synchron ist, bevor der Befehl ausgeführt wird. [104][109]

## Installation

```bash
git clone <dein-repository-url>
cd rag-embedding-migration

uv sync
```

Beim ersten Lauf werden die Embedding-Modelle aus dem Hugging Face Hub heruntergeladen und lokal im Hugging-Face-Cache gespeichert. Danach laufen weitere Ausführungen schneller.

## Projektstruktur

```text
rag-embedding-migration/
├── data/
│   ├── documents.jsonl
│   └── golden_queries.jsonl
├── artifacts/
│   ├── index-v1/
│   ├── index-v2/
│   ├── evaluation-v1.json
│   └── evaluation-v2.json
├── src/
│   └── rag_embedding_migration/
│       ├── config.py
│       ├── corpus.py
│       ├── embeddings.py
│       ├── index.py
│       ├── metrics.py
│       ├── build_indexes.py
│       ├── evaluate.py
│       └── compare.py
├── tests/
│   └── test_retrieval_regression.py
└── .github/
    └── workflows/
        └── retrieval-eval.yml
```

## Datenformat

### Dokumente

Jede Zeile in `data/documents.jsonl` ist ein eigenständiger Chunk:

```json
{
  "id": "docs/configuration.md#http-timeout",
  "title": "HTTP-Timeout konfigurieren",
  "text": "Die Umgebungsvariable IMPORT_HTTP_TIMEOUT bestimmt das maximale HTTP-Timeout für Datenimporte in Sekunden.",
  "version": "2.4",
  "language": "de",
  "category": "configuration"
}
```

Die `id` muss stabil und eindeutig sein. Sie verbindet eine Nutzerfrage im Golden Set mit dem tatsächlich erwarteten Retrieval-Treffer.

### Golden Set

Jede Zeile in `data/golden_queries.jsonl` beschreibt eine Testfrage:

```json
{
  "id": "timeout-001",
  "query": "Wie erhöhe ich das Timeout für einen Datenimport?",
  "relevant_chunk_ids": [
    "docs/configuration.md#http-timeout"
  ],
  "segment": "de-config",
  "critical": true
}
```

| Feld | Bedeutung |
|---|---|
| `id` | Stabile Kennung der Testfrage |
| `query` | Realistische Nutzerfrage |
| `relevant_chunk_ids` | Chunks, die die Frage direkt oder ausreichend beantworten |
| `segment` | Auswertungsgruppe, zum Beispiel Sprache, Dokumenttyp oder Produktbereich |
| `critical` | Wenn `true`, darf diese Frage beim Cutover keinen relevanten Treffer verlieren |

## Nutzung

### 1. Indizes bauen

```bash
uv run python -m rag_embedding_migration.build_indexes
```

Dieser Befehl:

- lädt die Dokumente,
- berechnet Embeddings für Modell v1 und v2,
- erstellt zwei getrennte Vektorindizes,
- speichert je Index ein Manifest mit Modellname, Dokumentanzahl und Vektordimension.

Erwartete Ausgabe:

```text
{
  "index_version": "v1",
  "embedding_model": "intfloat/multilingual-e5-base",
  "document_count": 12,
  "vector_dimension": 768
}
```

```text
{
  "index_version": "v2",
  "embedding_model": "BAAI/bge-m3",
  "document_count": 12,
  "vector_dimension": 1024
}
```

### 2. Einen einzelnen Index evaluieren

```bash
uv run python -m rag_embedding_migration.evaluate --version v1
```

oder:

```bash
uv run python -m rag_embedding_migration.evaluate --version v2
```

Der Befehl erzeugt beziehungsweise aktualisiert:

```text
artifacts/evaluation-v1.json
artifacts/evaluation-v2.json
```

### 3. Beide Indizes evaluieren

```bash
uv run python -m rag_embedding_migration.evaluate --all
```

Erwartete Ausgabe:

```text
v1: intfloat/multilingual-e5-base
{
  "recall_at_5": 1.0,
  "mrr_at_5": 0.958,
  "ndcg_at_5": 0.969,
  "critical_pass_rate": 1.0
}

v2: BAAI/bge-m3
{
  "recall_at_5": 1.0,
  "mrr_at_5": 0.958,
  "ndcg_at_5": 0.969,
  "critical_pass_rate": 1.0
}
```

### 4. Alter und neuer Index vergleichen

```bash
uv run python -m rag_embedding_migration.compare
```

Beispiel:

```text
Alt: intfloat/multilingual-e5-base
Neu: BAAI/bge-m3

recall_at_5          1.000 -> 1.000 (+0.000)
mrr_at_5             0.958 -> 0.958 (+0.000)
ndcg_at_5            0.969 -> 0.969 (+0.000)
critical_pass_rate   1.000 -> 1.000 (+0.000)

Keine Recall@5-Regressionen.
```

### 5. Regressionstests ausführen

```bash
uv run pytest -q
```

Erwartete Ausgabe bei bestandenem Gate:

```text
..                                                                                                                      [100%]
2 passed in 30.89s
```

## Metriken

| Metrik | Frage | Bedeutung |
|---|---|---|
| `recall_at_5` | Wurde mindestens ein relevanter Chunk in den Top 5 gefunden? | Verhindert, dass entscheidende Evidenz überhaupt nicht im Antwortkontext ankommt |
| `mrr_at_5` | Wie weit oben steht der erste relevante Treffer? | Misst die Position des ersten brauchbaren Treffers |
| `ndcg_at_5` | Wie gut ist die gesamte Rangfolge? | Wird besonders aussagekräftig, wenn mehrere Treffer unterschiedliche Relevanzgrade haben |
| `critical_pass_rate` | Haben alle kritischen Fragen einen relevanten Treffer? | Schützt wichtige Release-, Sicherheits- oder Konfigurationsfragen vor Ausfällen |

Ein guter Durchschnittswert ist nicht ausreichend. Ein Kandidatenmodell kann insgesamt leicht besser sein und trotzdem eine kritische Frage verlieren. Deshalb prüft dieses Projekt Gesamtwerte und kritische Einzelfälle getrennt.

## Was die Tests prüfen

`tests/test_retrieval_regression.py` enthält zwei Gates:

### 1. Keine zu starke Verschlechterung

```python
MAX_RECALL_DROP = 0.02
MAX_MRR_DROP = 0.02
```

Der Kandidatenindex darf gegenüber dem Baseline-Index höchstens zwei Prozentpunkte bei Recall@5 und MRR@5 verlieren.

### 2. Kritische Fragen dürfen nicht ausfallen

Für jede Frage mit `"critical": true` muss mindestens ein relevanter Chunk in den Top 5 erscheinen.

## CI ausführen

Der Workflow `.github/workflows/retrieval-eval.yml` läuft bei jedem Push und Pull Request:

```bash
git add .
git commit -m "Add embedding migration evaluation"
git push
```

Der Workflow baut beide Indizes und führt anschließend die Regressionstests aus. Er scheitert, wenn der Kandidatenindex die definierten Schwellenwerte nicht erreicht.

## Grenzen

Dieses Projekt ist ein Demonstrator, kein vollständiges Produktions-RAG-System.

- Der Vektorindex ist bewusst eine einfache In-Memory-Implementierung. In Produktion gehört er in eine Vektordatenbank oder ein Datenbanksystem mit persistenter Speicherung, Filterung, ACLs und skalierbarer ANN-Suche.
- Das Golden Set ist klein. Für eine echte Migration sollte es kontinuierlich aus realen Nutzerfragen, Support-Tickets und kritischen Suchfällen wachsen.
- Das Projekt evaluiert Retrieval, nicht die endgültige LLM-Antwort. Ein guter Treffer ist Voraussetzung für eine gute Antwort, garantiert aber keine korrekte, vollständige oder gut formulierte Antwort.
- `nDCG@5` hat in diesem kleinen Set begrenzte Aussagekraft, weil jede Frage aktuell genau einen relevanten Chunk hat. Mit mehreren relevanten Chunks und abgestuften Relevanzwerten wird die Metrik informativer.
- Ein bestandener Test bedeutet nicht, dass Modell B generell besser ist. Er bedeutet nur, dass Modell B für dieses Golden Set die definierte Mindestqualität erreicht.

## Nächste Schritte

- Golden Set auf 30 bis 50 echte Fragen erweitern
- Mehrere relevante Chunks und Relevanzgrade je Frage definieren
- Chunking-Version, Modellrevision und Indexparameter im Manifest dokumentieren
- Shadow Reads gegen den Kandidatenindex einführen
- Rollback-Routing vom Kandidaten- auf den Baseline-Index testen
- Optional: ein Decision Model als Relevance Filter oder Reranker für die Top-\(k\)-Treffer ergänzen

## Lizenz

Bitte ergänze hier die gewählte Lizenz, zum Beispiel MIT oder Apache-2.0.