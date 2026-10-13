from rag_embedding_migration.evaluate import evaluate


def test_incomplete_reindex_fails_critical_gate() -> None:
    report = evaluate("v2-bad")

    failed_critical_queries = [
        row["id"]
        for row in report["queries"]
        if row["critical"] and row["recall_at_k"] == 0
    ]

    assert failed_critical_queries == ["postgres-001"], (
        "Der unvollständige Reindex sollte genau die kritische "
        "PostgreSQL-17-Frage verfehlen."
    )