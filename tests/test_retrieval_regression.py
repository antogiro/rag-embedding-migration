import json
import os
from pathlib import Path
import pytest

MAX_RECALL_DROP = 0.02
MAX_MRR_DROP = 0.02


@pytest.fixture
def baseline_report():
    path = Path(os.getenv("BASELINE_REPORT", "reports/report_v1.json"))
    if not path.exists():
        pytest.skip(f"Baseline-Report nicht gefunden: {path} (erst 'python -m src.evaluation' ausführen)")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def candidate_report():
    path = Path(os.getenv("CANDIDATE_REPORT", "reports/report_v2.json"))
    if not path.exists():
        pytest.skip(f"Kandidaten-Report nicht gefunden: {path} (erst 'python -m src.evaluation' ausführen)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_candidate_retrieval_regression(baseline_report, candidate_report):
    base_m = baseline_report["metrics"]
    cand_m = candidate_report["metrics"]

    recall_drop = base_m["recall_at_5"] - cand_m["recall_at_5"]
    assert recall_drop <= MAX_RECALL_DROP, (
        f"Recall@5 um {recall_drop:.3f} gefallen (Toleranz: {MAX_RECALL_DROP})"
    )

    mrr_drop = base_m["mrr_at_5"] - cand_m["mrr_at_5"]
    assert mrr_drop <= MAX_MRR_DROP, (
        f"MRR@5 um {mrr_drop:.3f} gefallen (Toleranz: {MAX_MRR_DROP})"
    )


def test_critical_queries_zero_tolerance(candidate_report):
    failed_critical = [
        row["id"]
        for row in candidate_report["queries"]
        if row["critical"] and row["recall_at_k"] == 0
    ]
    assert not failed_critical, (
        f"Release blockiert! Fehlende Chunks für kritische Abfragen: {failed_critical}"
    )
