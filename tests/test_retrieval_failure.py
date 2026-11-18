from copy import deepcopy

from eval_harness.retrieval_experiment import identify_retrieval_failures


def test_identify_retrieval_failures_top1_miss():
    results = [{"case_id": "Q1", "recall_at_1": 0.0, "recall_at_5": 1.0}]
    failures = identify_retrieval_failures(results)
    assert len(failures) == 1
    assert failures[0]["case_id"] == "Q1"
    assert failures[0]["failure_types"] == ["top1_miss"]


def test_identify_retrieval_failures_top5_partial():
    results = [{"case_id": "Q2", "recall_at_1": 0.0, "recall_at_5": 0.5}]
    failures = identify_retrieval_failures(results)
    assert len(failures) == 1
    assert failures[0]["failure_types"] == ["top1_miss", "top5_partial"]


def test_identify_retrieval_failures_top5_miss():
    results = [{"case_id": "Q3", "recall_at_1": 0.0, "recall_at_5": 0.0}]
    failures = identify_retrieval_failures(results)
    assert len(failures) == 1
    assert failures[0]["failure_types"] == ["top1_miss", "top5_miss"]


def test_identify_retrieval_failures_full_recall():
    results = [{"case_id": "Q4", "recall_at_1": 1.0, "recall_at_5": 1.0}]
    assert identify_retrieval_failures(results) == []


def test_identify_retrieval_failures_multiple_gold_partial_top1():
    results = [{"case_id": "Q5", "recall_at_1": 0.5, "recall_at_5": 1.0}]
    failures = identify_retrieval_failures(results)
    assert len(failures) == 1
    assert failures[0]["failure_types"] == ["top1_miss"]


def test_identify_retrieval_failures_empty_input():
    assert identify_retrieval_failures([]) == []


def test_identify_retrieval_failures_does_not_modify_input():
    results = [{
        "case_id": "Q6",
        "recall_at_1": 0.0,
        "recall_at_5": 0.5,
        "retrieved_chunk_ids": ["A", "B"],
    }]
    original = deepcopy(results)
    failures = identify_retrieval_failures(results)
    assert results == original
    assert "failure_types" not in results[0]
    assert failures[0]["failure_types"] == ["top1_miss", "top5_partial"]
