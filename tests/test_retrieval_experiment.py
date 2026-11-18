import json
import pytest
from eval_harness.retrieval_experiment import main


def test_bm25_experiment_saves_results(tmp_path):
    output_path = tmp_path / "bm25_baseline.json"

    main(output_path=output_path)

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    assert result["summary"]["query_count"] == 12
    assert len(result["per_query_results"]) == 12

    summary = result["summary"]
    per_query_results = result["per_query_results"]

    # 检查 Summary 中的 Recall@1 是否等于逐 Query 的平均值。
    expected_recall_at_1 = sum(
        item["recall_at_1"]
        for item in per_query_results
    ) / len(per_query_results)

    assert summary["recall_at_1"] == pytest.approx(
        expected_recall_at_1
    )