import json
from unittest.mock import Mock

import pytest

from eval_harness.dataset import load_retrieval_cases
from eval_harness.rag_experiment import run_rag_experiment
from eval_harness.systems import MockSystem


def test_run_rag_experiment_with_mock(tmp_path):
    """验证批量运行、指标汇总和 JSON 保存。"""

    # 1. 在临时目录中保存结果，不污染正式 results 目录。
    # reports 子目录也用于验证程序会自动创建父目录。
    output_path = tmp_path / "reports" / "rag_mock.json"

    # 2. 创建固定返回答案的本地模型。
    llm = MockSystem(response="Test answer")

    # 包装 generate：保留原有行为，同时记录调用情况。
    generate_spy = Mock(wraps=llm.generate)
    llm.generate = generate_spy

    # 3. 执行实验，不调用真实 API。
    report = run_rag_experiment(
        llm=llm,
        model_name="mock",
        output_path=output_path,
        top_k=3,
    )

    summary = report["summary"]
    per_query_results = report["per_query_results"]

    # 4. 当前 v0.1 数据集应包含 12 道题。
    assert summary["query_count"] == 12
    assert len(per_query_results) == 12

    # 5. 检查报告中的实验配置。
    assert summary["dataset_version"] == "retrieval_cases_v0.1"
    assert summary["retriever"] == "BM25"
    assert summary["generator"] == "mock"
    assert summary["chunking"] == "paragraph"
    assert summary["top_k"] == 3
    assert summary["document_count"] == 4
    assert summary["chunk_count"] > 0

    # 6. 当前数据集每道题都有非空检索上下文，
    # 因此每道题应调用一次 generate，总共 12 次。
    assert generate_spy.call_count == 12

    # 7. 检查逐题结果。
    for result in per_query_results:
        assert result["answer"] == "Test answer"
        assert result["top_k"] == 3

        # 应保存非空排名，且结果数量不超过 top_k。
        assert 0 < len(result["retrieved_chunk_ids"]) <= 3

        # 应保存用于生成答案的非空上下文。
        assert result["context"].strip() != ""

        # 单题指标应处于合法范围。
        assert 0.0 <= result["recall_at_k"] <= 1.0
        assert 0.0 <= result["reciprocal_rank_at_k"] <= 1.0

        # 不能将答案采集误标为答案质量评测完成。
        assert result["answer_evaluation"] == {
            "status": "not_evaluated",
        }

    # 8. 检查汇总指标范围。
    assert 0.0 <= summary["mean_recall_at_k"] <= 1.0
    assert 0.0 <= summary["mrr_at_k"] <= 1.0

    # 9. 从逐题结果核对平均 Recall。
    expected_mean_recall = sum(
        result["recall_at_k"]
        for result in per_query_results
    ) / len(per_query_results)

    # MRR 应等于所有题目 RR 的平均值。
    expected_mrr = sum(
        result["reciprocal_rank_at_k"]
        for result in per_query_results
    ) / len(per_query_results)

    # approx 允许合理的浮点数误差。
    assert summary["mean_recall_at_k"] == pytest.approx(
        expected_mean_recall
    )
    assert summary["mrr_at_k"] == pytest.approx(expected_mrr)

    # 10. 汇总中也应明确标记答案未评测。
    assert summary["answer_evaluation"] == {
        "status": "not_evaluated",
    }

    # 11. 检查指纹记录的基本结构。
    # SHA-256 算法本身由 reproducibility 模块的测试负责。
    assert len(summary["dataset_sha256"]) == 64

    assert set(summary["document_sha256"]) == {
        "python-001",
        "network-001",
        "database-001",
        "redis-001",
    }

    for digest in summary["document_sha256"].values():
        assert len(digest) == 64

    # 12. 检查文件确实创建。
    assert output_path.is_file()

    # 13. 读取保存的 JSON，检查与函数返回值一致。
    saved_report = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_report == report


def test_run_rag_experiment_invalid_top_k(tmp_path):
    """top_k=0 时提前报错，不调用模型、不生成报告。"""

    output_path = tmp_path / "invalid_rag.json"

    llm = MockSystem(response="Should not be returned")

    # 记录生成方法是否被调用。
    generate_spy = Mock(wraps=llm.generate)
    llm.generate = generate_spy

    # 预期抛出 ValueError，并检查错误信息。
    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        run_rag_experiment(
            llm=llm,
            model_name="mock",
            output_path=output_path,
            top_k=0,
        )

    # 非法参数不能触发模型调用。
    generate_spy.assert_not_called()

    # 实验被拒绝，不应创建结果文件。
    assert not output_path.exists()


def test_run_rag_experiment_preserves_case_order(tmp_path):
    """逐题结果的顺序、问题和 Gold 应与数据集一致。"""

    # 1. 读取原始数据集，得到预期顺序。
    cases = load_retrieval_cases(
        "data/retrieval_cases_v0.1.jsonl"
    )

    # 避免两个空列表比较也通过测试。
    assert len(cases) == 12

    expected_ids = [
        case.id
        for case in cases
    ]

    # 2. 使用本地 Mock 运行实验。
    report = run_rag_experiment(
        llm=MockSystem(response="Test answer"),
        model_name="mock",
        output_path=tmp_path / "rag_order.json",
        top_k=3,
    )

    per_query_results = report["per_query_results"]

    actual_ids = [
        result["case_id"]
        for result in per_query_results
    ]

    # 3. 列表相等会检查长度、内容和顺序。
    # 不能改为 set 比较，否则无法发现顺序变化。
    assert actual_ids == expected_ids

    # 4. 检查每道题的其他信息是否对应。
    # 上面的断言已经确认长度一致，因此 zip 不会漏掉多余记录。
    for case, result in zip(cases, per_query_results):
        assert result["query"] == case.query
        assert result["relevant_chunk_ids"] == (
            case.relevant_chunk_ids
        )