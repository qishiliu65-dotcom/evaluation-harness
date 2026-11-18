import pytest

from eval_harness.retrieval_metrics import (
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_mrr,
)


# ---------- Recall@K：11 条 ----------


def test_recall_at_k_first_result_matches():
    """第一名命中唯一 Gold。"""
    recall = compute_recall_at_k(
        retrieved_chunk_ids=["A", "B", "C"],
        relevant_chunk_ids=["A"],
        k=1,
    )

    assert recall == 1.0


def test_recall_at_k_first_result_does_not_match():
    """Gold 在第二名，不计入 Top-1。"""
    recall = compute_recall_at_k(
        retrieved_chunk_ids=["A", "B", "C"],
        relevant_chunk_ids=["B"],
        k=1,
    )

    assert recall == 0.0


def test_recall_at_k_matches_within_top_three():
    """Gold 在前三名内。"""
    recall = compute_recall_at_k(
        retrieved_chunk_ids=["A", "B", "C"],
        relevant_chunk_ids=["B"],
        k=3,
    )

    assert recall == 1.0


def test_recall_at_k_partial_match_with_multiple_gold_ids():
    """三个 Gold 找回两个。"""
    recall = compute_recall_at_k(
        retrieved_chunk_ids=["X", "A", "Y", "C"],
        relevant_chunk_ids=["A", "B", "C"],
        k=4,
    )

    assert recall == pytest.approx(2 / 3)


def test_recall_at_k_exceeding_result_count():
    """K 大于结果数量时，使用全部已有结果。"""
    recall = compute_recall_at_k(
        retrieved_chunk_ids=["A", "B"],
        relevant_chunk_ids=["A"],
        k=10,
    )

    assert recall == 1.0


def test_recall_at_k_duplicate_results_count_once():
    """先取 Top-K，再去重；重复结果仍占排名位置。"""
    recall = compute_recall_at_k(
        retrieved_chunk_ids=["A", "A", "B"],
        relevant_chunk_ids=["A", "B"],
        k=2,
    )

    assert recall == 0.5


def test_recall_at_k_empty_retrieved_results():
    """没有检索结果，召回为零。"""
    recall = compute_recall_at_k(
        retrieved_chunk_ids=[],
        relevant_chunk_ids=["A"],
        k=3,
    )

    assert recall == 0.0


def test_recall_at_k_duplicate_gold_ids_count_once():
    """Gold 中重复的 ID 只计一次。"""
    recall = compute_recall_at_k(
        retrieved_chunk_ids=["A"],
        relevant_chunk_ids=["A", "A", "B"],
        k=1,
    )

    assert recall == 0.5


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_invalid_k(k):
    """分别验证 K 为零和负数。"""
    with pytest.raises(ValueError, match="k must be greater than 0"):
        compute_recall_at_k(
            retrieved_chunk_ids=["A", "B"],
            relevant_chunk_ids=["A"],
            k=k,
        )


def test_recall_at_k_rejects_empty_gold():
    """没有 Gold 时，拒绝计算。"""
    with pytest.raises(
        ValueError,
        match="relevant_chunk_ids must not be empty",
    ):
        compute_recall_at_k(
            retrieved_chunk_ids=["A", "B"],
            relevant_chunk_ids=[],
            k=2,
        )


# ---------- Reciprocal Rank：7 条 ----------


def test_reciprocal_rank_first_result_matches():
    rr = compute_reciprocal_rank(["A", "B", "C"], ["A"])

    assert rr == 1.0


def test_reciprocal_rank_second_result_matches():
    rr = compute_reciprocal_rank(["A", "B", "C"], ["B"])

    assert rr == 0.5


def test_reciprocal_rank_third_result_matches():
    rr = compute_reciprocal_rank(["A", "B", "C"], ["C"])

    assert rr == pytest.approx(1 / 3)


def test_reciprocal_rank_no_match():
    rr = compute_reciprocal_rank(["A", "B", "C"], ["X"])

    assert rr == 0.0


def test_reciprocal_rank_empty_retrieved():
    rr = compute_reciprocal_rank([], ["A"])

    assert rr == 0.0


def test_reciprocal_rank_multiple_gold():
    """存在多个 Gold 时，只看第一次命中的排名。"""
    rr = compute_reciprocal_rank(["A", "B", "C"], ["B", "C"])

    assert rr == 0.5


def test_reciprocal_rank_rejects_empty_gold():
    with pytest.raises(
        ValueError,
        match="relevant_chunk_ids must not be empty",
    ):
        compute_reciprocal_rank(["A", "B", "C"], [])


# ---------- MRR：8 条 ----------


def test_mrr_with_mixed_ranks():
    """RR 分别为 1、0.5、0，平均为 0.5。"""
    retrieved_rankings = [
        ["A", "B", "C"],
        ["D", "E", "F"],
        ["G", "H", "I"],
    ]
    relevant_ids_per_query = [
        ["A"],
        ["E"],
        ["X"],
    ]

    mrr = compute_mrr(
        retrieved_rankings,
        relevant_ids_per_query,
    )

    assert mrr == pytest.approx(0.5)


def test_mrr_all_first_rank_matches():
    """所有 Query 都在第一名命中，MRR 为 1。"""
    retrieved_rankings = [
        ["A", "B"],
        ["C", "D"],
        ["E", "F"],
    ]
    relevant_ids_per_query = [
        ["A"],
        ["C"],
        ["E"],
    ]

    mrr = compute_mrr(
        retrieved_rankings,
        relevant_ids_per_query,
    )

    assert mrr == 1.0


def test_mrr_all_no_matches():
    """所有 Query 都未命中，MRR 为 0。"""
    retrieved_rankings = [
        ["A", "B"],
        ["C", "D"],
    ]
    relevant_ids_per_query = [
        ["X"],
        ["Y"],
    ]

    mrr = compute_mrr(
        retrieved_rankings,
        relevant_ids_per_query,
    )

    assert mrr == 0.0


def test_mrr_single_query():
    """只有一道题时，MRR 等于这道题的 RR。"""
    mrr = compute_mrr(
        retrieved_rankings=[["A", "B", "C"]],
        relevant_ids_per_query=[["C"]],
    )

    assert mrr == pytest.approx(1 / 3)


def test_mrr_mismatched_query_counts():
    """两组 Query 数量不同，应拒绝计算。"""
    retrieved_rankings = [
        ["A", "B"],
        ["C", "D"],
    ]
    relevant_ids_per_query = [
        ["A"],
    ]

    with pytest.raises(ValueError, match="same query count"):
        compute_mrr(
            retrieved_rankings,
            relevant_ids_per_query,
        )


def test_mrr_empty_dataset():
    """整个数据集没有 Query，不能计算平均值。"""
    with pytest.raises(
        ValueError,
        match="Query lists must not be empty",
    ):
        compute_mrr(
            retrieved_rankings=[],
            relevant_ids_per_query=[],
        )


def test_mrr_rejects_empty_gold():
    """即使前一道题合法，后续题目 Gold 为空也必须报错。"""
    retrieved_rankings = [
        ["A", "B"],
        ["C", "D"],
    ]
    relevant_ids_per_query = [
        ["A"],
        [],
    ]

    with pytest.raises(
        ValueError,
        match="relevant_chunk_ids must not be empty",
    ):
        compute_mrr(
            retrieved_rankings,
            relevant_ids_per_query,
        )


def test_mrr_empty_retrieved_query():
    """一题第一名命中，一题空检索，平均为 0.5。"""
    retrieved_rankings = [
        ["A", "B"],
        [],
    ]
    relevant_ids_per_query = [
        ["A"],
        ["X"],
    ]

    mrr = compute_mrr(
        retrieved_rankings,
        relevant_ids_per_query,
    )

    # 空检索的题仍计入分母：(1 + 0) / 2。
    assert mrr == pytest.approx(0.5)