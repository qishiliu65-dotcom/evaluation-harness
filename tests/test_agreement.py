import pytest

from eval_harness.agreement import (
    compute_raw_agreement,
    compute_cohens_kappa,
    compute_confusion_matrix,
    find_disagreements,
)
from eval_harness.models import HumanLabel


def test_raw_agreement_all_agree():
    """所有样本标签一致，返回 1.0。"""
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
    ]
    labels_b = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
    ]

    result = compute_raw_agreement(labels_a, labels_b)

    assert result == 1.0


def test_raw_agreement_partial_agreement():
    """三条中两条一致，返回 2/3。"""
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
        HumanLabel(case_id="003", label="PASS"),
    ]
    labels_b = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="PASS"),
        HumanLabel(case_id="003", label="PASS"),
    ]

    result = compute_raw_agreement(labels_a, labels_b)

    # 浮点数比较使用 approx，允许很小的数值误差。
    assert result == pytest.approx(2 / 3)


def test_raw_agreement_aligns_by_case_id():
    """B 的顺序不同，仍应按 ID 正确配对。"""
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
        HumanLabel(case_id="003", label="PASS"),
    ]
    labels_b = [
        HumanLabel(case_id="002", label="FAIL"),
        HumanLabel(case_id="003", label="PASS"),
        HumanLabel(case_id="001", label="PASS"),
    ]

    result = compute_raw_agreement(labels_a, labels_b)

    # 如果错误地按列表位置配对，这个测试就无法通过。
    assert result == 1.0


def test_raw_agreement_rejects_different_case_ids():
    """两组数量相同，但包含不同 ID，应拒绝计算。"""
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
    ]
    labels_b = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="003", label="FAIL"),
    ]

    with pytest.raises(ValueError, match="same case_ids"):
        compute_raw_agreement(labels_a, labels_b)


@pytest.mark.parametrize("duplicate_side", ["a", "b"])
def test_raw_agreement_rejects_duplicate_case_ids(duplicate_side):
    """分别检查 A、B 中存在重复 ID 的情况。"""
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
    ]
    labels_b = [
        HumanLabel(case_id="001", label="PASS"),
    ]

    # 即使重复条目的标签相同，也必须报错，
    # 防止字典转换静默覆盖重复记录。
    duplicate = HumanLabel(case_id="001", label="PASS")

    if duplicate_side == "a":
        labels_a.append(duplicate)
    else:
        labels_b.append(duplicate)

    with pytest.raises(
        ValueError,
        match=f"Duplicate case_id in labels_{duplicate_side}",
    ):
        compute_raw_agreement(labels_a, labels_b)


@pytest.mark.parametrize(
    "empty_side",
    ["a", "b", "both"],
)
def test_raw_agreement_rejects_empty_lists(empty_side):
    """分别检查 A 为空、B 为空、两边都为空。"""
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
    ]
    labels_b = [
        HumanLabel(case_id="001", label="PASS"),
    ]

    if empty_side in ("a", "both"):
        labels_a = []

    if empty_side in ("b", "both"):
        labels_b = []

    with pytest.raises(ValueError, match="must not be empty"):
        compute_raw_agreement(labels_a, labels_b)


def test_cohens_kappa_perfect_agreement():
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
    ]
    labels_b = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
    ]

    result = compute_cohens_kappa(labels_a, labels_b)

    assert result == pytest.approx(1.0)


def test_cohens_kappa_partial_agreement():
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="PASS"),
        HumanLabel(case_id="003", label="FAIL"),
        HumanLabel(case_id="004", label="FAIL"),
    ]

    labels_b = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
        HumanLabel(case_id="003", label="FAIL"),
        HumanLabel(case_id="004", label="FAIL"),
    ]

    result = compute_cohens_kappa(labels_a, labels_b)

    assert result == pytest.approx(0.5)

def test_cohens_kappa_rejects_expected_agreement_one():
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="PASS"),
    ]
    labels_b = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="PASS"),
    ]

    with pytest.raises(
        ValueError,
        match="expected agreement is 1",
    ):
        compute_cohens_kappa(labels_a, labels_b)

def test_cohens_kappa_rejects_different_case_ids():
    labels_a = [
        HumanLabel(case_id="001", label="PASS"),
    ]
    labels_b = [
        HumanLabel(case_id="002", label="PASS"),
    ]

    with pytest.raises(ValueError, match="same case_ids"):
        compute_cohens_kappa(labels_a, labels_b)


def test_confusion_matrix_counts_all_four_categories():
    reference = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="PASS"),
        HumanLabel(case_id="003", label="FAIL"),
        HumanLabel(case_id="004", label="FAIL"),
    ]

    predicted = [
        HumanLabel(case_id="001", label="PASS"),  # TP
        HumanLabel(case_id="002", label="FAIL"),  # FN
        HumanLabel(case_id="003", label="PASS"),  # FP
        HumanLabel(case_id="004", label="FAIL"),  # TN
    ]

    result = compute_confusion_matrix(
        reference,
        predicted,
    )

    assert result == {
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
    }


def test_confusion_matrix_aligns_by_case_id():
    reference = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
    ]

    predicted = [
        HumanLabel(case_id="002", label="FAIL"),
        HumanLabel(case_id="001", label="PASS"),
    ]

    result = compute_confusion_matrix(reference, predicted)

    assert result == {
        "tp": 1,
        "tn": 1,
        "fp": 0,
        "fn": 0,
    }


def test_confusion_matrix_rejects_different_case_ids():
    reference = [
        HumanLabel(case_id="001", label="PASS"),
    ]

    predicted = [
        HumanLabel(case_id="002", label="PASS"),
    ]

    with pytest.raises(ValueError, match="same case_ids"):
        compute_confusion_matrix(reference, predicted)


@pytest.mark.parametrize("duplicate_side", ["reference", "predicted"])
def test_confusion_matrix_rejects_duplicate_case_ids(duplicate_side):
    reference = [
        HumanLabel(case_id="001", label="PASS"),
    ]

    predicted = [
        HumanLabel(case_id="001", label="PASS"),
    ]

    duplicate = HumanLabel(case_id="001", label="PASS")

    if duplicate_side == "reference":
        reference.append(duplicate)
    else:
        predicted.append(duplicate)

    with pytest.raises(ValueError, match="Duplicate case_id"):
        compute_confusion_matrix(reference, predicted)


@pytest.mark.parametrize(
    "empty_side",
    ["reference", "predicted", "both"],
)
def test_confusion_matrix_rejects_empty_lists(empty_side):
    reference = [
        HumanLabel(case_id="001", label="PASS"),
    ]

    predicted = [
        HumanLabel(case_id="001", label="PASS"),
    ]

    if empty_side in ("reference", "both"):
        reference = []

    if empty_side in ("predicted", "both"):
        predicted = []

    with pytest.raises(ValueError, match="must not be empty"):
        compute_confusion_matrix(reference, predicted)

def test_find_disagreements_returns_mismatched_case_ids():
    reference = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="PASS"),
        HumanLabel(case_id="003", label="FAIL"),
    ]

    predicted = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
        HumanLabel(case_id="003", label="FAIL"),
    ]

    result = find_disagreements(reference, predicted)

    assert result == ["002"]


def test_find_disagreements_returns_empty_list_when_all_agree():
    reference = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
    ]

    predicted = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
    ]

    result = find_disagreements(reference, predicted)

    assert result == []