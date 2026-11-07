import pytest

from eval_harness.experiment import build_experiment_summary
from eval_harness.models import HumanLabel


def test_build_experiment_summary():
    human = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="PASS"),
        HumanLabel(case_id="003", label="FAIL"),
        HumanLabel(case_id="004", label="FAIL"),
    ]

    gold = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
        HumanLabel(case_id="003", label="FAIL"),
        HumanLabel(case_id="004", label="FAIL"),
    ]

    strict = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="FAIL"),
        HumanLabel(case_id="003", label="FAIL"),
        HumanLabel(case_id="004", label="FAIL"),
    ]

    lenient = [
        HumanLabel(case_id="001", label="PASS"),
        HumanLabel(case_id="002", label="PASS"),
        HumanLabel(case_id="003", label="PASS"),
        HumanLabel(case_id="004", label="FAIL"),
    ]

    summary = build_experiment_summary(
        human_labels=human,
        gold_labels=gold,
        strict_labels=strict,
        lenient_labels=lenient,
        dataset_version="v0.2",
        model="test-model",
    )

    assert summary.dataset_version == "v0.2"
    assert summary.model == "test-model"

    assert summary.strict.raw_agreement_with_human == pytest.approx(0.75)
    assert summary.strict.raw_agreement_with_gold == pytest.approx(1.0)

    assert summary.lenient.raw_agreement_with_human == pytest.approx(0.75)
    assert summary.lenient.raw_agreement_with_gold == pytest.approx(0.5)

    assert summary.strict.confusion_vs_gold == {
        "tp": 1,
        "tn": 3,
        "fp": 0,
        "fn": 0,
    }

    assert summary.lenient.confusion_vs_gold == {
        "tp": 1,
        "tn": 1,
        "fp": 2,
        "fn": 0,
    }