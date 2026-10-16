from eval_harness.models import EvaluationCase
from eval_harness.systems import MockSystem
from eval_harness.runner import run_evaluation


def test_run_evaluation_passes():
    case = EvaluationCase(
        id="case-001",
        input="2 + 2 = ?",
        expected_output="4"
    )

    system = MockSystem("4")

    result = run_evaluation(case, system)

    assert result.case_id == "case-001"
    assert result.actual_output == "4"
    assert result.score == 1.0
    assert result.passed is True
    assert result.reason == "output matches expected output"


def test_run_evaluation_fails():
    case = EvaluationCase(
        id="case-002",
        input="2 + 2 = ?",
        expected_output="4"
    )

    system = MockSystem("5")

    result = run_evaluation(case, system)

    assert result.case_id == "case-002"
    assert result.actual_output == "5"
    assert result.score == 0.0
    assert result.passed is False
    assert result.reason == "output does not match expected output"