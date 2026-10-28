import pytest

from eval_harness.models import EvaluationCase, AnswerOutput
from eval_harness.runner import (
    run_evaluation,
    run_batch_evaluation,
    run_structured_evaluation,
)
from eval_harness.systems import MockSystem
from eval_harness.evaluator import evaluate_normalized_match

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


def test_run_batch_evaluation_all_pass():
    cases = [
        EvaluationCase(
            id="case-001",
            input="question 1",
            expected_output="4"
        ),
        EvaluationCase(
            id="case-002",
            input="question 2",
            expected_output="4"
        ),
        EvaluationCase(
            id="case-003",
            input="question 3",
            expected_output="4"
        ),
    ]

    system = MockSystem("4")

    result = run_batch_evaluation(cases, system)

    assert result.total == 3
    assert result.passed == 3
    assert result.pass_rate == 1.0
    assert result.average_score == 1.0
    assert len(result.results) == 3


def test_run_batch_evaluation_mixed_results():
    cases = [
        EvaluationCase(
            id="case-001",
            input="question 1",
            expected_output="4"
        ),
        EvaluationCase(
            id="case-002",
            input="question 2",
            expected_output="5"
        ),
        EvaluationCase(
            id="case-003",
            input="question 3",
            expected_output="4"
        ),
    ]

    system = MockSystem("4")

    result = run_batch_evaluation(cases, system)

    assert result.total == 3
    assert result.passed == 2
    assert result.pass_rate == pytest.approx(2 / 3)
    assert result.average_score == pytest.approx(2 / 3)
    assert len(result.results) == 3

def test_run_batch_evaluation_rejects_empty_cases():
    system = MockSystem("4")

    with pytest.raises(
        ValueError,
        match="cases must not be empty"
    ):
        run_batch_evaluation([], system)


class FakeStructuredSystem:
    def generate_structured(self, input_text: str) -> AnswerOutput:
        return AnswerOutput(
            answer="4",
            confidence=1.0
        )

def test_run_structured_evaluation_passes():
    case = EvaluationCase(
        id="case-structured-001",
        input="2 + 2 = ?",
        expected_output="4"
    )

    system = FakeStructuredSystem()

    result = run_structured_evaluation(
        case,
        system
    )

    assert result.case_id == "case-structured-001"
    assert result.actual_output == "4"
    assert result.score == 1.0
    assert result.passed is True


def test_run_evaluation_with_normalized_match():
    case = EvaluationCase(
        id="case-normalized-001",
        input="What is the capital of France?",
        expected_output="Paris"
    )

    system = MockSystem(" Paris. ")

    result = run_evaluation(
        case,
        system,
        evaluate_normalized_match
    )

    assert result.actual_output == " Paris. "
    assert result.score == 1.0
    assert result.passed is True


def test_run_batch_evaluation_with_normalized_match():
    cases = [
        EvaluationCase(
            id="case-001",
            input="question 1",
            expected_output="Paris"
        ),
        EvaluationCase(
            id="case-002",
            input="question 2",
            expected_output="London"
        ),
    ]

    system = MockSystem(" Paris. ")

    result = run_batch_evaluation(
        cases,
        system,
        evaluate_normalized_match
    )

    assert result.total == 2
    assert result.passed == 1
    assert result.pass_rate == 0.5
    assert result.average_score == 0.5