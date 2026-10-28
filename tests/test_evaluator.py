from eval_harness.models import EvaluationCase
from eval_harness.evaluator import (
    evaluate_exact_match,
    evaluate_normalized_match,
    normalize_text,
)


def test_exact_match_passes():
    case = EvaluationCase(
        id="case-001",
        input="2 + 2 = ?",
        expected_output="4"
    )

    result = evaluate_exact_match(case, "4")

    assert result.case_id == "case-001"
    assert result.actual_output == "4"
    assert result.score == 1.0
    assert result.passed is True
    assert result.reason == "output matches expected output"


def test_exact_match_fails():
    case = EvaluationCase(
        id="case-001",
        input="2 + 2 = ?",
        expected_output="4"
    )

    result = evaluate_exact_match(case, "5")

    assert result.case_id == "case-001"
    assert result.actual_output == "5"
    assert result.score == 0.0
    assert result.passed is False
    assert result.reason == "output does not match expected output"

def test_normalize_text():
    result = normalize_text(" Paris. ")

    assert result == "paris"

def test_evaluate_normalized_match_passes():
    case = EvaluationCase(
        id="case-normalized-001",
        input="What is the capital of France?",
        expected_output="Paris"
    )

    result = evaluate_normalized_match(
        case,
        " Paris. "
    )

    assert result.actual_output == " Paris. "
    assert result.score == 1.0
    assert result.passed is True