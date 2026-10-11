from .models import EvaluationCase, EvaluationResult


def evaluate_exact_match(case: EvaluationCase, actual_output: str) -> EvaluationResult:
    passed = actual_output == case.expected_output
    score = 1.0 if passed else 0.0
    reason = "output matches expected output" if passed else "output does not match expected output"

    return EvaluationResult(
        case_id=case.id,
        actual_output=actual_output,
        score=score,
        passed=passed,
        reason=reason
    )