from .models import EvaluationCase, EvaluationResult, BatchEvaluationResult
from .systems import SystemUnderTest
from .evaluator import evaluate_exact_match



def run_evaluation(case: EvaluationCase, system: SystemUnderTest) -> EvaluationResult:
    actual_output = system.generate(case.input)
    return evaluate_exact_match(case, actual_output)

def run_batch_evaluation(cases: list[EvaluationCase], system: SystemUnderTest) -> BatchEvaluationResult:
    if not cases:
        raise ValueError("cases must not be empty")

    evaluation_results = []
    passed = 0
    sum_score = 0

    for case in cases:
        evaluation_result = run_evaluation(case, system)

        if evaluation_result.passed:
            passed += 1

        sum_score += evaluation_result.score
        evaluation_results.append(evaluation_result)

    total = len(evaluation_results)
    pass_rate = passed / total
    average_score = sum_score / total

    return BatchEvaluationResult (results=evaluation_results, total=total, passed=passed, pass_rate=pass_rate, average_score=average_score)

#拿到 case
#   ↓
#把 case.input 交给 system
#   ↓
#拿到 actual_output
#   ↓
#把 case + actual_output 交给 evaluator
#   ↓
#得到 EvaluationResult