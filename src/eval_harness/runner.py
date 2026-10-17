from .models import EvaluationCase, EvaluationResult
from .systems import SystemUnderTest
from .evaluator import evaluate_exact_match


def run_evaluation(case: EvaluationCase, system: SystemUnderTest) -> EvaluationResult:
    actual_output = system.generate(case.input)
    return evaluate_exact_match(case, actual_output)


#拿到 case
#   ↓
#把 case.input 交给 system
#   ↓
#拿到 actual_output
#   ↓
#把 case + actual_output 交给 evaluator
#   ↓
#得到 EvaluationResult