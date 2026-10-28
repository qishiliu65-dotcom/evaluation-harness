from fastapi import FastAPI

from .models import (
    EvaluationRequest,
    EvaluationResult,
    BatchEvaluationRequest,
    BatchEvaluationResult,
)

from .evaluator import (
    evaluate_exact_match,
    evaluate_normalized_match,
)

from .systems import LLMSystem
from .runner import run_batch_evaluation


app = FastAPI()


evaluators = {
    "exact": evaluate_exact_match,
    "normalized": evaluate_normalized_match,
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate(request: EvaluationRequest) -> EvaluationResult:
    evaluator = evaluators[request.evaluator]

    return evaluator(
        request.case,
        request.actual_output
    )


@app.post("/run/batch", response_model=BatchEvaluationResult)
def run_batch(request: BatchEvaluationRequest) -> BatchEvaluationResult:
    system = LLMSystem(
        model=request.model
    )

    evaluator = evaluators[request.evaluator]

    return run_batch_evaluation(
        request.cases,
        system,
        evaluator
    )