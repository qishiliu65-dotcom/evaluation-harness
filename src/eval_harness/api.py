from fastapi import FastAPI

from .models import (
    EvaluationRequest,
    EvaluationResult,
    BatchEvaluationRequest,
    BatchEvaluationResult,
)
from .evaluator import evaluate_exact_match
from .systems import LLMSystem
from .runner import run_batch_evaluation


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate(request: EvaluationRequest) -> EvaluationResult:
    return evaluate_exact_match(
        request.case,
        request.actual_output
    )


@app.post("/run/batch", response_model=BatchEvaluationResult)
def run_batch(request: BatchEvaluationRequest) -> BatchEvaluationResult:
    system = LLMSystem(model=request.model)

    return run_batch_evaluation(
        request.cases,
        system
    )