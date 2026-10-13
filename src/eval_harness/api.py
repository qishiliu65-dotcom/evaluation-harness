from fastapi import FastAPI

from .models import EvaluationRequest, EvaluationResult
from .evaluator import evaluate_exact_match


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