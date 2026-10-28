from pydantic import BaseModel, Field
from typing import Literal

class EvaluationCase(BaseModel):
    id: str
    input: str
    expected_output: str

class EvaluationResult(BaseModel):
    case_id: str
    actual_output: str
    score: float
    passed: bool
    reason: str

class BatchEvaluationResult(BaseModel):
    results: list[EvaluationResult]
    total: int
    passed: int
    pass_rate: float
    average_score: float
    dataset_version: str | None = None
    dataset_sha256: str | None = None

class EvaluationRequest(BaseModel):
    case: EvaluationCase
    actual_output: str
    evaluator: Literal["exact", "normalized"] = "exact"


class BatchEvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1)
    model: str = Field(min_length=1)
    evaluator: Literal["exact", "normalized"] = "exact"

class AnswerOutput(BaseModel):
    answer: str
    confidence: float

    
