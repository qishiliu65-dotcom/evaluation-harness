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


class JudgeCase(BaseModel):
    id: str
    input: str
    candidate_output: str
    gold_label: Literal["PASS", "FAIL"]
    gold_reason: str


class JudgeResult(BaseModel):
    case_id: str
    label: Literal["PASS", "FAIL"]
    reason: str


class HumanLabel(BaseModel):
    case_id: str
    label: Literal["PASS", "FAIL"]


class JudgeOutput(BaseModel):
    label: Literal["PASS", "FAIL"]
    reason: str

class DisagreementRecord(BaseModel):
    case_id: str
    reference_label: Literal["PASS", "FAIL"]
    predicted_label: Literal["PASS", "FAIL"]
    category: str

class JudgeExperimentMetrics(BaseModel):
    raw_agreement_with_human: float
    kappa_with_human: float
    raw_agreement_with_gold: float
    kappa_with_gold: float
    confusion_vs_human: dict[str, int]
    confusion_vs_gold: dict[str, int]

class JudgeExperimentSummary(BaseModel):
    dataset_version: str
    model: str
    strict: JudgeExperimentMetrics
    lenient: JudgeExperimentMetrics