from pydantic import BaseModel


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

