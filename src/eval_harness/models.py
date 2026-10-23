from pydantic import BaseModel, Field


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

class EvaluationRequest(BaseModel):
    case: EvaluationCase
    actual_output: str

class BatchEvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1)  
    #列表长度至少为 1
    model: str = Field(min_length=1)


    
