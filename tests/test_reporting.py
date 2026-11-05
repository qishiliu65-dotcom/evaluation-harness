import json

from eval_harness.models import (
    EvaluationResult,
    BatchEvaluationResult,
    JudgeResult,
)

from eval_harness.reporting import (
    save_evaluation_result_json,
    save_judge_results_json,
)


def test_save_evaluation_result_json(tmp_path):
    result = BatchEvaluationResult(
        results=[
            EvaluationResult(
                case_id="case-001",
                actual_output="4",
                score=1.0,
                passed=True,
                reason="output matches expected output"
            )
        ],
        total=1,
        passed=1,
        pass_rate=1.0,
        average_score=1.0,
        dataset_version="v0.1",
        dataset_sha256="abc123"
    )

    output_path = tmp_path / "result.json"

    save_evaluation_result_json(
        result,
        str(output_path)
    )

    with open(output_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["total"] == 1
    assert data["passed"] == 1
    assert data["dataset_version"] == "v0.1"
    assert data["dataset_sha256"] == "abc123"
    assert data["results"][0]["case_id"] == "case-001"

def test_save_evaluation_result_json_creates_parent_directories(tmp_path):
    result = BatchEvaluationResult(
        results=[],
        total=0,
        passed=0,
        pass_rate=0.0,
        average_score=0.0,
        dataset_version="v0.1",
        dataset_sha256="abc123"
    )

    output_path = tmp_path / "nested" / "results" / "result.json"

    save_evaluation_result_json(
        result,
        str(output_path)
    )

    assert output_path.exists()


def test_save_judge_results_json(tmp_path):
    results = [
        JudgeResult(
            case_id="judge-001",
            label="PASS",
            reason="回答正确。",
        ),
        JudgeResult(
            case_id="judge-002",
            label="FAIL",
            reason="回答错误。",
        ),
    ]

    output_path = tmp_path / "results" / "judge_results.json"

    save_judge_results_json(
        results,
        str(output_path),
    )

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert len(data) == 2
    assert data[0]["case_id"] == "judge-001"
    assert data[0]["label"] == "PASS"
    assert data[1]["case_id"] == "judge-002"
    assert data[1]["label"] == "FAIL"