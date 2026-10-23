import pytest

from fastapi.testclient import TestClient
from eval_harness.api import app
from eval_harness.systems import MockSystem

client = TestClient(app)

def test_health():
	response = client.get("/health")
	assert response.status_code == 200
	assert response.json() == {"status": "ok"}
"""
你写的 test_api.py
    │
    │  client.get("/health")
    ▼
第三方 TestClient
    │
    ▼
你写的 app
    │
    │  查找 GET /health
    ▼
你写的路由
@app.get("/health")
    │
    ▼
你写的 health()
    │
    ▼
你写的
{"status": "ok"}
"""

def test_evaluate_exact_match_passes():
    response = client.post(
        "/evaluate",
        json={
            "case": {
                "id": "case-001",
                "input": "2 + 2 = ?",
                "expected_output": "4"
            },
            "actual_output": "4"
        }
    )

    assert response.status_code == 200

    data = response.json()


    assert data["case_id"] == "case-001"
    assert data["actual_output"] == "4"
    assert data["score"] == 1.0
    assert data["passed"] is True
    assert data["reason"] == "output matches expected output"

def test_evaluate_exact_match_fails():
    response = client.post(
        "/evaluate",
        json={
            "case": {
                "id": "case-001",
                "input": "2 + 2 = ?",
                "expected_output": "4"
            },
            "actual_output": "5"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["case_id"] == "case-001"
    assert data["actual_output"] == "5"
    assert data["score"] == 0.0
    assert data["passed"] is False
    assert data["reason"] == "output does not match expected output"


def test_evaluate_rejects_missing_expected_output():
    # 向 /evaluate 发送一个不完整的请求：
    # case 里面故意缺少 expected_output
    response = client.post(
        "/evaluate",
        json={
            "case": {
                "id": "case-001",
                "input": "2 + 2 = ?"
            },
            "actual_output": "4"
        }
    )


    # 因为 expected_output 是 EvaluationCase 的必填字段，
    # FastAPI 会在进入 evaluate() 之前调用 Pydantic 做数据校验。
    # 请求结构不符合模型要求时，FastAPI 默认返回 422。
    assert response.status_code == 422


def test_run_batch(monkeypatch):
    monkeypatch.setattr(
        "eval_harness.api.LLMSystem",
        lambda model: MockSystem("4")
    )

    response = client.post(
        "/run/batch",
        json={
            "model": "qwen3.8-max",
            "cases": [
                {
                    "id": "case-001",
                    "input": "question 1",
                    "expected_output": "4"
                },
                {
                    "id": "case-002",
                    "input": "question 2",
                    "expected_output": "5"
                },
                {
                    "id": "case-003",
                    "input": "question 3",
                    "expected_output": "4"
                }
            ]
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 3
    assert data["passed"] == 2
    assert data["pass_rate"] == pytest.approx(2 / 3)
    assert data["average_score"] == pytest.approx(2 / 3)
    assert len(data["results"]) == 3


def test_run_batch_rejects_empty_cases():
    response = client.post(
        "/run/batch",
        json={
            "model": "qwen3.8-max",
            "cases": []
        }
    )

    assert response.status_code == 422


def test_run_batch_rejects_empty_model():
    response = client.post(
        "/run/batch",
        json={
            "model": "",
            "cases": [
                {
                    "id": "case-001",
                    "input": "question",
                    "expected_output": "4"
                }
            ]
        }
    )

    assert response.status_code == 422