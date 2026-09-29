from fastapi.testclient import TestClient
from eval_harness.api import app

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