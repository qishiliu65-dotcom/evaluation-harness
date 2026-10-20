import pytest
from types import SimpleNamespace

from eval_harness.systems import LLMSystem


def test_llm_system_requires_api_key(monkeypatch):
    # 缺少 API Key 时应报错
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com")

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY is not set"):
        LLMSystem("qwen3.8-max")


def test_llm_system_requires_base_url(monkeypatch):
    # 缺少 Base URL 时应报错
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_BASE_URL is not set"):
        LLMSystem("qwen3.8-max")


def test_llm_system_rejects_empty_content(monkeypatch):
    # LLM 没返回文本时应报错
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com/v1")

    system = LLMSystem("qwen3.8-max")

    # 伪造 content=None 的 API 返回
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None)
            )
        ]
    )

    # 用假返回替换真实 API 调用
    monkeypatch.setattr(
        system.client.chat.completions,
        "create",
        lambda **kwargs: fake_response
    )

    with pytest.raises(ValueError, match="LLM returned no text content"):
        system.generate("hello")