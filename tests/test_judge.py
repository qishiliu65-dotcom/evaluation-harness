import pytest

from types import SimpleNamespace
from unittest.mock import Mock, call

from eval_harness.judge import LLMJudge, run_judge_batch
from eval_harness.models import (
    JudgeCase,
    JudgeOutput,
    JudgeResult,
)


def test_judge_returns_structured_result(monkeypatch):
    # 设置临时环境变量，让 LLMJudge 可以完成初始化。
    # 这里是假配置，后面会把真实 API 调用替换掉。
    monkeypatch.setenv(
        "DASHSCOPE_API_KEY",
        "test-api-key",
    )
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://example.com/v1",
    )

    judge = LLMJudge(
        model="test-model"
    )

    # 创建一条待评样本。
    # gold 信息保存在 case 中，但不应该发送给 Judge。
    case = JudgeCase(
        id="judge-001",
        input="法国首都是哪里？",
        candidate_output="里昂",
        gold_label="FAIL",
        gold_reason="GOLD_ONLY_MARKER：法国首都是巴黎。",
    )

    # 模拟 LLM structured output。
    expected_reason = "法国首都是巴黎，而不是里昂。"

    fake_output = JudgeOutput(
        label="FAIL",
        reason=expected_reason,
    )

    # 模拟 SDK 返回结构：
    # response.choices[0].message.parsed
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=fake_output,
                ),
            ),
        ],
    )

    # 假的 parse 方法。
    fake_parse = Mock(
        return_value=fake_response
    )

    # 替换真实 API 调用。
    monkeypatch.setattr(
        judge.client.chat.completions,
        "parse",
        fake_parse,
    )

    # 执行真正需要测试的 judge()。
    result = judge.judge(case)

    # 检查返回类型和内容。
    assert isinstance(result, JudgeResult)
    assert result.case_id == case.id
    assert result.label == "FAIL"
    assert result.reason == expected_reason

    # 确认 parse 只调用了一次。
    fake_parse.assert_called_once()

    # 检查传给 API 的参数。
    sent_kwargs = fake_parse.call_args.kwargs

    assert sent_kwargs["model"] == "test-model"
    assert sent_kwargs["response_format"] is JudgeOutput

    # 检查 prompt 中包含必要信息。
    sent_prompt = sent_kwargs["messages"][0]["content"]

    assert case.input in sent_prompt
    assert case.candidate_output in sent_prompt

    # 检查 gold reason 没有泄漏进 prompt。
    assert case.gold_reason not in sent_prompt


def test_judge_requires_api_key(monkeypatch):
    # 删除 API key，模拟缺少配置。
    monkeypatch.delenv(
        "DASHSCOPE_API_KEY",
        raising=False,
    )

    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://example.com/v1",
    )

    with pytest.raises(
        ValueError,
        match="DASHSCOPE_API_KEY is not set",
    ):
        LLMJudge(
            model="test-model"
        )


def test_judge_rejects_empty_structured_output(monkeypatch):
    # 设置假环境变量。
    monkeypatch.setenv(
        "DASHSCOPE_API_KEY",
        "test-api-key",
    )
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://example.com/v1",
    )

    judge = LLMJudge(
        model="test-model"
    )

    # 模拟 structured output 解析失败。
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=None,
                ),
            ),
        ],
    )

    monkeypatch.setattr(
        judge.client.chat.completions,
        "parse",
        Mock(return_value=fake_response),
    )

    case = JudgeCase(
        id="judge-001",
        input="法国首都是哪里？",
        candidate_output="里昂",
        gold_label="FAIL",
        gold_reason="法国首都是巴黎。",
    )

    with pytest.raises(
        ValueError,
        match="LLM judge returned no structured output",
    ):
        judge.judge(case)

def test_run_judge_batch():
    """两条样本应依次交给 Judge，并按顺序返回两个结果。"""

    # 1. 准备两条输入样本。
    cases = [
        JudgeCase(
            id="judge-001",
            input="法国首都是哪里？",
            candidate_output="巴黎",
            gold_label="PASS",
            gold_reason="法国首都是巴黎。",
        ),
        JudgeCase(
            id="judge-002",
            input="1 + 1 等于多少？",
            candidate_output="3",
            gold_label="FAIL",
            gold_reason="1 + 1 等于 2。",
        ),
    ]

    # 2. 准备假的 Judge 返回结果。
    # 使用不同的 ID、标签和理由，方便检查结果是否被正确收集。
    expected_results = [
        JudgeResult(
            case_id="judge-001",
            label="PASS",
            reason="候选回答正确。",
        ),
        JudgeResult(
            case_id="judge-002",
            label="FAIL",
            reason="候选回答计算错误。",
        ),
    ]

    # 3. 创建假的 Judge 对象。
    #
    # 不实例化真正的 LLMJudge，因此不会执行 __init__，
    # 不需要 API key，也不会创建真实 API 客户端。
    #
    # spec=LLMJudge 限制 Mock 可访问的属性，
    # 例如把 .judge 错写成 .judeg 时会报错。
    fake_judge = Mock(spec=LLMJudge)

    # 4. 设置 judge() 每次调用的返回值。
    #
    # side_effect 接收列表时：
    # 第一次调用 → 返回 expected_results[0]
    # 第二次调用 → 返回 expected_results[1]
    fake_judge.judge.side_effect = expected_results

    # 5. 执行真实的批量处理函数，传入假的 Judge。
    results = run_judge_batch(cases, fake_judge)

    # 6. 检查返回列表的数量、内容和顺序。
    assert len(results) == 2
    assert results == expected_results

    # 7. 检查 judge() 恰好调用两次，
    # 且第一次收到第一条样本，第二次收到第二条样本。
    assert fake_judge.judge.call_count == 2
    assert fake_judge.judge.call_args_list == [
        call(cases[0]),
        call(cases[1]),
    ]


def test_run_judge_batch_rejects_empty_cases():
    """空样本列表应报错，并且不调用 Judge。"""
    fake_judge = Mock(spec=LLMJudge)

    # 必须抛出 ValueError，错误信息也应符合预期。
    with pytest.raises(ValueError, match="Judge cases must not be empty"):
        run_judge_batch([], fake_judge)

    # 空输入应该在循环前被拒绝，不应该调用任何单条评审。
    fake_judge.judge.assert_not_called()


def test_judge_uses_custom_rubric(monkeypatch):
    monkeypatch.setenv(
        "DASHSCOPE_API_KEY",
        "test-api-key",
    )
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://example.com/v1",
    )

    custom_rubric = """
    PASS：核心内容正确即可。
    FAIL：核心内容错误。
    """

    judge = LLMJudge(
        model="test-model",
        rubric=custom_rubric,
    )

    case = JudgeCase(
        id="judge-001",
        input="测试问题",
        candidate_output="测试回答",
        gold_label="PASS",
        gold_reason="测试 gold reason",
    )

    fake_output = JudgeOutput(
        label="PASS",
        reason="测试通过。",
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=fake_output,
                ),
            ),
        ],
    )

    fake_parse = Mock(
        return_value=fake_response
    )

    monkeypatch.setattr(
        judge.client.chat.completions,
        "parse",
        fake_parse,
    )

    judge.judge(case)

    sent_prompt = fake_parse.call_args.kwargs["messages"][0]["content"]

    assert custom_rubric in sent_prompt