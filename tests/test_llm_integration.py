import os

import pytest

from eval_harness.models import EvaluationCase
from eval_harness.runner import run_evaluation
from eval_harness.systems import LLMSystem


# 标记为集成测试
@pytest.mark.integration

# 默认跳过；只有 RUN_LLM_INTEGRATION=1 时才执行
@pytest.mark.skipif(
    os.getenv("RUN_LLM_INTEGRATION") != "1",
    reason="LLM integration tests are disabled"
)
def test_real_llm_evaluation():
    # 准备评测样本
    case = EvaluationCase(
        id="case-llm-001",
        input="只回答数字：2 + 2 等于多少？",
        expected_output="4"
    )

    # 创建真实 LLM 被测系统
    system = LLMSystem("qwen3.8-max")

    # 运行完整评测流程
    result = run_evaluation(case, system)

    # 验证结果
    assert result.actual_output.strip() == "4"
    assert result.score == 1.0
    assert result.passed is True