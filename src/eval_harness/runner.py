from collections.abc import Callable

from .models import (
    EvaluationCase,
    EvaluationResult,
    BatchEvaluationResult,
)

from .systems import (
    SystemUnderTest,
    StructuredSystemUnderTest,
)

from .evaluator import evaluate_exact_match


# Evaluator 表示一种函数， Callable本质是定义一种函数
# 输入 EvaluationCase 和 str，
# 返回 EvaluationResult。
Evaluator = Callable[
    [EvaluationCase, str],
    EvaluationResult
]


def run_evaluation(case: EvaluationCase, system: SystemUnderTest, evaluator: Evaluator = evaluate_exact_match) -> EvaluationResult:
    # 让被测系统根据 case.input 生成实际输出
    actual_output = system.generate(case.input)

    # 使用传入的 evaluator 评测实际输出
    result = evaluator(
        case,
        actual_output
    )

    return result


def run_batch_evaluation(cases: list[EvaluationCase], system: SystemUnderTest, evaluator: Evaluator = evaluate_exact_match) -> BatchEvaluationResult:
    # 批量评测不接受空列表
    if not cases:
        raise ValueError("cases must not be empty")

    # 保存每一条 case 的评测结果
    evaluation_results = []

    # 逐条运行评测
    for case in cases:
        result = run_evaluation(
            case,
            system,
            evaluator
        )

        evaluation_results.append(result)

    # 总 case 数量
    total = len(evaluation_results)

    # 统计通过数量
    passed = sum(
        result.passed
        for result in evaluation_results
    )

    # 计算通过率
    pass_rate = passed / total

    # 计算平均分
    average_score = sum(
        result.score
        for result in evaluation_results
    ) / total

    # 返回批量评测结果
    return BatchEvaluationResult(
        total=total,
        passed=passed,
        pass_rate=pass_rate,
        average_score=average_score,
        results=evaluation_results
    )


def run_structured_evaluation(case: EvaluationCase, system: StructuredSystemUnderTest, evaluator: Evaluator = evaluate_exact_match) -> EvaluationResult:
    # 生成结构化输出
    structured_output = system.generate_structured(
        case.input
    )

    # 从结构化输出中取出真正需要评测的答案
    actual_output = structured_output.answer

    # 使用传入的 evaluator 评测答案
    return evaluator(
        case,
        actual_output
    )