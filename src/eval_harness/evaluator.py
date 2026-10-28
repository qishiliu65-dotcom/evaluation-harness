from .models import EvaluationCase, EvaluationResult
import string


def normalize_text(text: str) -> str:
    # 去掉文本首尾的空白
    normalized = text.strip()

    # 统一转换成小写
    normalized = normalized.lower()

    # 保存去掉标点后的结果
    result = ""

    # 逐个检查字符串中的字符
    for char in normalized:
        # 如果当前字符不是标点，就保留下来
        if char not in string.punctuation:
            result += char

    # 返回规范化后的文本
    return result

def evaluate_exact_match(case: EvaluationCase, actual_output: str) -> EvaluationResult:
    passed = actual_output == case.expected_output
    score = 1.0 if passed else 0.0
    reason = "output matches expected output" if passed else "output does not match expected output"

    return EvaluationResult(
        case_id=case.id,
        actual_output=actual_output,
        score=score,
        passed=passed,
        reason=reason
    )

def evaluate_normalized_match(case: EvaluationCase, actual_output: str) -> EvaluationResult:
    # 分别规范化实际输出和标准答案
    normalized_actual = normalize_text(actual_output)
    normalized_expected = normalize_text(case.expected_output)

    # 比较规范化后的文本
    passed = normalized_actual == normalized_expected

    # 通过得 1 分，否则 0 分
    score = 1.0 if passed else 0.0

    # 记录评测原因
    reason = (
        "normalized output matches expected output"
        if passed
        else "normalized output does not match expected output"
    )

    # 返回结构化评测结果
    return EvaluationResult(
        case_id=case.id,
        actual_output=actual_output,
        score=score,
        passed=passed,
        reason=reason
    )