import json

from .models import EvaluationCase

"""
    从 JSONL 文件中读取评测数据，
    并转换成 EvaluationCase 对象列表。

    参数：
        path: JSONL 文件路径

    返回：
        list[EvaluationCase]
"""
def load_evaluation_cases(path: str) -> list[EvaluationCase]:
    cases = []

    # 以只读模式打开文件，并明确使用 UTF-8 编码
    with open(path, "r", encoding="utf-8") as file:
        for line in file:

            # 如果这一行只有空格或换行，就直接跳过
            if not line.strip():
                continue

            # 把这一行 JSON 字符串转换成 Python dict
            data = json.loads(line)

            # 把 dict 解包，创建 EvaluationCase， ** 叫“字典解包”。
            case = EvaluationCase(**data)

            cases.append(case)

    # 文件全部读取完成后，返回所有 EvaluationCase
    return cases