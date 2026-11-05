import json

from .models import EvaluationCase, JudgeCase, HumanLabel, JudgeResult

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


def load_judge_cases(path: str) -> list[JudgeCase]:
    cases = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            data = json.loads(line)
            case = JudgeCase(**data)
            cases.append(case)

    return cases


def load_human_labels(path: str) -> list[HumanLabel]:
    labels = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            data = json.loads(line)
            label = HumanLabel(**data)
            labels.append(label)

    return labels


def load_judge_results(path: str) -> list[JudgeResult]:
    # 创建一个空列表，用来保存最终转换好的 JudgeResult 对象。
    results = []

    # 打开 JSON 文件。
    # "r" 表示只读模式。
    # encoding="utf-8" 保证中文内容可以正常读取。
    with open(path, "r", encoding="utf-8") as file:

        # json.load(file) 会一次性读取整个 JSON 文件，
        # 并把 JSON 内容转换成 Python 对象。
        #
        # 例如文件内容：
        # [
        #     {
        #         "case_id": "judge-001",
        #         "label": "PASS",
        #         "reason": "回答正确。"
        #     },
        #     {
        #         "case_id": "judge-002",
        #         "label": "FAIL",
        #         "reason": "回答错误。"
        #     }
        # ]
        #
        # 读取后 data 就是一个 Python 列表：
        # [
        #     {"case_id": "judge-001", "label": "PASS", ...},
        #     {"case_id": "judge-002", "label": "FAIL", ...},
        # ]
        data = json.load(file)

    # data 是一个 list，
    # item 每次代表其中的一条字典记录。
    for item in data:

        # **item 会把字典拆成关键字参数。
        #
        # 例如：
        # item = {
        #     "case_id": "judge-001",
        #     "label": "PASS",
        #     "reason": "回答正确。"
        # }
        #
        # JudgeResult(**item)
        # 相当于：
        #
        # JudgeResult(
        #     case_id="judge-001",
        #     label="PASS",
        #     reason="回答正确。",
        # )
        result = JudgeResult(**item)

        # 把当前转换好的 JudgeResult 对象加入 results 列表。
        results.append(result)

    # 返回所有 JudgeResult。
    #
    # 最终类型是：
    # list[JudgeResult]
    return results