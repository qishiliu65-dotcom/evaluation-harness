import json
from pathlib import Path

from .models import (
    BatchEvaluationResult,
    JudgeResult,
    JudgeExperimentSummary,
)


def save_evaluation_result_json(result: BatchEvaluationResult, path: str) -> None:
    data = result.model_dump()

    output_path = Path(path)

    # mkdir(...) = 如果这个目录不存在，就创建；parents=True表示中间层目录也可以一起创建；exist_ok=True表示目录已经存在也不要报错
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def save_judge_results_json(results: list[JudgeResult], path: str,) -> None:

    data = [result.model_dump() for result in results]
    # 等同于：
    #data = []

    #for result in results:
    #   result_dict = result.model_dump() 把 result 这个 Pydantic 对象转换成普通字典，再保存到变量 result_dict 中。
    #    data.append(result_dict)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_experiment_summary_json(summary: JudgeExperimentSummary, path: str) -> None:
    data = summary.model_dump()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )