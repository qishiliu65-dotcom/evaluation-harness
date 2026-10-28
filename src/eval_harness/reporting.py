import json
from pathlib import Path

from .models import BatchEvaluationResult


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