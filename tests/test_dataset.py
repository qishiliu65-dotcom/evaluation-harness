from eval_harness.dataset import (
    load_evaluation_cases,
    load_judge_cases,
    load_human_labels,
    load_judge_results,
    load_retrieval_cases
)

from eval_harness.models import (
    EvaluationCase,
    JudgeCase,
    HumanLabel,
    JudgeResult,
    RetrievalCase
)
import json
import pytest
from pydantic import ValidationError

def test_load_evaluation_cases(tmp_path):
    file_path = tmp_path / "cases.jsonl"

    file_path.write_text(
        '{"id":"case-001","input":"question 1","expected_output":"4"}\n'
        '{"id":"case-002","input":"question 2","expected_output":"5"}\n',
        encoding="utf-8"
    )

    cases = load_evaluation_cases(str(file_path))

    assert len(cases) == 2
    assert cases[0].id == "case-001"
    assert cases[0].input == "question 1"
    assert cases[0].expected_output == "4"

    assert cases[1].id == "case-002"
    assert cases[1].expected_output == "5"


def test_load_evaluation_cases_skips_blank_lines(tmp_path):
    file_path = tmp_path / "cases.jsonl"

    file_path.write_text(
        '{"id":"case-001","input":"question 1","expected_output":"4"}\n'
        '\n'
        '   \n'
        '{"id":"case-002","input":"question 2","expected_output":"5"}\n',
        encoding="utf-8"
    )

    cases = load_evaluation_cases(str(file_path))

    assert len(cases) == 2
    assert cases[0].id == "case-001"
    assert cases[1].id == "case-002"

def test_load_evaluation_cases_rejects_invalid_json(tmp_path):
    file_path = tmp_path / "cases.jsonl"

    file_path.write_text(
        '{"id":"case-001","input":"question 1","expected_output":"4"}\n'
        'this is not valid json\n',
        encoding="utf-8"
    )

    with pytest.raises(json.JSONDecodeError):
        load_evaluation_cases(str(file_path))


def test_load_judge_cases(tmp_path):
    file_path = tmp_path / "judge_cases.jsonl"

    file_path.write_text(
        '{"id":"judge-001","input":"法国首都是哪里？","candidate_output":"里昂","gold_label":"FAIL","gold_reason":"法国首都是巴黎。"}\n',
        encoding="utf-8",
    )

    cases = load_judge_cases(str(file_path))

    assert len(cases) == 1
    assert isinstance(cases[0], JudgeCase)
    assert cases[0].id == "judge-001"
    assert cases[0].candidate_output == "里昂"
    assert cases[0].gold_label == "FAIL"


def test_load_judge_cases_skips_blank_lines(tmp_path):
    file_path = tmp_path / "judge_cases.jsonl"

    file_path.write_text(
        '\n'
        '{"id":"judge-001","input":"法国首都是哪里？","candidate_output":"里昂","gold_label":"FAIL","gold_reason":"法国首都是巴黎。"}\n'
        '\n',
        encoding="utf-8",
    )

    cases = load_judge_cases(str(file_path))

    assert len(cases) == 1


def test_load_human_labels(tmp_path):
    file_path = tmp_path / "human_labels.jsonl"

    file_path.write_text(
        '{"case_id":"judge-001","label":"PASS"}\n',
        encoding="utf-8",
    )

    labels = load_human_labels(str(file_path))

    assert len(labels) == 1
    assert isinstance(labels[0], HumanLabel)
    assert labels[0].case_id == "judge-001"
    assert labels[0].label == "PASS"


def test_load_judge_results(tmp_path):
    file_path = tmp_path / "judge_results.json"

    file_path.write_text(
        '''
[
  {
    "case_id": "judge-001",
    "label": "PASS",
    "reason": "回答正确。"
  },
  {
    "case_id": "judge-002",
    "label": "FAIL",
    "reason": "回答错误。"
  }
]
''',
        encoding="utf-8",
    )

    results = load_judge_results(str(file_path))

    assert len(results) == 2
    assert isinstance(results[0], JudgeResult)

    assert results[0].case_id == "judge-001"
    assert results[0].label == "PASS"
    assert results[0].reason == "回答正确。"

    assert results[1].case_id == "judge-002"
    assert results[1].label == "FAIL"


def test_load_retrieval_cases_reads_multiple_cases(tmp_path):
    """正确加载多条 JSONL，并保留各字段和记录顺序。"""
    records = [
        {
            "id": "retrieval-001",
            "query": "What does HTTP 500 indicate?",
            "relevant_chunk_ids": ["network-001-chunk-2"],
        },
        {
            "id": "retrieval-002",
            "query": "Compare TCP and UDP delivery guarantees.",
            "relevant_chunk_ids": [
                "network-001-chunk-0",
                "network-001-chunk-1",
            ],
        },
    ]

    # tmp_path 是 pytest 提供的临时目录，不会覆盖项目中的数据。
    path = tmp_path / "retrieval_cases.jsonl"

    # 每条记录单独转换为 JSON，再用换行连接。
    content = "\n".join(json.dumps(record) for record in records)
    path.write_text(content + "\n", encoding="utf-8")

    cases = load_retrieval_cases(str(path))

    assert len(cases) == 2
    assert all(isinstance(case, RetrievalCase) for case in cases)

    # 与手工准备的期望记录比较，检查所有字段及顺序。
    assert [case.model_dump() for case in cases] == records


def test_load_retrieval_cases_ignores_blank_lines(tmp_path):
    """忽略开头、中间和末尾的空行，以及只有空白的行。"""
    record_a = {
        "id": "retrieval-001",
        "query": "What does HTTP 500 indicate?",
        "relevant_chunk_ids": ["network-001-chunk-2"],
    }
    record_b = {
        "id": "retrieval-002",
        "query": "What is a Redis TTL?",
        "relevant_chunk_ids": ["redis-001-chunk-1"],
    }

    path = tmp_path / "retrieval_cases_with_blanks.jsonl"

    content = (
        "\n"
        "   \n"
        + json.dumps(record_a)
        + "\n\n"
        "\t\n"
        + json.dumps(record_b)
        + "\n\n"
    )
    path.write_text(content, encoding="utf-8")

    cases = load_retrieval_cases(str(path))

    assert len(cases) == 2
    assert [case.model_dump() for case in cases] == [
        record_a,
        record_b,
    ]


@pytest.mark.parametrize(
    "missing_field",
    ["id", "query", "relevant_chunk_ids"],
)
def test_load_retrieval_cases_rejects_missing_fields(
    tmp_path,
    missing_field,
):
    """任一必填字段缺失时，向调用方抛出 Pydantic 验证错误。"""
    record = {
        "id": "retrieval-001",
        "query": "What does HTTP 500 indicate?",
        "relevant_chunk_ids": ["network-001-chunk-2"],
    }

    # 每次测试删除一个必填字段。
    del record[missing_field]

    path = tmp_path / "retrieval_cases_missing_field.jsonl"
    path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as exc_info:
        load_retrieval_cases(str(path))

    # 确认报错原因确实是指定字段缺失。
    assert any(
        error["loc"] == (missing_field,)
        and error["type"] == "missing"
        for error in exc_info.value.errors()
    )