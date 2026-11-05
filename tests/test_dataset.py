from eval_harness.dataset import (
    load_evaluation_cases,
    load_judge_cases,
    load_human_labels,
    load_judge_results,
)

from eval_harness.models import (
    EvaluationCase,
    JudgeCase,
    HumanLabel,
    JudgeResult,
)
import json
import pytest

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