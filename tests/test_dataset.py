from eval_harness.dataset import load_evaluation_cases
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