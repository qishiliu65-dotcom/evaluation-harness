import sys

from eval_harness import cli
from eval_harness.systems import MockSystem


def test_cli_runs_batch_evaluation(tmp_path, monkeypatch, capsys):
    dataset_path = tmp_path / "cases.jsonl"

    dataset_path.write_text(
        '{"id":"case-001","input":"question 1","expected_output":"4"}\n'
        '{"id":"case-002","input":"question 2","expected_output":"4"}\n',
        encoding="utf-8"
    )

    monkeypatch.setattr(
        cli,
        "LLMSystem",
        lambda model: MockSystem("4")
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli",
            "--dataset",
            str(dataset_path),
            "--model",
            "fake-model"
        ]
    )

    cli.main()

    output = capsys.readouterr().out

    assert "total=2" in output
    assert "passed=2" in output
    assert "pass_rate=1.0" in output