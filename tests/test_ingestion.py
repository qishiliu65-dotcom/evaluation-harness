import pytest

from eval_harness.ingestion import load_text_document


def test_load_text_document_reads_file(tmp_path):
    """能够正确读取 UTF-8 文本文件。"""
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "Python is a programming language.\n\nTCP provides reliable transport.",
        encoding="utf-8",
    )

    document = load_text_document(
        path=str(file_path),
        document_id="doc-001",
    )

    assert document.text == (
        "Python is a programming language.\n\n"
        "TCP provides reliable transport."
    )


def test_load_text_document_preserves_document_id(tmp_path):
    """返回的 Document 保留传入的 document_id。"""
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "Hello world.",
        encoding="utf-8",
    )

    document = load_text_document(
        path=str(file_path),
        document_id="network-001",
    )

    assert document.id == "network-001"


def test_load_text_document_missing_file_raises_error(tmp_path):
    """文件不存在时抛出 FileNotFoundError。"""
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_text_document(
            path=str(missing_path),
            document_id="doc-missing",
        )