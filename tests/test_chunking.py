import pytest

from eval_harness.chunking import (
    chunk_document_fixed,
    chunk_document_by_paragraph,
)
from eval_harness.models import Document


def test_chunk_document_fixed_splits_text():
    """正常切分，并保留最后不足 chunk_size 的一段。"""
    document = Document(
        id="doc-001",
        text="ABCDEFGHIJ",
    )

    chunks = chunk_document_fixed(document, chunk_size=4)

    # 检查段数、内容和顺序。
    assert len(chunks) == 3
    assert [chunk.text for chunk in chunks] == [
        "ABCD",
        "EFGH",
        "IJ",
    ]

    # 检查每段的 ID 是否符合约定。
    assert [chunk.id for chunk in chunks] == [
        "doc-001-chunk-0",
        "doc-001-chunk-1",
        "doc-001-chunk-2",
    ]


def test_chunk_document_fixed_preserves_document_id():
    """每个段都记录正确的来源文档 ID。"""
    document = Document(
        id="network-001",
        text="ABCDEFGHIJ",
    )

    chunks = chunk_document_fixed(document, chunk_size=4)

    assert len(chunks) == 3

    for chunk in chunks:
        assert chunk.document_id == document.id


def test_chunk_document_fixed_empty_text():
    """空正文返回空列表。"""
    document = Document(
        id="doc-empty",
        text="",
    )

    chunks = chunk_document_fixed(document, chunk_size=4)

    assert chunks == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_document_fixed_rejects_invalid_size(chunk_size):
    """大小为 0 或负数时，明确抛出异常。"""
    document = Document(
        id="doc-001",
        text="ABCDEFGHIJ",
    )

    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than 0",
    ):
        chunk_document_fixed(document, chunk_size=chunk_size)


def test_chunk_document_by_paragraph_splits_text():
    """正常三段：检查段数、正文顺序和生成的 ID。"""
    document = Document(
        id="doc-001",
        text="AAA\n\nBBB\n\nCCC",
    )

    chunks = chunk_document_by_paragraph(document)

    assert len(chunks) == 3

    assert [chunk.text for chunk in chunks] == [
        "AAA",
        "BBB",
        "CCC",
    ]

    assert [chunk.id for chunk in chunks] == [
        "doc-001-chunk-0",
        "doc-001-chunk-1",
        "doc-001-chunk-2",
    ]


def test_chunk_document_by_paragraph_preserves_document_id():
    """每个段都保留来源文档的 ID。"""
    document = Document(
        id="network-001",
        text="TCP provides reliable transport.\n\nHTTP is an application protocol.",
    )

    chunks = chunk_document_by_paragraph(document)

    # 先确认生成了两段，避免空列表让下面的循环检查落空。
    assert len(chunks) == 2

    for chunk in chunks:
        assert chunk.document_id == document.id


def test_chunk_document_by_paragraph_skips_empty_paragraphs():
    """跳过空段、清理首尾空白，且有效段的编号连续。"""
    document = Document(
        id="doc-001",
        text="\n\n  AAA  \n\n   \n\n\n\n  BBB  \n\n",
    )

    chunks = chunk_document_by_paragraph(document)

    assert len(chunks) == 2

    # 不生成空 Chunk，并删除有效段首尾的空白。
    assert [chunk.text for chunk in chunks] == [
        "AAA",
        "BBB",
    ]

    # 跳过空段后，编号仍应连续。
    assert [chunk.id for chunk in chunks] == [
        "doc-001-chunk-0",
        "doc-001-chunk-1",
    ]


def test_chunk_document_by_paragraph_empty_text():
    """空正文返回空列表。"""
    document = Document(
        id="doc-empty",
        text="",
    )

    chunks = chunk_document_by_paragraph(document)

    assert chunks == []