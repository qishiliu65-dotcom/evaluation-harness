import pytest

from eval_harness.models import Chunk, RetrievedChunk
from eval_harness.retrieval import BM25Retriever, tokenize


@pytest.fixture
def sample_chunks():
    """提供三条测试资料，每个测试都会获得一组新的对象。"""
    return [
        Chunk(
            id="chunk-1",
            document_id="python-001",
            text="Python lists are mutable sequences.",
        ),
        Chunk(
            id="chunk-2",
            document_id="network-001",
            text="HTTP status code 500 means Internal Server Error.",
        ),
        Chunk(
            id="chunk-3",
            document_id="redis-001",
            text="Redis keys can have a TTL.",
        ),
    ]


def test_tokenize_lowercases_and_removes_punctuation():
    """转为小写、去除标点，并保留数字。"""
    tokens = tokenize("HTTP 500 Error!")

    assert tokens == ["http", "500", "error"]


def test_bm25_retriever_rejects_empty_chunks():
    """没有资料时，应拒绝创建检索器。"""
    with pytest.raises(ValueError, match="chunks must not be empty"):
        BM25Retriever([])


@pytest.mark.parametrize("top_k", [0, -1])
def test_bm25_retriever_rejects_invalid_top_k(sample_chunks, top_k):
    """top_k 为零或负数时，应抛出异常。"""
    retriever = BM25Retriever(sample_chunks)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retriever.retrieve(
            query="HTTP 500 server error",
            top_k=top_k,
        )


def test_bm25_retriever_returns_requested_count(sample_chunks):
    """有三条资料，要求前两条，应返回两个 RetrievedChunk。"""
    retriever = BM25Retriever(sample_chunks)

    results = retriever.retrieve(
        query="HTTP 500 server error",
        top_k=2,
    )

    assert len(results) == 2
    assert all(isinstance(item, RetrievedChunk) for item in results)

    # 返回结果应按分数从高到低排列。
    assert results[0].score >= results[1].score


def test_bm25_retriever_ranks_relevant_chunk_first(sample_chunks):
    """与 HTTP 500 明显相关的资料应排在第一位。"""
    retriever = BM25Retriever(sample_chunks)

    results = retriever.retrieve(
        query="HTTP 500 server error",
        top_k=3,
    )

    assert results[0].chunk.id == "chunk-2"

    # 确认返回的是对应的原始资料对象内容。
    assert results[0].chunk == sample_chunks[1]


def test_bm25_retriever_top_k_exceeding_corpus_returns_all(sample_chunks):
    """请求数量超过资料总数时，返回全部资料，不重复补齐。"""
    retriever = BM25Retriever(sample_chunks)

    results = retriever.retrieve(
        query="HTTP 500 server error",
        top_k=10,
    )

    assert len(results) == 3

    # 检查没有漏掉资料，也没有用重复结果凑数量。
    assert {item.chunk.id for item in results} == {
        "chunk-1",
        "chunk-2",
        "chunk-3",
    }