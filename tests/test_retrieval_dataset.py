from eval_harness.chunking import chunk_document_by_paragraph
from eval_harness.dataset import load_retrieval_cases
from eval_harness.ingestion import load_text_document


def test_all_relevant_chunk_ids_exist():
    """所有 retrieval case 引用的 relevant chunk 都必须真实存在。"""

    # 1. 加载 4 个 RAG 文档。
    documents = [
        load_text_document(
            "data/rag_docs/python-001.txt",
            "python-001",
        ),
        load_text_document(
            "data/rag_docs/network-001.txt",
            "network-001",
        ),
        load_text_document(
            "data/rag_docs/database-001.txt",
            "database-001",
        ),
        load_text_document(
            "data/rag_docs/redis-001.txt",
            "redis-001",
        ),
    ]

    # 2. 对所有文档执行 paragraph-based chunking。
    all_chunks = []

    for document in documents:
        chunks = chunk_document_by_paragraph(document)
        all_chunks.extend(chunks)

    # 3. 收集所有实际生成的 chunk ID。
    chunk_ids = {chunk.id for chunk in all_chunks}

    # 4. 加载 retrieval evaluation cases。
    cases = load_retrieval_cases(
        "data/retrieval_cases_v0.1.jsonl"
    )

    # 5. 检查每一个 Gold relevant chunk ID 都真实存在。
    for case in cases:
        for relevant_chunk_id in case.relevant_chunk_ids:
            assert relevant_chunk_id in chunk_ids, (
                f"{case.id} references missing chunk "
                f"{relevant_chunk_id}"
            )




