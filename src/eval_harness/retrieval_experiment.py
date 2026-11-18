import json
from pathlib import Path

from .chunking import chunk_document_by_paragraph
from .dataset import load_retrieval_cases
from .ingestion import load_text_document
from .models import Chunk
from .retrieval import BM25Retriever
from .retrieval_metrics import (
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_mrr,
)
from .reproducibility import compute_file_sha256


def identify_retrieval_failures(
    per_query_results: list[dict],
) -> list[dict]:
    """根据逐题 Recall 识别问题，返回新的记录，不修改输入。"""

    failures: list[dict] = []

    for result in per_query_results:
        failure_types: list[str] = []

        # 当前约定：Top-1 未完整召回 Gold，包括部分召回。
        if result["recall_at_1"] < 1.0:
            failure_types.append("top1_miss")

        # Top-5 找回了一部分 Gold，但没有全部找回。
        if 0 < result["recall_at_5"] < 1.0:
            failure_types.append("top5_partial")

        # Top-5 没有找到任何 Gold。
        if result["recall_at_5"] == 0:
            failure_types.append("top5_miss")

        # 只保存存在问题的 Query。
        if failure_types:
            failure_record = result.copy()
            failure_record["failure_types"] = failure_types
            failures.append(failure_record)

    return failures


def main(
    output_path: Path = Path("results/bm25_baseline_v0.1.json"),
) -> None:
    """运行 BM25 Baseline，保存指标、输入指纹及失败详情。"""

    # 1. 加载四个文档。
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

    # 2. 按段落切分，汇总所有 Chunk。
    all_chunks: list[Chunk] = []

    for document in documents:
        chunks = chunk_document_by_paragraph(document)
        all_chunks.extend(chunks)

    # 3. 创建一次检索器，供全部 Query 共用。
    retriever = BM25Retriever(all_chunks)

    # 4. 加载检索评测题。
    cases = load_retrieval_cases(
        "data/retrieval_cases_v0.1.jsonl"
    )

    if not cases:
        raise ValueError("Retrieval cases must not be empty")

    # 5. 准备结果容器。
    retrieved_rankings: list[list[str]] = []
    relevant_ids_per_query: list[list[str]] = []
    per_query_results: list[dict] = []

    # 6. 逐题执行 Top-5 检索，并计算单题指标。
    for case in cases:
        results = retriever.retrieve(
            query=case.query,
            top_k=5,
        )

        # 保留检索顺序，不转成集合。
        retrieved_ids = [
            item.chunk.id
            for item in results
        ]

        # 在同一轮中保存排名和对应 Gold。
        retrieved_rankings.append(retrieved_ids)
        relevant_ids_per_query.append(case.relevant_chunk_ids)

        case_result = {
            "case_id": case.id,
            "query": case.query,
            "relevant_chunk_ids": case.relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "recall_at_1": compute_recall_at_k(
                retrieved_ids,
                case.relevant_chunk_ids,
                1,
            ),
            "recall_at_3": compute_recall_at_k(
                retrieved_ids,
                case.relevant_chunk_ids,
                3,
            ),
            "recall_at_5": compute_recall_at_k(
                retrieved_ids,
                case.relevant_chunk_ids,
                5,
            ),
            "reciprocal_rank_at_5": compute_reciprocal_rank(
                retrieved_ids,
                case.relevant_chunk_ids,
            ),
        }

        per_query_results.append(case_result)

    # 7. 检查每道题都有对应结果。
    if len(per_query_results) != len(cases):
        raise ValueError(
            "Per-query result count does not match query count"
        )

    # 8. 计算各 Query 的平均 Recall，每道题权重相同。
    recall_at_1 = sum(
        result["recall_at_1"]
        for result in per_query_results
    ) / len(cases)

    recall_at_3 = sum(
        result["recall_at_3"]
        for result in per_query_results
    ) / len(cases)

    recall_at_5 = sum(
        result["recall_at_5"]
        for result in per_query_results
    ) / len(cases)

    # 9. 排名截断到 Top-5，因此报告 MRR@5。
    mrr_at_5 = compute_mrr(
        retrieved_rankings=retrieved_rankings,
        relevant_ids_per_query=relevant_ids_per_query,
    )

    # 10. 调用已通过测试的失败分析函数。
    failures = identify_retrieval_failures(per_query_results)

    # 11. 构造汇总信息，保留原始浮点数。
    summary = {
        "dataset_version": "retrieval_cases_v0.1",
        "retriever": "BM25",
        "chunking": "paragraph",
        "top_k": 5,
        "query_count": len(cases),
        "chunk_count": len(all_chunks),
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "mrr_at_5": mrr_at_5,

        # 统计存在问题的 Query 数量，不是失败标签总数。
        "failure_count": len(failures),

        # 记录输入文件指纹，便于识别数据版本。
        "dataset_sha256": compute_file_sha256(
            "data/retrieval_cases_v0.1.jsonl"
        ),
        "document_sha256": {
            "python-001": compute_file_sha256(
                "data/rag_docs/python-001.txt"
            ),
            "network-001": compute_file_sha256(
                "data/rag_docs/network-001.txt"
            ),
            "database-001": compute_file_sha256(
                "data/rag_docs/database-001.txt"
            ),
            "redis-001": compute_file_sha256(
                "data/rag_docs/redis-001.txt"
            ),
        },
    }

    # 12. 合并汇总、逐题结果和失败详情。
    experiment_result = {
        "summary": summary,
        "per_query_results": per_query_results,
        "failures": failures,
    }

    # 13. 创建输出目录，并保存 JSON。
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            experiment_result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # 14. 打印实验汇总。
    print("\nBM25 Retrieval Baseline")
    print("Dataset: retrieval_cases_v0.1")
    print("Retriever: BM25")
    print("Chunking: paragraph")
    print("Retrieval depth: 5")
    print("Documents:", len(documents))
    print("Queries:", len(cases))
    print("Chunks:", len(all_chunks))

    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@3: {recall_at_3:.4f}")
    print(f"Recall@5: {recall_at_5:.4f}")
    print(f"MRR@5:    {mrr_at_5:.4f}")
    print("Failure count:", len(failures))

    print("Saved rankings:", len(retrieved_rankings))
    print("Saved Gold lists:", len(relevant_ids_per_query))
    print("Saved per-query results:", len(per_query_results))
    print("Results saved to:", output_path)

    # 15. 打印第一道题的实际排名。
    first_case = cases[0]

    print("\nFirst case ID:", first_case.id)
    print("Query:", first_case.query)
    print("Gold:", relevant_ids_per_query[0])
    print("Top-5 ranking:")

    for rank, chunk_id in enumerate(
        retrieved_rankings[0],
        start=1,
    ):
        print(f"{rank}. {chunk_id}")


# 入口位于所有函数定义之后。
if __name__ == "__main__":
    main()