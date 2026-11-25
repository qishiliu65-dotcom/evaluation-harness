import json
from pathlib import Path

from .chunking import chunk_document_by_paragraph
from .dataset import load_retrieval_cases
from .ingestion import load_text_document
from .models import Chunk
from .rag_pipeline import run_rag
from .reproducibility import compute_file_sha256
from .retrieval import BM25Retriever
from .retrieval_metrics import (
    compute_mrr,
    compute_recall_at_k,
    compute_reciprocal_rank,
)
from .systems import SystemUnderTest


def run_rag_experiment(
    llm: SystemUnderTest,
    model_name: str,
    output_path: Path,
    top_k: int = 3,
) -> dict:
    """
    批量运行 RAG，计算检索指标并保存实验报告。

    1. 加载四份文档并按段落切分。
    2. 创建共享的 BM25Retriever。
    3. 加载评测题并检查 Gold Chunk ID。
    4. 每道题调用一次 run_rag()。
    5. 保存答案、上下文、检索排名和单题指标。
    6. 汇总 Mean Recall@K 和 MRR@K。
    7. 保存包含输入指纹的 JSON 报告。
    8. 返回实验报告字典。

    注意：
    - 当前只评估检索质量，不评估生成答案的质量。
    - model_name 只是报告标签，应与传入的 llm 保持一致。
    - 相对文件路径以当前工作目录为基准。
    - 实验运行期间不应修改输入数据文件。
    """

    # 1. 在读取文件和调用模型之前检查参数。
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    # 2. 加载四份文档。
    # 第二个参数是 Document ID，会用于生成 Chunk ID。
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

    # 3. 按段落切分，并把各文档的 Chunk 汇总到一个列表。
    all_chunks: list[Chunk] = []

    for document in documents:
        chunks = chunk_document_by_paragraph(document)

        # extend 加入列表中的各个元素，不产生嵌套列表。
        all_chunks.extend(chunks)

    # 4. 只创建一次检索器，供所有评测题共用。
    retriever = BM25Retriever(all_chunks)

    # 5. 加载评测题。
    cases = load_retrieval_cases(
        "data/retrieval_cases_v0.1.jsonl"
    )

    if not cases:
        raise ValueError("Retrieval cases must not be empty")

    # 6. 在调用模型之前，检查所有题目的 Gold。
    available_chunk_ids = {
        chunk.id
        for chunk in all_chunks
    }

    for case in cases:
        if not case.relevant_chunk_ids:
            raise ValueError(
                f"relevant_chunk_ids must not be empty: {case.id}"
            )

        # 差集：找出 Gold 中不存在于知识库的 Chunk ID。
        missing_ids = (
            set(case.relevant_chunk_ids) - available_chunk_ids
        )

        if missing_ids:
            raise ValueError(
                f"Unknown relevant_chunk_ids for {case.id}: "
                f"{sorted(missing_ids)}"
            )

    # 7. 准备结果容器。
    # 前两个列表用于计算多题 MRR，位置必须一一对应。
    retrieved_rankings: list[list[str]] = []
    relevant_ids_per_query: list[list[str]] = []
    per_query_results: list[dict] = []

    # 8. 按数据集顺序逐题运行 RAG。
    for case in cases:
        # 每道题只调用一次 run_rag。
        # Gold 不传入生成流程，仅在运行后用于评测。
        rag_result = run_rag(
            query=case.query,
            retriever=retriever,
            llm=llm,
            top_k=top_k,
        )

        # 保留检索排名顺序，不能转换成 set。
        retrieved_ids = rag_result.retrieved_chunk_ids

        # 在同一轮循环内保存排名及对应 Gold。
        retrieved_rankings.append(retrieved_ids)
        relevant_ids_per_query.append(
            case.relevant_chunk_ids
        )

        # 9. 计算单题 Recall@K。
        recall_at_k = compute_recall_at_k(
            retrieved_chunk_ids=retrieved_ids,
            relevant_chunk_ids=case.relevant_chunk_ids,
            k=top_k,
        )

        # 10. 计算单题 RR。
        # 输入只有 Top-K 排名，因此这里也是截断到 K 的 RR。
        reciprocal_rank_at_k = compute_reciprocal_rank(
            retrieved_chunk_ids=retrieved_ids,
            relevant_chunk_ids=case.relevant_chunk_ids,
        )

        # 11. 保存当前问题的运行证据和指标。
        case_result = {
            "case_id": case.id,
            "query": case.query,
            "relevant_chunk_ids": case.relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "context": rag_result.context,
            "answer": rag_result.answer,
            "top_k": top_k,
            "recall_at_k": recall_at_k,
            "reciprocal_rank_at_k": reciprocal_rank_at_k,

            # 答案已采集，但未执行答案质量评测。
            "answer_evaluation": {
                "status": "not_evaluated",
            },
        }

        per_query_results.append(case_result)

    # 12. 对所有题目的 Recall 求平均。
    # 每道题权重相同，未命中的 0 分也必须保留。
    # 前面已拒绝空数据集，因此分母不会为 0。
    mean_recall_at_k = sum(
        result["recall_at_k"]
        for result in per_query_results
    ) / len(cases)

    # 13. 复用已有函数计算 MRR@K。
    # 这里只计算指标，不会重复检索或调用模型。
    mrr_at_k = compute_mrr(
        retrieved_rankings=retrieved_rankings,
        relevant_ids_per_query=relevant_ids_per_query,
    )

    # 14. 记录数据集文件指纹。
    dataset_sha256 = compute_file_sha256(
        "data/retrieval_cases_v0.1.jsonl"
    )

    # 15. 记录四份知识文档的文件指纹。
    document_sha256 = {
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
    }

    # 16. 构造汇总，保留原始浮点数。
    summary = {
        "dataset_version": "retrieval_cases_v0.1",
        "retriever": "BM25",
        "generator": model_name,
        "chunking": "paragraph",
        "top_k": top_k,
        "query_count": len(cases),
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "mean_recall_at_k": mean_recall_at_k,
        "mrr_at_k": mrr_at_k,
        "dataset_sha256": dataset_sha256,
        "document_sha256": document_sha256,
        "answer_evaluation": {
            "status": "not_evaluated",
        },
    }

    # 17. 合并汇总信息和逐题结果。
    experiment_result = {
        "summary": summary,
        "per_query_results": per_query_results,
    }

    # 18. 创建输出目录，已存在时不报错。
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 19. 保存 JSON。
    # "w" 会覆盖同名文件，因此 Mock 和真实实验应使用不同文件名。
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            experiment_result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # 20. 返回与保存内容一致的报告。
    return experiment_result