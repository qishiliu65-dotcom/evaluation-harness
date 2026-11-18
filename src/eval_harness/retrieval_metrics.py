


def compute_recall_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")

    top_k_ids = retrieved_chunk_ids[:k]

    # 把两个列表转换成集合； ["A", "B", "C"] 为  {"A", "B", "C"}
    top_k_set = set(top_k_ids)
    gold_set = set(relevant_chunk_ids)

    # & 是集合 set 支持的交集运算符
    recall = len(top_k_set & gold_set) / len(gold_set)

    return recall


def compute_reciprocal_rank(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str]) -> float:
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")

    gold_set = set(relevant_chunk_ids)

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in gold_set:
            return  1 / rank
        
    return  0


def compute_mrr(retrieved_rankings: list[list[str]], relevant_ids_per_query: list[list[str]]) -> float:
    """计算多条 Query 的平均倒数排名，包括未命中的 Query。"""

    # 1. 每组检索结果必须有一组对应的 Gold。
    if len(retrieved_rankings) != len(relevant_ids_per_query):
        raise ValueError(
            "Retrieved rankings and Gold labels must have the same query count"
        )

    # 2. 至少需要一道 Query，才能计算平均值。
    if not retrieved_rankings:
        raise ValueError("Query lists must not be empty")

    # 3. 保存每道 Query 的 RR。
    reciprocal_ranks: list[float] = []

    # 4. 按位置配对每道 Query 的检索结果和 Gold。`zip()` 把两个列表按位置配对
    # 比如配对前retrieved = ["A", "B", "C"] relevant = ["A"] 配对后(["A", "B", "C"], ["A"])
    for retrieved, relevant in zip(
        retrieved_rankings,
        relevant_ids_per_query,
    ):
        # 5. 复用已有函数，计算当前 Query 的 RR。
        rr = compute_reciprocal_rank(
            retrieved_chunk_ids=retrieved,
            relevant_chunk_ids=relevant,
        )

        # 6. 保存当前 RR，未命中的 0.0 也保留。
        reciprocal_ranks.append(rr)

    # 7. 全部 Query 的 RR 之和，除以 Query 数量。
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
