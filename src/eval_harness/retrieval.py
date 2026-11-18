from .models import Chunk, RetrievedChunk
from rank_bm25 import BM25Okapi
import re

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())

    return tokens

# BM25Retriever这种类是“既保存状态，又执行行为的业务类”。
# BM25Retriever
# ├── chunks      ← 他掌握的资料
# ├── __init__()  ← 上岗时把资料交给他
# └── retrieve()  ← 让他执行搜索
class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("chunks must not be empty")
        
        self.chunks = chunks

        #self.tokenized_corpus = []
        #for chunk in chunks:
        #   tokens = tokenize(chunk.text)
        #   self.tokenized_corpus.append(tokens)
        #价于：
        self.tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        results: list[RetrievedChunk] = []

        for chunk, score in zip(self.chunks, scores):
            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                score=float(score),
            )

            results.append(retrieved_chunk)

        # sorted(results) 给results排序
        # key=lambda item: item.score 其实可以通俗理解成一个很短的小函数。它基本等价于：def get_score(item): return item.score
        # reverse=True 从大到小排序
        results = sorted(results, key=lambda item: item.score, reverse=True)

        # 只返回前 top_k 个
        return results[:top_k]