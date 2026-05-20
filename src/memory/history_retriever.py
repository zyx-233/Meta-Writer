"""HistoryRetriever: dense retrieval over HistoryCorpus."""
from __future__ import annotations

from typing import List

from .history_corpus import HistoryCorpus


class HistoryRetriever:
    """历史语料库检索层：直接使用 dense cosine 检索，不再使用 HyDE / RRF。"""

    def __init__(self, corpus: HistoryCorpus) -> None:
        self._corpus = corpus

    def retrieve(self, query: str, top_k: int = 8) -> List[dict]:
        """检索历史语料；返回按 cosine similarity 排序的 history item。"""
        if self._corpus.item_count() == 0:
            return []
        return self._corpus.retrieve_dense(query, top_k=top_k)
