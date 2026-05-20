"""HistoryRetriever: BM25 + HyDE-memory + RRF over HistoryCorpus."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from ..references.fusion import reciprocal_rank_fusion
from ..references.hyde import HyDEGenerator
from .history_corpus import HistoryCorpus

if TYPE_CHECKING:
    from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class HistoryRetriever:
    """历史语料检索层：两路 BM25 检索后用 RRF 融合。"""

    def __init__(
        self,
        corpus: HistoryCorpus,
        llm_client: Optional["LLMClient"] = None,
    ) -> None:
        self._corpus = corpus
        self._llm_client = llm_client
        self._hyde = HyDEGenerator()

    def retrieve(self, query: str, top_k: int = 8) -> List[dict]:
        """检索历史语料；HyDE 失败时自动退化为纯 BM25。"""
        if self._corpus.item_count() == 0:
            return []

        bm25_results = self._corpus.retrieve_bm25(query, top_k=top_k * 4)
        bm25_ids = [r["item_id"] for r in bm25_results]

        hyde_results: List[dict] = []
        hyde_query = self._hyde.generate_for_memory(query, self._llm_client)
        if hyde_query:
            hyde_results = self._corpus.retrieve_bm25(hyde_query, top_k=top_k * 4)
        else:
            logger.info("HistoryRetriever: HyDE memory query failed; fallback to BM25 only")
        hyde_ids = [r["item_id"] for r in hyde_results]

        if not bm25_ids and not hyde_ids:
            return []

        by_id: Dict[str, dict] = {}
        for result in bm25_results + hyde_results:
            item_id = result["item_id"]
            if item_id not in by_id or result["score"] > by_id[item_id]["score"]:
                by_id[item_id] = dict(result)

        fused = reciprocal_rank_fusion([bm25_ids, hyde_ids])
        results: List[dict] = []
        for item_id, rrf_score in fused[:top_k]:
            result = dict(by_id[item_id])
            result["score"] = rrf_score
            results.append(result)
        return results
