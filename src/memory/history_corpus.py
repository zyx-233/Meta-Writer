"""HistoryCorpus: generated-section history store with dense retrieval."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..utils.embedding_model import EmbeddingModel


@dataclass
class HistoryItem:
    item_id: str
    item_type: str
    section_id: str
    text: str


class HistoryCorpus:
    """历史语料库数据层：存储条目、重建 dense 索引、执行 cosine 检索和落盘。"""

    def __init__(self, embed_model: Optional["EmbeddingModel"] = None) -> None:
        self._items: Dict[str, HistoryItem] = {}
        self._embed_model = embed_model
        self._embeddings: Optional[np.ndarray] = None
        self._index_item_ids: List[str] = []

    def add_item(self, item: HistoryItem) -> None:
        """按 item_id 去重写入；新 item 会覆盖同 id 的旧 item。"""
        if not item.item_id:
            raise ValueError("HistoryItem.item_id must not be empty")
        self._items[item.item_id] = item

    def remove_section(self, section_id: str) -> None:
        """回滚时删除某一节写入的所有历史 item，并重建当前 dense 索引。"""
        to_delete = [
            item_id
            for item_id, item in self._items.items()
            if item.section_id == section_id
        ]
        for item_id in to_delete:
            del self._items[item_id]
        if to_delete:
            self.build_index()

    def build_index(self) -> None:
        """重建 dense 索引；空文本 item 不进入索引。"""
        self._embeddings = None
        self._index_item_ids = []

        if self._embed_model is None:
            return

        texts_to_embed: List[str] = []
        index_item_ids: List[str] = []
        for item in self._items.values():
            text = (item.text or "").strip()
            if not text:
                continue
            texts_to_embed.append(text)
            index_item_ids.append(item.item_id)

        if not texts_to_embed:
            return

        # EmbeddingModel 返回 L2 归一化向量，后续内积即 cosine similarity。
        self._embeddings = self._embed_model.embed(texts_to_embed)
        self._index_item_ids = index_item_ids

    def retrieve_dense(self, query: str, top_k: int = 8) -> List[dict]:
        """用 dense embedding 检索历史 item，score 为 cosine similarity。"""
        if self._embed_model is None:
            return []
        if self._embeddings is None or not self._index_item_ids:
            self.build_index()
        if self._embeddings is None or not self._index_item_ids:
            return []
        if not (query or "").strip():
            return []

        q_emb = self._embed_model.embed([query])
        sims: np.ndarray = self._embeddings @ q_emb[0]
        k = min(top_k, len(sims))
        if k <= 0:
            return []

        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results: List[dict] = []
        for idx in top_indices:
            item_id = self._index_item_ids[int(idx)]
            item = self._items[item_id]
            results.append({
                "item_id": item.item_id,
                "item_type": item.item_type,
                "section_id": item.section_id,
                "text": item.text,
                "score": float(sims[idx]),
            })
        return results

    def save_to_disk(self, path: str | Path) -> None:
        """保存为最小 JSON 结构，不写入 embedding 缓存。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(item) for item in self._items.values()]
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_from_disk(
        cls,
        path: str | Path,
        embed_model: Optional["EmbeddingModel"] = None,
    ) -> "HistoryCorpus":
        """从 JSON 文件恢复语料库；不存在时返回空语料库。"""
        corpus = cls(embed_model=embed_model)
        source = Path(path)
        if not source.exists():
            return corpus

        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("HistoryCorpus file must contain a JSON list")

        for raw in data:
            if not isinstance(raw, dict):
                continue
            corpus.add_item(HistoryItem(
                item_id=str(raw.get("item_id", "")),
                item_type=str(raw.get("item_type", "")),
                section_id=str(raw.get("section_id", "")),
                text=str(raw.get("text", "")),
            ))
        corpus.build_index()
        return corpus

    def item_count(self) -> int:
        """返回当前存储的 item 数量。"""
        return len(self._items)
