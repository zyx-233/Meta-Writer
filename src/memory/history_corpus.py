"""HistoryCorpus: generated-section history store with BM25 retrieval."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..references.corpus import _BM25Index, _tokenize


@dataclass
class HistoryItem:
    item_id: str
    item_type: str
    section_id: str
    text: str


class HistoryCorpus:
    """历史语料库数据层：只负责存储、索引、BM25 检索和落盘。"""

    def __init__(self) -> None:
        self._items: Dict[str, HistoryItem] = {}
        self._index: Optional[_BM25Index] = None
        self._index_item_ids: List[str] = []

    def add_item(self, item: HistoryItem) -> None:
        """按 item_id 去重写入；新 item 会覆盖同 id 的旧 item。"""
        if not item.item_id:
            raise ValueError("HistoryItem.item_id must not be empty")
        self._items[item.item_id] = item

    def remove_section(self, section_id: str) -> None:
        """回滚时删除某一节写入的所有历史 item。"""
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
        """重建 BM25 索引；空文本 item 不进入索引。"""
        documents: List[List[str]] = []
        index_item_ids: List[str] = []

        for item in self._items.values():
            text = (item.text or "").strip()
            if not text:
                continue
            tokens = _tokenize(text)
            if not tokens:
                continue
            documents.append(tokens)
            index_item_ids.append(item.item_id)

        self._index_item_ids = index_item_ids
        self._index = _BM25Index(documents) if documents else None

    def retrieve_bm25(self, query: str, top_k: int = 8) -> List[dict]:
        """返回 BM25 命中的历史 item，结果保留 item 基本字段和 score。"""
        if self._index is None:
            self.build_index()
        if self._index is None or not self._index_item_ids:
            return []

        query_tokens = _tokenize(query or "")
        if not query_tokens:
            return []

        hits = self._index.top_k(query_tokens, top_k)
        results: List[dict] = []
        for doc_idx, score in hits:
            item_id = self._index_item_ids[doc_idx]
            item = self._items[item_id]
            results.append({
                "item_id": item.item_id,
                "item_type": item.item_type,
                "section_id": item.section_id,
                "text": item.text,
                "score": score,
            })
        return results

    def save_to_disk(self, path: str | Path) -> None:
        """保存为最小 JSON 结构，不写入索引缓存。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(item) for item in self._items.values()]
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_from_disk(cls, path: str | Path) -> "HistoryCorpus":
        """从 JSON 文件恢复语料库；不存在时返回空语料库。"""
        corpus = cls()
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
        """返回当前存储的 item 数量"""
        return len(self._items)
