import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import uuid4

from langchain_core.documents import Document

import config_data as config
from services.exceptions import StorageError
from services.text_analyzer import tokenize


@dataclass(frozen=True)
class LexicalHit:
    document: Document
    raw_score: float
    score: float


class BM25LexicalIndex:
    """Persistent BM25 index for independent lexical candidate retrieval."""

    _lock = RLock()

    def __init__(self, path=None):
        self.path = Path(path or config.LEXICAL_INDEX_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def add_document(self, document_id, chunks, metadatas):
        records = []
        for index, (content, metadata) in enumerate(zip(chunks, metadatas, strict=True)):
            records.append(
                {
                    "chunk_id": f"{document_id}-{index}",
                    "content": content,
                    "metadata": {**metadata, "document_id": document_id, "chunk_index": index},
                }
            )
        with self._lock:
            current = [
                item
                for item in self._read()
                if item.get("metadata", {}).get("document_id") != document_id
            ]
            self._write(current + records)

    def delete_document(self, document_id):
        with self._lock:
            current = self._read()
            remaining = [
                item
                for item in current
                if item.get("metadata", {}).get("document_id") != document_id
            ]
            if len(remaining) != len(current):
                self._write(remaining)

    def replace_all(self, chunks):
        with self._lock:
            self._write(chunks)

    def is_empty(self):
        return not self._read()

    def search(self, query, category="全部", limit=None):
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        records = self._read()
        if not records:
            return []

        tokenized_documents = [tokenize(item.get("content", "")) for item in records]
        average_length = sum(map(len, tokenized_documents)) / len(tokenized_documents)
        document_frequency = {
            token: sum(token in set(document_tokens) for document_tokens in tokenized_documents)
            for token in query_tokens
        }
        hits = []
        for record, document_tokens in zip(records, tokenized_documents, strict=True):
            if category != "全部" and record.get("metadata", {}).get("category") != category:
                continue
            raw_score = self._bm25_score(
                query_tokens,
                document_tokens,
                document_frequency,
                len(records),
                average_length,
            )
            if raw_score <= 0:
                continue
            confidence = 1.0 - math.exp(-raw_score / config.bm25_saturation)
            hits.append(
                LexicalHit(
                    document=Document(
                        page_content=record["content"],
                        metadata=record.get("metadata", {}),
                    ),
                    raw_score=round(raw_score, 4),
                    score=round(confidence, 4),
                )
            )
        hits.sort(key=lambda item: item.raw_score, reverse=True)
        return hits[: limit or config.retrieval_candidate_count]

    @staticmethod
    def _bm25_score(query_tokens, document_tokens, document_frequency, corpus_size, average_length):
        frequencies = Counter(document_tokens)
        document_length = len(document_tokens)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            frequency_in_corpus = document_frequency[token]
            inverse_document_frequency = math.log(
                1 + (corpus_size - frequency_in_corpus + 0.5) / (frequency_in_corpus + 0.5)
            )
            normalization = frequency + config.bm25_k1 * (
                1 - config.bm25_b
                + config.bm25_b * document_length / max(average_length, 1.0)
            )
            score += inverse_document_frequency * frequency * (config.bm25_k1 + 1) / normalization
        return score

    def _read(self):
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StorageError(f"BM25 索引文件已损坏：{self.path}") from exc
        if not isinstance(records, list):
            raise StorageError(f"BM25 索引格式错误：{self.path}")
        return records

    def _write(self, records):
        temporary_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
