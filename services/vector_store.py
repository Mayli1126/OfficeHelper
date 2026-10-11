import hashlib
import logging
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config_data as config
from services.exceptions import RetrievalError
from services.lexical_index import BM25LexicalIndex
from services.model_factory import create_embeddings, create_embeddings_secondary
from services.reranker import CrossEncoderReranker
from services.semantic_chunker import SemanticChunker
from services.text_analyzer import tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    distance: float
    vector_score: float
    lexical_score: float
    relevance_score: float
    rrf_score: float = 0.0
    lexical_raw_score: float = 0.0
    retrieval_sources: tuple[str, ...] = ()
    embedding2_score: float = 0.0


class OfficeMateVectorStore:
    """Compatibility facade implementing vector + BM25 hybrid retrieval.

    Supports multi-embedding fusion: two embedding models + BM25,
    fused via weighted RRF across three channels.
    """

    def __init__(self, lexical_index=None):
        config.ensure_runtime_dirs()

        # Primary embedding（通过模型工厂创建，支持多厂商）
        self.embedding = create_embeddings()

        # Secondary embedding (multi-embedding fusion)
        self.embedding2 = create_embeddings_secondary()
        if config.embedding2_model_name:
            try:
                self.embedding2 = DashScopeEmbeddings(
                    model=config.embedding2_model_name,
                    dashscope_api_key=api_key,
                )
            except Exception as exc:
                logger.warning("第二个 Embedding 模型初始化失败: %s", exc)

        self.splitter = SemanticChunker(
            max_chunk_size=config.chunk_size,
            min_chunk_size=config.chunk_min_size,
        )

        # Primary vector store
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

        # Secondary vector store (if embedding2 available)
        self.vector_store2 = None
        if self.embedding2 is not None:
            self.vector_store2 = Chroma(
                collection_name=config.collection_name + "_v2",
                embedding_function=self.embedding2,
                persist_directory=config.persist_directory,
            )

        self.lexical_index = lexical_index or BM25LexicalIndex()
        self._bootstrap_lexical_index()
        self._reranker = None

    def add_document(self, document_id, text, metadata):
        chunks = self.splitter.split_text(text, metadata=metadata) if len(text) > config.max_split_char_number else [text]
        metadatas = [
            {**metadata, "document_id": document_id, "chunk_index": index}
            for index, _ in enumerate(chunks)
        ]
        ids = [f"{document_id}-{index}" for index, _ in enumerate(chunks)]

        # Primary vector store
        self.vector_store.add_texts(chunks, metadatas=metadatas, ids=ids)

        # Secondary vector store (if available)
        if self.vector_store2 is not None:
            try:
                self.vector_store2.add_texts(chunks, metadatas=metadatas, ids=ids)
            except Exception as exc:
                logger.warning("第二个 Embedding 索引添加失败: %s", exc)

        # BM25 index
        try:
            self.lexical_index.add_document(document_id, chunks, metadatas)
        except Exception:
            self.vector_store.delete(ids=ids)
            if self.vector_store2 is not None:
                self.vector_store2.delete(ids=ids)
            raise
        return len(chunks)

    def delete_document(self, document_id, chunk_count=0):
        chunk_total = int(chunk_count or 0)
        if chunk_total > 0:
            self.vector_store.delete(ids=[f"{document_id}-{index}" for index in range(chunk_total)])
        else:
            self.vector_store.delete(where={"document_id": document_id})
        self.lexical_index.delete_document(document_id)

    def search(self, query, category="全部", limit=None):
        result_limit = limit or config.retrieval_result_count
        candidate_count = max(config.retrieval_candidate_count, result_limit)
        filters = None if category == "全部" else {"category": category}
        try:
            # Primary vector + BM25
            vector_candidates = self.vector_store.similarity_search_with_score(
                query,
                k=candidate_count,
                filter=filters,
            )
            lexical_candidates = self.lexical_index.search(
                query,
                category=category,
                limit=candidate_count,
            )

            # Secondary vector (if available)
            vector2_candidates = []
            if self.vector_store2 is not None:
                try:
                    vector2_candidates = self.vector_store2.similarity_search_with_score(
                        query,
                        k=candidate_count,
                        filter=filters,
                    )
                except Exception as exc:
                    logger.warning("第二个 Embedding 检索失败: %s", exc)
        except Exception as exc:
            raise RetrievalError(f"混合知识检索失败：{exc}") from exc

        fused = self._fuse_results(vector_candidates, lexical_candidates, vector2_candidates)
        # Cross-Encoder 精排
        if config.enable_reranker and fused:
            fused = self._rerank(query, fused)
        return [result for result in fused if result.relevance_score >= config.min_relevance_score][
            :result_limit
        ]

    def _rerank(self, query: str, candidates: list) -> list:
        """使用 Cross-Encoder 对候选结果进行语义精排。"""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=config.reranker_model,
                top_k=config.reranker_top_k,
            )
        # 用 document 对象调用精排器
        docs = [c.document for c in candidates]
        ranked_docs = self._reranker.rerank(query, docs, top_k=config.reranker_top_k)
        ranked_ids = {id(d) for d in ranked_docs}
        # 保持精排后的顺序
        ranked_results = [c for c in candidates if id(c.document) in ranked_ids]
        return ranked_results

    @classmethod
    def _fuse_results(cls, vector_candidates, lexical_candidates, vector2_candidates=None):
        """三通道 RRF 融合：向量1 + 向量2 + BM25。"""
        candidates = {}
        # Channel 1: Primary vector
        for rank, (document, distance) in enumerate(vector_candidates, start=1):
            key = cls._chunk_key(document)
            safe_distance = max(float(distance), 0.0)
            candidates.setdefault(key, {"document": document})
            candidates[key].update({
                "distance": safe_distance,
                "vector_score": cls._vector_confidence(safe_distance),
                "vector_rank": rank,
            })

        # Channel 2: Secondary vector (multi-embedding)
        if vector2_candidates:
            for rank, (document, distance) in enumerate(vector2_candidates, start=1):
                key = cls._chunk_key(document)
                safe_distance = max(float(distance), 0.0)
                candidates.setdefault(key, {"document": document})
                candidates[key].update({
                    "embedding2_score": cls._vector_confidence(safe_distance),
                    "embedding2_rank": rank,
                })

        # Channel 3: BM25 lexical
        for rank, hit in enumerate(lexical_candidates, start=1):
            key = cls._chunk_key(hit.document)
            candidates.setdefault(key, {"document": hit.document})
            candidates[key].update({
                "lexical_score": hit.score,
                "lexical_raw_score": hit.raw_score,
                "lexical_rank": rank,
            })

        maximum_rrf = 1.0 / (config.rrf_rank_constant + 1)
        results = []
        for candidate in candidates.values():
            vector_score = candidate.get("vector_score", 0.0)
            lexical_score = candidate.get("lexical_score", 0.0)
            embedding2_score = candidate.get("embedding2_score", 0.0)
            weighted_rrf = 0.0
            sources = []

            if "vector_rank" in candidate:
                weighted_rrf += config.vector_score_weight / (
                    config.rrf_rank_constant + candidate["vector_rank"]
                )
                sources.append("vector")

            if "embedding2_rank" in candidate:
                weighted_rrf += config.embedding2_score_weight / (
                    config.rrf_rank_constant + candidate["embedding2_rank"]
                )
                sources.append("vector2")

            if "lexical_rank" in candidate:
                weighted_rrf += config.lexical_score_weight / (
                    config.rrf_rank_constant + candidate["lexical_rank"]
                )
                sources.append("bm25")

            rrf_score = min(weighted_rrf / maximum_rrf, 1.0)
            channel_confidence = max(vector_score, lexical_score, embedding2_score)
            relevance = channel_confidence * (0.7 + 0.3 * rrf_score)

            results.append(
                RetrievalResult(
                    document=candidate["document"],
                    distance=candidate.get("distance", config.vector_distance_scale),
                    vector_score=round(vector_score, 4),
                    lexical_score=round(lexical_score, 4),
                    relevance_score=round(relevance, 4),
                    rrf_score=round(rrf_score, 4),
                    lexical_raw_score=round(candidate.get("lexical_raw_score", 0.0), 4),
                    retrieval_sources=tuple(sources),
                    embedding2_score=round(embedding2_score, 4),
                )
            )

        results.sort(key=lambda item: (item.rrf_score, item.relevance_score), reverse=True)
        return results

    @staticmethod
    def _vector_confidence(distance):
        return round(max(0.0, 1.0 - distance / config.vector_distance_scale), 4)

    @staticmethod
    def _chunk_key(document):
        document_id = document.metadata.get("document_id")
        chunk_index = document.metadata.get("chunk_index")
        if document_id is not None and chunk_index is not None:
            return f"{document_id}-{chunk_index}"
        digest = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()
        return f"content-{digest}"

    def _bootstrap_lexical_index(self):
        if not self.lexical_index.is_empty():
            return
        stored = self.vector_store.get(include=["documents", "metadatas"])
        records = []
        for chunk_id, content, metadata in zip(
            stored.get("ids", []),
            stored.get("documents", []),
            stored.get("metadatas", []),
            strict=True,
        ):
            if not content:
                continue
            records.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": metadata or {},
                }
            )
        if records:
            self.lexical_index.replace_all(records)

    @staticmethod
    def _lexical_coverage(query, content):
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return 0.0
        return len(query_tokens & set(tokenize(content))) / len(query_tokens)
