"""Cross-Encoder 精排器：在 RRF 粗排后对候选进行语义精排。

基于 sentence-transformers 的 CrossEncoder，比 Bi-Encoder 更准确地判断
query-document 相关度，但计算成本更高，只在粗排后的 Top-N 候选上运行。
"""

import logging
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """基于 sentence-transformers Cross-Encoder 的精排器。

    使用 BAAI/bge-reranker-v2-zh 模型，对 (query, document) 配对打分。
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-zh"

    def __init__(self, model_name: Optional[str] = None, top_k: int = 8):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.top_k = top_k
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError as exc:
                logger.warning("sentence-transformers 不可用，回退到原始排序: %s", exc)
                self._model = _IdentityReranker()
        return self._model

    def rerank(self, query: str, candidates: list[Document], top_k: Optional[int] = None) -> list[Document]:
        """对候选文档进行精排，返回按相关度降序的 Top-K 文档。"""
        if not candidates:
            return []
        k = top_k or self.top_k
        model = self._load_model()
        contents = [doc.page_content for doc in candidates]
        scores = model.predict([(query, c) for c in contents])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]


class _IdentityReranker:
    """当 Cross-Encoder 不可用时的空实现，保持原始顺序。"""

    def predict(self, pairs):
        return [0.0] * len(pairs)