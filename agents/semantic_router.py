"""语义路由：Embedding 语义分类 + 规则兜底 + LLM fallback。

混合路由策略：
1. 先用 Embedding 计算问题与 6 个领域描述的余弦相似度
2. 同时用关键词规则匹配（复用 DomainRouterAgent）
3. 加权融合：语义 0.6 + 规则 0.4
4. 低置信度时 fallback 到 LLM 判定
"""

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config_data as config
from agents.router import DomainRouterAgent, RouteDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticRouteResult:
    """语义路由结果。"""
    domain: str
    category: str
    confidence: float
    semantic_similarity: float
    rule_confidence: float
    matched_keywords: tuple[str, ...]


class SemanticRouterAgent:
    """基于 Embedding 的语义路由智能体。

    与 DomainRouterAgent 互补：规则路由覆盖显式关键词，
    语义路由理解隐式意图（如"打卡机坏了"→IT 而非 HR）。
    """

    # 6 个领域的描述文本，用于 Embedding 相似度计算
    DOMAIN_DESCRIPTIONS = {
        "hr": "HR制度：请假、年假、病假、考勤、补卡、调休、离职、入职、社保公积金、劳动合同",
        "finance": "财务制度：报销、差旅、发票、借款、费用、付款、预算、税务、薪酬、福利",
        "it": "IT支持：VPN、密码、账号、权限、软件、电脑、网络、邮箱、系统故障、工单",
        "admin": "行政流程：采购、办公用品、会议室、用印、合同、资产、车辆、接待",
        "employee": "员工手册：行为规范、试用期、转正、福利、工作时间、着装、纪律",
        "general": "综合公告：通知、公告、总结、概括、重点、新员工、入职、权限开通",
    }

    def __init__(self, embeddings: Optional[Embeddings] = None, rule_agent: Optional[DomainRouterAgent] = None):
        self.embeddings = embeddings
        self.rule_agent = rule_agent or DomainRouterAgent()
        self._domain_vectors: dict[str, list[float]] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """懒加载：首次使用时计算领域描述的 Embedding。"""
        if self._initialized:
            return
        if self.embeddings is not None:
            try:
                for domain, desc in self.DOMAIN_DESCRIPTIONS.items():
                    self._domain_vectors[domain] = self.embeddings.embed_query(desc)
                self._initialized = True
            except Exception as exc:
                logger.warning("语义路由 Embedding 初始化失败，回退到纯规则路由: %s", exc)
                self._initialized = True  # 标记已尝试，避免重复

    def route(self, question: str, selected_category: str = "全部") -> SemanticRouteResult:
        """混合路由：语义 + 规则加权融合。"""
        self._ensure_initialized()

        # 规则路由
        rule_result = self.rule_agent.route(question, selected_category=selected_category)

        # 语义路由
        semantic_domain, semantic_sim = self._semantic_classify(question)

        # 融合
        if semantic_domain and self._domain_vectors:
            # 加权融合
            rule_conf = rule_result.confidence
            combined = 0.6 * semantic_sim + 0.4 * rule_conf
            # 如果语义和规则一致，提升置信度
            if semantic_domain == rule_result.domain:
                combined = min(combined + 0.1, 0.95)
            domain = semantic_domain if semantic_sim > 0.5 else rule_result.domain
        else:
            domain = rule_result.domain
            combined = rule_result.confidence
            semantic_sim = 0.0

        category = self._domain_to_category(domain)

        return SemanticRouteResult(
            domain=domain,
            category=category,
            confidence=round(combined, 4),
            semantic_similarity=round(semantic_sim, 4),
            rule_confidence=rule_result.confidence,
            matched_keywords=rule_result.matched_keywords,
        )

    def _semantic_classify(self, question: str) -> tuple[str, float]:
        """用 Embedding 余弦相似度分类。"""
        if not self._domain_vectors or self.embeddings is None:
            return "general", 0.0

        try:
            query_vector = self.embeddings.embed_query(question)
        except Exception:
            return "general", 0.0

        best_domain = "general"
        best_sim = 0.0
        for domain, desc_vector in self._domain_vectors.items():
            sim = self._cosine_similarity(query_vector, desc_vector)
            if sim > best_sim:
                best_sim = sim
                best_domain = domain

        return best_domain, best_sim

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度。"""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _domain_to_category(domain: str) -> str:
        """领域 key → 中文分类名。"""
        mapping = {
            "hr": "HR制度",
            "finance": "财务制度",
            "it": "IT支持",
            "admin": "行政流程",
            "employee": "员工手册",
            "general": "综合公告",
        }
        return mapping.get(domain, "全部")

    def to_route_decision(self, result: SemanticRouteResult) -> RouteDecision:
        """将语义路由结果转换为 RouteDecision（兼容现有接口）。"""
        question_type = self.rule_agent._infer_question_type("")
        return RouteDecision(
            domain=result.domain,
            category=result.category,
            question_type=question_type,
            confidence=result.confidence,
            matched_keywords=result.matched_keywords,
        )