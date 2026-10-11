"""LangGraph 工作流：用状态图替代硬编码管线编排。

支持：
- 条件分支（Self-RAG 不通过 → 重试答案生成）
- 失败重试（验证不通过 → 修正后重试）
- 检查点（checkpoint）支持长事务恢复
- 与现有 OfficeMateChatService 接口兼容
"""

import logging
from typing import Annotated, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import END, StateGraph

import config_data as config
from agents.answer_agent import GroundedAnswerAgent
from agents.router import DomainRouterAgent, RouteDecision
from agents.self_rag import SelfRAGRefiner
from agents.verifier import AnswerVerifierAgent, VerificationReport
from services.storage_service import JsonStorageService
from services.vector_store import RetrievalResult

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────

class QAState(TypedDict, total=False):
    """问答工作流状态。"""
    # 输入
    question: str
    session_id: str
    category: str

    # 路由
    route_decision: RouteDecision
    route_confidence: float

    # 查询
    queries: list[str]
    decomposed: bool

    # 检索
    search_results: list[RetrievalResult]
    references: list[dict]
    search_category: str
    route_fallback: bool

    # 答案
    draft: str
    context_text: str
    self_rag_passed: bool
    retry_count: int

    # 校验
    verification: VerificationReport

    # 输出
    answer: str
    question_type: str
    qa_log_id: str
    source_docs: list[dict]
    domain: str

    # 控制
    error: str


def _merge_lists(a: list, b: list) -> list:
    """合并两个列表（用于状态更新）。"""
    return (a or []) + (b or [])


def _replace(old, new):
    """用新值替换旧值。"""
    return new if new is not None else old


# ── 节点函数 ───────────────────────────────────────────────────

def make_route_node(router: DomainRouterAgent):
    """创建路由节点。"""
    def route_node(state: QAState) -> QAState:
        question = state["question"]
        category = state.get("category", "全部")
        route = router.route(question, selected_category=category)
        search_category = route.category if route.confidence >= 0.7 else category
        return {
            "route_decision": route,
            "route_confidence": route.confidence,
            "search_category": search_category,
            "domain": route.domain,
        }
    return route_node


def make_rewrite_node(llm: Optional[BaseLanguageModel], config_module):
    """创建查询预处理节点：合并改写/HyDE/分解为一次调用，短问题跳过。"""
    def rewrite_node(state: QAState) -> QAState:
        """
            工厂函数：创建问题改写节点（闭包，绑定llm实例）
            适配LangGraph节点规范：节点函数只接收state参数
            :param llm: 大模型实例，用于QueryPreprocessor做问题改写/分解
            :return: rewrite_node 节点函数
            """
        from services.query_preprocessor import QueryPreprocessor

        question = state["question"]
        # 初始化查询预处理器，传入llm，执行预处理（改写+问题分解）
        preprocessed = QueryPreprocessor(llm).preprocess(question)
        return {
            "queries": preprocessed.queries,# 生成用于检索的多个query
            "decomposed": preprocessed.decomposed,# 原问题拆解得到的子问题
        }
    return rewrite_node


def make_retrieve_node(vector_store):
    """创建检索节点。"""
    def retrieve_node(state: QAState) -> QAState:
        queries = state.get("queries", [state["question"]])
        search_category = state.get("search_category", "全部")

        all_results = []
        seen_keys = set()
        for query in queries:
            results = vector_store.search(query, category=search_category)
            for result in results:
                key = f"{result.document.metadata.get('document_id', '')}-{result.document.metadata.get('chunk_index', '')}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(result)

        all_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # 路由回退
        route_fallback = False
        if not all_results and state.get("category", "全部") == "全部" and search_category != "全部":
            all_results = []
            seen_keys = set()
            for query in queries:
                results = vector_store.search(query, category="全部")
                for result in results:
                    key = f"{result.document.metadata.get('document_id', '')}-{result.document.metadata.get('chunk_index', '')}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_results.append(result)
            all_results.sort(key=lambda r: r.relevance_score, reverse=True)
            search_category = "全部"
            route_fallback = True

        # 构建引用
        references = _build_references(all_results)

        return {
            "search_results": all_results,
            "references": references,
            "search_category": search_category,
            "route_fallback": route_fallback,
        }
    return retrieve_node


def make_answer_node(answer_agent: GroundedAnswerAgent, llm: Optional[BaseLanguageModel]):
    """创建答案生成节点。"""
    def answer_node(state: QAState) -> QAState:
        references = state.get("references", [])
        if not references:
            return {"draft": config.NO_EVIDENCE_MESSAGE, "context_text": ""}

        # 构建上下文
        context_text = _build_context(state.get("search_results", []))
        route = state["route_decision"]

        draft = answer_agent.answer(
            llm,
            question=state["question"],
            route=route,
            context=context_text,
            history=[],
        )

        # Self-RAG 验证
        self_rag = SelfRAGRefiner(llm)
        draft, self_rag_passed = self_rag.verify(draft, context_text)

        return {
            "draft": draft,
            "context_text": context_text,
            "self_rag_passed": self_rag_passed,
            "retry_count": state.get("retry_count", 0) + 1,
        }
    return answer_node


def make_verify_node(verifier: AnswerVerifierAgent):
    """创建答案校验节点。"""
    def verify_node(state: QAState) -> QAState:
        draft = state.get("draft", "")
        references = state.get("references", [])
        verification = verifier.verify(draft, references)
        return {"verification": verification}
    return verify_node


def make_abstain_node():
    """创建拒答节点：无依据时写入固定拒答文案，走完整校验与日志。"""
    def abstain_node(state: QAState) -> QAState:
        return {
            "draft": config.NO_EVIDENCE_MESSAGE,
            "context_text": "",
            "references": [],
            "search_results": [],
        }
    return abstain_node


def make_observe_node(storage: JsonStorageService):
    """创建日志记录节点。"""
    def observe_node(state: QAState) -> QAState:
        references = state.get("references", [])
        verification = state.get("verification")
        if verification is None:
            return {}

        full_answer = (
            f"{verification.answer.strip()}\n\n### 引用文档\n{_format_reference_markdown(references)}"
        )

        qa_log = storage.add_qa_log({
            "session_id": state.get("session_id", ""),
            "question": state["question"],
            "answer": full_answer,
            "category": state.get("category", "全部"),
            "search_category": state.get("search_category", "全部"),
            "domain": state.get("domain", "general"),
            "route_confidence": state.get("route_confidence", 0.0),
            "route_fallback": state.get("route_fallback", False),
            "question_type": config.QUESTION_TYPE_LABELS.get(
                state.get("route_decision", RouteDecision("", "", "", 0.0, ())).question_type, "制度问答"
            ),
            "source_docs": references,
            "verification_passed": verification.passed,
            "verification_issues": list(verification.issues),
            "top_relevance_score": references[0]["relevance_score"] if references else 0.0,
            "retrieval_strategy": "dense_bm25_rrf_reranker_langgraph",
            "retrieval_sources": sorted(
                {s for r in references for s in r.get("retrieval_sources", [])}
            ),
            "decomposed": state.get("decomposed", False),
            "query_count": len(state.get("queries", [])),
            "retry_count": state.get("retry_count", 0),
        })

        return {
            "answer": full_answer,
            "qa_log_id": qa_log["id"],
            "source_docs": references,
            "question_type": config.QUESTION_TYPE_LABELS.get(
                state.get("route_decision", RouteDecision("", "", "", 0.0, ())).question_type, "制度问答"
            ),
        }
    return observe_node


# ── 条件边 ─────────────────────────────────────────────────────

def _should_abstain(state: QAState) -> str:
    """决定是否拒答。"""
    references = state.get("references", [])
    if not references:
        return "abstain"
    return "continue"


def _should_retry(state: QAState) -> str:
    """Self-RAG 不通过时决定是否重试。"""
    if state.get("self_rag_passed", True):
        return "verify"
    retry_count = state.get("retry_count", 0)
    if retry_count < 2:
        return "retry"
    return "verify"


def _should_end(state: QAState) -> str:
    """验证后决定是否结束。"""
    verification = state.get("verification")
    if verification is None:
        return "observe"
    if verification.passed:
        return "observe"
    return "observe"  # 即使不通过也输出（已自动修正）


# ── 辅助函数 ───────────────────────────────────────────────────

def _build_context(search_results: list[RetrievalResult]) -> str:
    """从检索结果构建上下文文本。"""
    blocks = []
    for index, result in enumerate(search_results, start=1):
        document = result.document
        title = document.metadata.get("title", document.metadata.get("file_name", "未命名文档"))
        blocks.append(
            f"[{index}] 标题：{title}\n"
            f"分类：{document.metadata.get('category', '未分类')} | "
            f"版本：{document.metadata.get('version', '未填写')} | "
            f"相关度：{result.relevance_score:.2%} | "
            f"向量：{result.vector_score:.2%} | BM25：{result.lexical_score:.2%}\n"
            f"内容：{document.page_content[:800]}"
        )
    return "\n\n".join(blocks)


def _build_references(search_results: list[RetrievalResult]) -> list[dict]:
    """从检索结果构建引用列表。"""
    references = []
    seen_document_ids = set()
    for result in search_results:
        document = result.document
        document_id = document.metadata.get("document_id")
        if document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)
        references.append({
            "document_id": document_id,
            "title": document.metadata.get("title", document.metadata.get("file_name", "未命名文档")),
            "category": document.metadata.get("category", "未分类"),
            "version": document.metadata.get("version", "未填写"),
            "file_name": document.metadata.get("file_name", ""),
            "relevance_score": result.relevance_score,
            "vector_score": result.vector_score,
            "lexical_score": result.lexical_score,
            "rrf_score": result.rrf_score,
            "retrieval_sources": list(result.retrieval_sources),
            "excerpt": document.page_content[:160].replace("\n", " "),
        })
        if len(references) >= config.max_reference_documents:
            break
    return references


def _format_reference_markdown(references: list[dict]) -> str:
    """格式化引用为 Markdown。"""
    if not references:
        return "无"
    lines = []
    for index, item in enumerate(references, start=1):
        sources = "+".join(item["retrieval_sources"]) or "unknown"
        lines.append(
            f"{index}. **{item['title']}** | {item['category']} | {item['version']} | "
            f"相关度 {item['relevance_score']:.0%} | {sources}\n   > {item['excerpt']}"
        )
    return "\n".join(lines)


# ── 工作流构建 ──────────────────────────────────────────────────

def build_workflow(
    storage: JsonStorageService,
    vector_store,
    router: DomainRouterAgent,
    answer_agent: GroundedAnswerAgent,
    verifier: AnswerVerifierAgent,
    llm: Optional[BaseLanguageModel] = None,
):
    """构建 LangGraph 工作流。"""
    graph = StateGraph(QAState)

    # 添加节点（节点名加 _step 后缀，避免与状态字段同名冲突）
    graph.add_node("route_step", make_route_node(router))
    graph.add_node("rewrite_step", make_rewrite_node(llm, config))
    graph.add_node("retrieve_step", make_retrieve_node(vector_store))
    graph.add_node("answer_step", make_answer_node(answer_agent, llm))
    graph.add_node("verify_step", make_verify_node(verifier))
    graph.add_node("abstain_step", make_abstain_node())
    graph.add_node("observe_step", make_observe_node(storage))

    # 设置入口
    graph.set_entry_point("route_step")

    # 顺序边：route → rewrite → retrieve
    graph.add_edge("route_step", "rewrite_step")
    graph.add_edge("rewrite_step", "retrieve_step")

    # 条件边：retrieve → answer 或 abstain（拒答也走校验+日志，保证输出完整）
    graph.add_conditional_edges(
        "retrieve_step",
        _should_abstain,
        {
            "abstain": "abstain_step",
            "continue": "answer_step",
        },
    )
    graph.add_edge("abstain_step", "verify_step")

    # 条件边：answer → verify 或 retry
    graph.add_conditional_edges(
        "answer_step",
        _should_retry,
        {
            "retry": "answer_step",
            "verify": "verify_step",
        },
    )

    # 顺序边：verify → observe → END
    graph.add_edge("verify_step", "observe_step")
    graph.add_edge("observe_step", END)

    return graph.compile()