import logging

from langchain_core.messages import AIMessage, HumanMessage

import config_data as config
from agents.answer_agent import GroundedAnswerAgent
from agents.router import DomainRouterAgent
from agents.self_rag import SelfRAGRefiner
from agents.verifier import AnswerVerifierAgent
from services.model_factory import create_chat_model
from services.query_preprocessor import QueryPreprocessor
from services.storage_service import JsonStorageService
from services.vector_store import OfficeMateVectorStore

logger = logging.getLogger(__name__)


class OfficeMateChatService:
    """Coordinator for route -> retrieve -> answer -> verify -> observe.

    Supports two execution modes:
    - Legacy: direct pipeline in answer_question()
    - LangGraph: state-graph based workflow (when config.enable_langgraph is True)
    """

    def __init__(
        self, storage=None, vector_store=None, chat_model=None, router=None, answer_agent=None, verifier=None,
        model_name=None,
    ):
        self.storage = storage or JsonStorageService()
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.model_name = model_name
        self.router = router or DomainRouterAgent()
        self.answer_agent = answer_agent or GroundedAnswerAgent()
        self.verifier = verifier or AnswerVerifierAgent()
        self._workflow = None

    def _get_workflow(self):
        """懒加载 LangGraph 工作流。"""
        if self._workflow is not None:
            return self._workflow
        try:
            from agents.workflow import build_workflow
            self._workflow = build_workflow(
                storage=self.storage,
                vector_store=self._get_vector_store(),
                router=self.router,
                answer_agent=self.answer_agent,
                verifier=self.verifier,
                llm=self._get_chat_model(),
            )
        except Exception as exc:
            logger.warning("LangGraph 工作流构建失败，回退到传统管线: %s", exc)
            self._workflow = None
        return self._workflow

    def answer_question(self, question, session_id, category="全部"):
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空。")

        # 尝试使用 LangGraph 工作流
        if config.enable_langgraph:
            workflow = self._get_workflow()
            if workflow is not None:
                try:
                    return self._answer_via_langgraph(
                        workflow, normalized_question, session_id, category
                    )
                except Exception as exc:
                    logger.warning("LangGraph 工作流执行失败，回退到传统管线: %s", exc)

        # 传统管线
        return self._answer_legacy(normalized_question, session_id, category)

    def _answer_via_langgraph(self, workflow, question, session_id, category):
        """通过 LangGraph 工作流执行问答。"""
        from agents.workflow import QAState

        initial_state = QAState(
            question=question,
            session_id=session_id,
            category=category,
            retry_count=0,
        )

        result = workflow.invoke(initial_state)

        return {
            "answer": result.get("answer", ""),
            "question_type": result.get("question_type", "制度问答"),
            "qa_log_id": result.get("qa_log_id", ""),
            "source_docs": result.get("source_docs", []),
            "domain": result.get("domain", "general"),
            "search_category": result.get("search_category", category),
            "route_confidence": result.get("route_confidence", 0.0),
            "route_fallback": result.get("route_fallback", False),
            "verification_passed": result.get("verification", None) is not None,
        }

    def _answer_legacy(self, question, session_id, category="全部"):
        """传统管线（与原始逻辑一致）。"""
        route = self.router.route(question, selected_category=category)
        search_category = route.category if route.confidence >= 0.7 else category

        # 查询预处理：合并改写/HyDE/分解为一次调用，短问题直接跳过
        preprocessed = self._preprocess_query(question)
        all_queries = preprocessed.queries

        # 多路检索
        search_results = self._multi_query_search(all_queries, search_category)

        # 路由回退
        route_fallback = False
        if not search_results and category == "全部" and search_category != "全部":
            search_results = self._multi_query_search(all_queries, "全部")
            search_category = "全部"
            route_fallback = True

        references = self._build_references(search_results)
        question_type = config.QUESTION_TYPE_LABELS[route.question_type]

        if references:
            chat_model = self._get_chat_model()
            draft = self.answer_agent.answer(
                chat_model,
                question=question,
                route=route,
                context=self._build_context(search_results),
                history=self._build_history(session_id),
            )
            self_rag = SelfRAGRefiner(chat_model)
            draft, _ = self_rag.verify(draft, self._build_context(search_results))
        else:
            draft = config.NO_EVIDENCE_MESSAGE

        verification = self.verifier.verify(draft, references)
        full_answer = (
            f"{verification.answer.strip()}\n\n### 引用文档\n{self._format_reference_markdown(references)}"
        )
        qa_log = self.storage.add_qa_log(
            {
                "session_id": session_id,
                "question": question,
                "answer": full_answer,
                "category": category,
                "search_category": search_category,
                "domain": route.domain,
                "route_confidence": route.confidence,
                "route_fallback": route_fallback,
                "question_type": question_type,
                "source_docs": references,
                "verification_passed": verification.passed,
                "verification_issues": list(verification.issues),
                "top_relevance_score": references[0]["relevance_score"] if references else 0.0,
                "retrieval_strategy": "dense_bm25_rrf_reranker",
                "retrieval_sources": sorted(
                    {source for reference in references for source in reference["retrieval_sources"]}
                ),
                "query_variants": len(all_queries),
                "query_preprocess_skipped": preprocessed.skipped,
                "query_decomposed": preprocessed.decomposed,
                "preprocess_llm_calls": preprocessed.llm_calls,
            }
        )
        if verification.issues:
            logger.info("Answer verification issues: %s", verification.issues)

        return {
            "answer": full_answer,
            "question_type": question_type,
            "qa_log_id": qa_log["id"],
            "source_docs": references,
            "domain": route.domain,
            "search_category": search_category,
            "route_confidence": route.confidence,
            "route_fallback": route_fallback,
            "verification_passed": verification.passed,
        }

    def _preprocess_query(self, question):
        """查询预处理：合并改写/HyDE/分解为一次 LLM 调用，短问题直接跳过。"""
        llm = self._get_chat_model() if self.chat_model is not None else None
        return QueryPreprocessor(llm).preprocess(question)

    def _multi_query_search(self, queries: list[str], category: str) -> list:
        """对多个查询变体执行检索，合并去重并按相关度排序。"""
        all_results = []
        seen_keys = set()
        # 去重查询，避免相同查询重复检索
        unique_queries = list(dict.fromkeys(queries))
        for query in unique_queries:
            results = self._get_vector_store().search(query, category=category)
            for result in results:
                key = self._result_key(result)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(result)
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return all_results

    @staticmethod
    def _result_key(result) -> str:
        """生成检索结果的去重键。"""
        doc_id = result.document.metadata.get("document_id", "")
        chunk_idx = result.document.metadata.get("chunk_index", "")
        return f"{doc_id}-{chunk_idx}"

    def infer_question_type(self, question):
        return self.router.route(question).question_type

    def _build_history(self, session_id):
        logs = self.storage.list_session_logs(session_id, limit=config.max_history_rounds)
        messages = []
        for log in logs:
            messages.append(HumanMessage(content=log["question"]))
            messages.append(AIMessage(content=log["answer"]))
        return messages

    @staticmethod
    def _build_context(search_results):
        blocks = []
        for index, result in enumerate(search_results, start=1):
            document = result.document
            title = document.metadata.get(
                "title", document.metadata.get("file_name", "未命名文档")
            )
            blocks.append(
                f"[{index}] 标题：{title}\n"
                f"分类：{document.metadata.get('category', '未分类')} | "
                f"版本：{document.metadata.get('version', '未填写')} | "
                f"相关度：{result.relevance_score:.2%} | "
                f"向量：{result.vector_score:.2%} | BM25：{result.lexical_score:.2%}\n"
                f"内容：{document.page_content[:800]}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_references(search_results):
        references = []
        seen_document_ids = set()
        for result in search_results:
            document = result.document
            document_id = document.metadata.get("document_id")
            if document_id in seen_document_ids:
                continue
            seen_document_ids.add(document_id)
            references.append(
                {
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
                }
            )
            if len(references) >= config.max_reference_documents:
                break
        return references

    @staticmethod
    def _format_reference_markdown(references):
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

    def _get_vector_store(self):
        if self.vector_store is None:
            self.vector_store = OfficeMateVectorStore()
        return self.vector_store

    def _get_chat_model(self):
        if self.chat_model is None:
            self.chat_model = create_chat_model(model_name=self.model_name)
        return self.chat_model
