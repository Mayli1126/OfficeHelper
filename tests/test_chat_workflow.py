import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from services.chat_service import OfficeMateChatService
from services.storage_service import JsonStorageService
from services.vector_store import RetrievalResult


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.last_category = None

    def search(self, query, category="全部"):
        self.last_category = category
        return self.results


class FallbackVectorStore(FakeVectorStore):
    def __init__(self, fallback_results):
        super().__init__(fallback_results)
        self.categories = []

    def search(self, query, category="全部"):
        self.categories.append(category)
        return self.results if category == "全部" else []


class FakeAnswerAgent:
    def __init__(self):
        self.called = False

    def answer(self, model, **kwargs):
        self.called = True
        return (
            "### 最终回答\n需要提交发票和行程单。[1]\n\n"
            "### 操作步骤/材料清单\n整理材料后发起报销。[1]\n\n"
            "### 风险提示\n逾期需补充说明。[1]"
        )


class ChatWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.storage = JsonStorageService(root / "docs.json", root / "qa.json", root / "feedback.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_workflow_routes_retrieves_verifies_and_logs(self):
        document = Document(
            page_content="差旅报销应提交发票、行程单，并在返程后发起申请。",
            metadata={
                "document_id": "doc-1",
                "title": "差旅与报销制度",
                "category": "财务制度",
                "version": "v1",
                "file_name": "finance.txt",
            },
        )
        result = RetrievalResult(document, 0.2, 0.83, 0.8, 0.82)
        vector_store = FakeVectorStore([result])
        answer_agent = FakeAnswerAgent()
        service = OfficeMateChatService(
            storage=self.storage,
            vector_store=vector_store,
            chat_model=object(),
            answer_agent=answer_agent,
        )

        response = service.answer_question("差旅报销需要哪些材料？", "session-1")

        self.assertEqual(vector_store.last_category, "财务制度")
        self.assertTrue(answer_agent.called)
        self.assertTrue(response["verification_passed"])
        self.assertIn("相关度 82%", response["answer"])
        log = self.storage.list_qa_logs()[0]
        self.assertEqual(log["domain"], "finance")
        self.assertEqual(log["top_relevance_score"], 0.82)

    def test_empty_retrieval_abstains_without_calling_model(self):
        answer_agent = FakeAnswerAgent()
        service = OfficeMateChatService(
            storage=self.storage,
            vector_store=FakeVectorStore([]),
            answer_agent=answer_agent,
        )
        response = service.answer_question("火星出差如何报销？", "session-2")
        self.assertFalse(answer_agent.called)
        self.assertIn("未找到明确依据", response["answer"])

    def test_empty_routed_category_falls_back_to_global_search(self):
        document = Document(
            page_content="通用差旅说明",
            metadata={"document_id": "doc-2", "chunk_index": 0, "title": "综合制度"},
        )
        vector_store = FallbackVectorStore(
            [RetrievalResult(document, 0.4, 0.8, 0.0, 0.7, retrieval_sources=("vector",))]
        )
        service = OfficeMateChatService(
            storage=self.storage,
            vector_store=vector_store,
            chat_model=object(),
            answer_agent=FakeAnswerAgent(),
        )

        response = service.answer_question("差旅报销怎么办？", "session-3")

        self.assertIn("财务制度", vector_store.categories)
        self.assertIn("全部", vector_store.categories)
        self.assertTrue(response["route_fallback"])
        self.assertEqual(response["search_category"], "全部")


if __name__ == "__main__":
    unittest.main()
