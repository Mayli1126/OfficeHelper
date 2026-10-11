import tempfile
import unittest
from pathlib import Path

from services.lexical_index import BM25LexicalIndex


class BM25LexicalIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index = BM25LexicalIndex(Path(self.temp_dir.name) / "lexical.json")
        self.index.add_document(
            "finance",
            ["差旅报销需要提交发票、行程单和酒店住宿水单。"],
            [{"title": "差旅制度", "category": "财务制度"}],
        )
        self.index.add_document(
            "it",
            ["VPN 无法连接时检查网络和客户端版本，然后提交 IT 工单。"],
            [{"title": "IT FAQ", "category": "IT支持"}],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bm25_retrieves_exact_business_terms(self):
        hits = self.index.search("差旅报销发票材料")
        self.assertEqual(hits[0].document.metadata["title"], "差旅制度")
        self.assertGreater(hits[0].raw_score, 0)
        self.assertGreater(hits[0].score, 0)

    def test_category_filter_is_applied_before_scoring(self):
        hits = self.index.search("申请流程", category="IT支持")
        self.assertTrue(all(hit.document.metadata["category"] == "IT支持" for hit in hits))

    def test_category_filter_keeps_global_idf_statistics(self):
        hits = self.index.search("VPN 连接失败", category="IT支持")
        self.assertEqual(hits[0].document.metadata["title"], "IT FAQ")
        self.assertGreater(hits[0].score, 0.3)

    def test_generic_policy_words_do_not_create_false_positive(self):
        self.assertEqual(self.index.search("海外办公室宠物政策是什么"), [])

    def test_delete_removes_document_chunks(self):
        self.index.delete_document("finance")
        self.assertEqual(self.index.search("差旅报销发票"), [])


if __name__ == "__main__":
    unittest.main()
