import unittest

from langchain_core.documents import Document

from services.lexical_index import LexicalHit
from services.vector_store import OfficeMateVectorStore


class VectorScoringTests(unittest.TestCase):
    def test_chinese_keyword_coverage(self):
        score = OfficeMateVectorStore._lexical_coverage(
            "差旅报销需要哪些材料",
            "员工完成差旅后，应提交报销申请、发票和行程单等材料。",
        )
        self.assertGreater(score, 0.3)

    def test_unrelated_content_has_low_coverage(self):
        score = OfficeMateVectorStore._lexical_coverage("VPN 密码重置", "采购申请需要部门负责人审批")
        self.assertEqual(score, 0.0)

    def test_rrf_promotes_candidate_found_by_both_retrievers(self):
        finance = Document(
            page_content="差旅报销材料",
            metadata={"document_id": "finance", "chunk_index": 0},
        )
        unrelated = Document(
            page_content="普通员工手册",
            metadata={"document_id": "manual", "chunk_index": 0},
        )
        vector_candidates = [(unrelated, 0.8), (finance, 0.5)]
        lexical_candidates = [LexicalHit(finance, raw_score=5.0, score=0.81)]

        results = OfficeMateVectorStore._fuse_results(vector_candidates, lexical_candidates)

        self.assertEqual(results[0].document.metadata["document_id"], "finance")
        self.assertEqual(results[0].retrieval_sources, ("vector", "bm25"))
        self.assertGreater(results[0].rrf_score, results[1].rrf_score)

    def test_vector_confidence_rejects_far_distance(self):
        self.assertEqual(OfficeMateVectorStore._vector_confidence(2.5), 0.0)


if __name__ == "__main__":
    unittest.main()
