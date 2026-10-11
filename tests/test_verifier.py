import unittest

import config_data as config
from agents.verifier import AnswerVerifierAgent


class AnswerVerifierAgentTests(unittest.TestCase):
    def setUp(self):
        self.verifier = AnswerVerifierAgent()
        self.references = [{"title": "差旅制度"}]

    def test_missing_sections_and_citations_are_reported(self):
        report = self.verifier.verify("### 最终回答\n可以报销。", self.references)
        self.assertFalse(report.passed)
        self.assertIn("missing_inline_citation", report.issues)
        self.assertIn("### 风险提示", report.answer)

    def test_no_evidence_forces_abstention(self):
        report = self.verifier.verify("这是一个没有来源的肯定回答。", [])
        self.assertTrue(report.passed)
        self.assertEqual(report.answer, config.NO_EVIDENCE_MESSAGE)

    def test_valid_grounded_answer_passes(self):
        answer = (
            "### 最终回答\n可以报销。[1]\n\n"
            "### 操作步骤/材料清单\n提交发票。[1]\n\n"
            "### 风险提示\n以最新制度为准。[1]"
        )
        self.assertTrue(self.verifier.verify(answer, self.references).passed)


if __name__ == "__main__":
    unittest.main()
