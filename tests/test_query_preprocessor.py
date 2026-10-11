import unittest

from langchain_core.runnables import Runnable

from services.query_preprocessor import PreprocessResult, QueryPreprocessor


class FakeLLM(Runnable):
    """记录调用次数并返回预设内容的假 LLM（Runnable 以兼容 LCEL 管道）。"""

    def __init__(self, response):
        super().__init__()
        self.response = response
        self.call_count = 0

    def invoke(self, *_input, _config=None, **_kwargs):
        self.call_count += 1
        return self.response


class QueryPreprocessorTests(unittest.TestCase):
    def test_short_clean_question_skips_llm_entirely(self):
        llm = FakeLLM("不应该被调用")
        result = QueryPreprocessor(llm).preprocess("年假提前几天申请？")

        self.assertEqual(llm.call_count, 0)
        self.assertTrue(result.skipped)
        self.assertEqual(result.queries, ["年假提前几天申请？"])
        self.assertEqual(result.llm_calls, 0)

    def test_long_question_merges_rewrite_hyde_decompose_into_one_call(self):
        question = "我们部门的同事出差在外面客户那边不方便走报销系统那这种差旅的费用应该怎么去报销需要准备什么样的发票和材料呢"
        response = (
            '{"variants": ["差旅报销流程及所需材料"], '
            '"hyde": "员工差旅结束后应在10个自然日内提交报销单，附发票与行程单。", '
            '"subqueries": []}'
        )
        llm = FakeLLM(response)
        result = QueryPreprocessor(llm).preprocess(question)

        self.assertEqual(llm.call_count, 1)
        self.assertFalse(result.skipped)
        self.assertEqual(result.queries[0], question)
        self.assertIn("差旅报销流程及所需材料", result.queries)
        self.assertTrue(any("10个自然日" in q for q in result.queries))

    def test_multi_part_question_uses_subqueries(self):
        question = "年假怎么申请，病假需要什么材料，忘记打卡怎么办，加班怎么调休"
        response = (
            '{"variants": [], "hyde": "", '
            '"subqueries": ["年假申请流程", "病假所需材料", "补卡规则", "调休规则"]}'
        )
        llm = FakeLLM(response)
        result = QueryPreprocessor(llm).preprocess(question)

        self.assertEqual(llm.call_count, 1)
        self.assertTrue(result.decomposed)
        self.assertEqual(result.queries[0], question)
        self.assertIn("病假所需材料", result.queries)

    def test_garbage_llm_output_falls_back_to_original(self):
        llm = FakeLLM("这不是JSON，模型抽风了")
        result = QueryPreprocessor(llm).preprocess(
            "这是一个很长很长很长很长很长很长很长很长很长很长的复杂问题需要处理"
        )

        self.assertEqual(llm.call_count, 1)
        self.assertTrue(result.skipped)
        self.assertEqual(result.queries, ["这是一个很长很长很长很长很长很长很长很长很长很长的复杂问题需要处理"])

    def test_no_llm_skips(self):
        result = QueryPreprocessor(None).preprocess("随便什么问题")
        self.assertTrue(result.skipped)
        self.assertEqual(result.queries, ["随便什么问题"])


if __name__ == "__main__":
    unittest.main()