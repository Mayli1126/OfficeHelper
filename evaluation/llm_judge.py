"""LLM-as-Judge 多维评估：对答案质量进行语义层面的评估。

评估维度：
1. Faithfulness（忠实度）：答案是否严格基于引用材料，有无幻觉
2. Relevance（相关度）：答案是否直接回答用户问题
3. Format（格式合规）：三段式结构、引用编号是否正确
4. Completeness（完整性）：是否遗漏关键信息
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    """单个维度的评分。"""
    score: int  # 1-5
    reason: str


@dataclass
class JudgeReport:
    """完整评估报告。"""
    faithfulness: JudgeScore
    relevance: JudgeScore
    format_compliance: JudgeScore
    completeness: JudgeScore
    overall: float  # 加权综合分 0-1
    issues: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "faithfulness": {"score": self.faithfulness.score, "reason": self.faithfulness.reason},
            "relevance": {"score": self.relevance.score, "reason": self.relevance.reason},
            "format": {"score": self.format_compliance.score, "reason": self.format_compliance.reason},
            "completeness": {"score": self.completeness.score, "reason": self.completeness.reason},
            "overall": self.overall,
            "issues": self.issues,
        }


class LLMJudge:
    """使用 LLM 对答案进行多维度语义评估。"""

    JUDGE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是企业知识问答的质量评估专家。请对以下答案进行多维度评估。

评估维度（每项 1-5 分）：
1. **忠实度 (Faithfulness)**：答案是否严格基于引用材料？有没有编造、常识补全或幻觉？
2. **相关度 (Relevance)**：答案是否直接、完整地回答了用户的问题？
3. **格式合规 (Format)**：是否包含"最终回答"、"操作步骤/材料清单"、"风险提示"三个部分？引用编号是否正确？
4. **完整性 (Completeness)**：是否遗漏了引用材料中的关键信息（金额、时限、审批人等）？

输出 JSON 格式：
{{
  "faithfulness": {{"score": 1-5, "reason": "简短理由"}},
  "relevance": {{"score": 1-5, "reason": "简短理由"}},
  "format": {{"score": 1-5, "reason": "简短理由"}},
  "completeness": {{"score": 1-5, "reason": "简短理由"}},
  "issues": ["问题1", "问题2"]
}}

只输出 JSON，不要其他内容。"""),
        ("human", """用户问题：{question}

参考材料：
{context}

待评估答案：
{answer}

请评估："""),
    ])

    # 维度权重
    WEIGHTS = {
        "faithfulness": 0.35,
        "relevance": 0.30,
        "format": 0.15,
        "completeness": 0.20,
    }

    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        self.llm = llm

    def judge(self, question: str, answer: str, context: str) -> JudgeReport:
        """对答案进行多维度评估。"""
        if self.llm is None:
            return self._empty_report("未配置 LLM")

        try:
            parser = JsonOutputParser()
            chain = self.JUDGE_PROMPT | self.llm | parser
            result = chain.invoke({
                "question": question,
                "answer": answer,
                "context": context,
            })
            return self._parse_result(result, answer)
        except Exception as exc:
            logger.warning("LLM-as-Judge 评估失败: %s", exc)
            return self._empty_report(str(exc))

    def _parse_result(self, result: dict, answer: str) -> JudgeReport:
        """解析 LLM 评估结果。"""
        try:
            faith = result.get("faithfulness", {})
            rel = result.get("relevance", {})
            fmt = result.get("format", {})
            comp = result.get("completeness", {})
            issues = result.get("issues", [])

            overall = (
                self.WEIGHTS["faithfulness"] * faith.get("score", 3)
                + self.WEIGHTS["relevance"] * rel.get("score", 3)
                + self.WEIGHTS["format"] * fmt.get("score", 3)
                + self.WEIGHTS["completeness"] * comp.get("score", 3)
            ) / 5.0

            return JudgeReport(
                faithfulness=JudgeScore(score=faith.get("score", 3), reason=faith.get("reason", "")),
                relevance=JudgeScore(score=rel.get("score", 3), reason=rel.get("reason", "")),
                format_compliance=JudgeScore(score=fmt.get("score", 3), reason=fmt.get("reason", "")),
                completeness=JudgeScore(score=comp.get("score", 3), reason=comp.get("reason", "")),
                overall=round(overall, 4),
                issues=issues if isinstance(issues, list) else [],
                raw_response=json.dumps(result, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("解析评估结果失败: %s", exc)
            return self._empty_report(str(exc))

    @staticmethod
    def _empty_report(reason: str) -> JudgeReport:
        return JudgeReport(
            faithfulness=JudgeScore(score=3, reason=f"评估不可用: {reason}"),
            relevance=JudgeScore(score=3, reason=f"评估不可用: {reason}"),
            format_compliance=JudgeScore(score=3, reason=f"评估不可用: {reason}"),
            completeness=JudgeScore(score=3, reason=f"评估不可用: {reason}"),
            overall=0.6,
            issues=[reason],
        )