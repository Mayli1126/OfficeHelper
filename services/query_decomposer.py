"""查询分解：将复杂问题拆分为独立子查询，分别检索后合并结果。

策略：
1. 检测问题是否包含多个子问题（如"年假怎么请、病假需要什么材料"）
2. LLM 拆分为 2-4 个独立子查询
3. 每个子查询独立执行路由 + 检索
4. 按 document_id 去重合并，按相关度重排
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """将复杂问题分解为独立子查询。"""

    DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是查询分解专家。请判断用户问题是否包含多个独立的子问题。

规则：
- 如果问题只涉及一个主题，输出 SINGLE
- 如果问题包含 2-4 个独立子问题，每个子问题应该可以独立检索和回答
- 输出格式：
  第一行：MULTI 或 SINGLE
  如果是 MULTI，后续每行一个子问题，不要编号，不要解释
  如果是 SINGLE，结束"""),
        ("human", "用户问题：{question}"),
    ])

    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        self.llm = llm

    def decompose(self, question: str) -> list[str]:
        """分解问题，返回子查询列表。

        如果是单一问题，返回 [question]。
        如果是多问题，返回拆分后的子查询列表。
        """
        if self.llm is None:
            return [question]

        try:
            chain = self.DECOMPOSE_PROMPT | self.llm | StrOutputParser()
            result = chain.invoke({"question": question}).strip()

            lines = result.split("\n")
            if not lines:
                return [question]

            if lines[0].strip().upper() == "SINGLE":
                return [question]

            sub_queries = [line.strip() for line in lines[1:] if line.strip()]
            if not sub_queries:
                return [question]

            # 限制最多 4 个子查询
            return sub_queries[:4]
        except Exception as exc:
            logger.warning("查询分解失败，使用原始问题: %s", exc)
            return [question]

    def is_complex(self, question: str) -> bool:
        """判断问题是否复杂（需要分解）。"""
        # 启发式：包含多个疑问词或并列结构
        question_markers = ["？", "?", "，", ",", "、", "；", ";"]
        question_words = ["什么", "怎么", "如何", "哪些", "多少", "谁", "哪里", "为什么"]
        # 计算问题中疑问词数量
        word_count = sum(1 for w in question_words if w in question)
        # 计算并列标记数量
        separator_count = sum(1 for m in question_markers if m in question)
        return word_count >= 2 or separator_count >= 2