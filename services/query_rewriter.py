"""查询改写与多查询扩展，提升口语化查询的召回效果。

策略：
1. 口语 → 书面语规范化
2. 领域术语补全
3. 生成多角度查询变体（用于多路召回）
4. HyDE（Hypothetical Document Embeddings）：生成假设文档片段用于向量检索
"""

from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class QueryRewriter:
    """利用 LLM 将用户问题改写为更适合检索的规范查询。"""

    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是企业知识库的查询优化专家。请将用户的口语化问题改写为
        更适合文档检索的规范查询。

        规则：
        - 保留所有关键业务实体（制度名、金额、时限、审批角色等）
        - 将口语表达转为书面语（如"VPN 连不上" → "VPN 无法连接"）
        - 如果问题涉及多个子问题，拆分为 2-3 个独立查询
        - 只输出改写后的查询，每行一个，不要解释"""),
        ("human", "用户问题：{question}"),
    ])

    HYDE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是企业制度文档的写作助手。请根据用户的问题，
        假设知识库中存在一份完美的答案文档，写出这段答案文档的简短片段（50-100字）。
        只输出片段内容，不要解释。"""),
        ("human", "问题：{question}"),
    ])

    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        self.llm = llm

    def rewrite(self, question: str) -> list[str]:
        """返回改写后的查询列表（原始查询 + 改写变体）。"""
        if self.llm is None:
            return [question]

        try:
            chain = self.REWRITE_PROMPT | self.llm | StrOutputParser()
            result = chain.invoke({"question": question})
            rewritten = [q.strip() for q in result.content.strip().split("\n") if q.strip()]
            all_queries = [question] + [q for q in rewritten if q != question]
            return all_queries[:3]
        except Exception:
            return [question]

    def hyde_embed(self, question: str) -> str:
        """生成假设文档片段，用于向量检索的查询扩展。"""
        if self.llm is None:
            return question
        try:
            chain = self.HYDE_PROMPT | self.llm | StrOutputParser()
            result = chain.invoke({"question": question})
            return result.content.strip()
        except Exception:
            return question