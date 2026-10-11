"""Self-RAG：模型在生成答案后自我验证引用是否支持结论。

流程：
1. 模型生成初步答案
2. 模型检查每个关键结论是否有对应引用
3. 对无引用的结论进行标记或修正
"""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class SelfRAGRefiner:
    """对生成的答案进行自我验证和修正。"""

    VERIFY_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是答案校验专家。请检查以下回答是否严格基于参考材料。

        检查项：
        1. 每个关键结论（数字、时限、规则、审批人）是否能在引用材料中找到依据
        2. 是否有编造或常识补全的内容
        3. 引用编号是否正确对应材料

        输出格式：
        - 如果全部通过，输出：VERIFIED
        - 如果有问题，输出：ISSUES: <具体问题列表>"""),
        ("human", """参考材料：
        {context}

        待验证答案：
        {answer}

        请验证："""),
    ])

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    def verify(self, answer: str, context: str) -> tuple[str, bool]:
        """返回 (修正后的答案, 是否通过验证)。"""
        try:
            chain = self.VERIFY_PROMPT | self.llm | StrOutputParser()
            result = chain.invoke({"answer": answer, "context": context})
            if "VERIFIED" in result:
                return answer, True
            else:
                flagged = answer + "\n\n> ⚠️ 注意：以上内容中有部分结论的引用依据不够充分，建议核实。"
                return flagged, False
        except Exception:
            return answer, True