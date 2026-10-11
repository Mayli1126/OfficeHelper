"""查询预处理器：把改写、HyDE、分解合并为一次 LLM 调用，并对规整短问题零调用跳过。

优化目标：主管线原本每次提问串行消耗 2~3 次前置 LLM 调用（改写、HyDE、分解），
现在压缩为 0 次（短问题跳过）或 1 次（合并 JSON 输出）。
"""

import json
import logging
from dataclasses import dataclass, field

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config_data as config
from services.query_decomposer import QueryDecomposer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessResult:
    """查询预处理结果。"""

    queries: list[str] = field(default_factory=list)
    skipped: bool = False
    decomposed: bool = False
    llm_calls: int = 0


class QueryPreprocessor:
    """统一查询预处理入口。

    策略：
    1. 短且不复杂的问题（长度 <= 阈值 且启发式判定单一主题）→ 直接跳过，零 LLM 调用
    2. 其余问题 → 一次 LLM 调用同时产出改写变体、HyDE 片段、子查询
    3. 任何异常 → 回退原始问题，绝不阻塞检索主流程
    """

    PREPROCESS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是企业知识库检索的查询预处理专家。对用户的问题一次性完成三项加工：

1. variants（改写变体）：将口语表达转为书面语，保留金额、时限、审批角色等业务实体，
   生成最多 2 个适合文档检索的查询变体。
2. hyde（假设文档）：假设知识库中存在一份完美的制度文档，写出 50~100 字的答案片段，
   用于辅助向量检索。
3. subqueries（子查询）：仅当问题包含多个互相独立的子问题时拆分（最多 4 个），
   每个子问题应可独立检索回答；单一主题则输出空数组。

只输出如下 JSON，不要任何解释或代码块标记：
{{"variants": ["变体1", "变体2"], "hyde": "假设文档片段", "subqueries": []}}"""),
        ("human", "用户问题：{question}"),
    ])

    def __init__(self, llm: BaseLanguageModel | None = None):
        self.llm = llm
        self._complexity_checker = QueryDecomposer()

    def preprocess(self, question: str) -> PreprocessResult:
        """返回最终用于检索的查询列表，可能跳过加工或仅用一次 LLM 调用。"""
        # 场景 1：无法或无需加工 → 零调用
        if self._should_skip(question):
            return PreprocessResult(queries=[question], skipped=True, llm_calls=0)

        # 场景 2：一次 LLM 调用完成改写 + HyDE + 分解
        try:
            chain = self.PREPROCESS_PROMPT | self.llm | StrOutputParser()
            raw = chain.invoke({"question": question})
            data = self._parse_json(raw)
        except Exception as exc:
            logger.warning("查询预处理失败，回退原始问题: %s", exc)
            return PreprocessResult(queries=[question], skipped=True, llm_calls=1)

        variants = self._clean_list(data.get("variants"))
        hyde = str(data.get("hyde") or "").strip()
        subqueries = self._clean_list(data.get("subqueries"))

        # 复杂问题优先使用子查询（与原分解逻辑一致）
        if len(subqueries) > 1:
            queries = subqueries[: config.max_sub_queries]
            if question not in queries:
                queries = [question] + queries
            return PreprocessResult(
                queries=queries[: config.max_sub_queries + 1],
                decomposed=True,
                llm_calls=1,
            )

        # 常规：原始问题 + 改写变体 + HyDE 片段（均作为检索查询）
        queries = [question]
        for item in variants[:2]:
            if item and item != question and item not in queries:
                queries.append(item)
        if hyde and hyde != question:
            queries.append(hyde)
        return PreprocessResult(
            queries=queries[: config.max_query_variants + 1],
            llm_calls=1,
        )

    def _should_skip(self, question: str) -> bool:
        """短且不复杂的问题跳过加工；功能关闭或无模型时也跳过。"""
        if self.llm is None:
            return True
        if not (config.enable_query_rewrite or config.enable_query_decompose):
            return True
        if len(question) > config.preprocess_skip_length:
            return False
        return not self._complexity_checker.is_complex(question)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """容错解析模型输出：剥掉代码块标记，提取首个 JSON 对象。"""
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"模型输出中没有 JSON: {raw[:80]}")
        data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("模型输出不是 JSON 对象")
        return data

    @staticmethod
    def _clean_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]