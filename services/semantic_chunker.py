"""基于标题层级和语义边界的文档切分器，支持动态切分策略。

策略：
1. 先按标题层级（章/节/条目）切分为语落
2. 根据文档类型自适应调整 chunk_size
3. 段落内按段落边界（双换行）二次切分
4. 超长段落按句子边界切分
5. 合并过小碎片到相邻块
"""

import re
from typing import Optional

from langchain_core.documents import Document


class SemanticChunker:
    """感知标题层级的语义切分器，支持动态切分策略。"""

    HEADING_PATTERNS = [
        (re.compile(r"^第[一二三四五六七八九十\d]+章\s*(.+)"), 1),
        (re.compile(r"^([一二三四五六七八九十]+)、\s*(.+)"), 2),
        (re.compile(r"^\d+\.\s*(.+)"), 2),
        (re.compile(r"^\(\d+\)\s*(.+)"), 3),
    ]

    # 按文档类型预设切分策略
    STRATEGIES = {
        "fine": {"max_chunk_size": 600, "min_chunk_size": 120},   # 政策/制度：条款独立
        "default": {"max_chunk_size": 800, "min_chunk_size": 150},  # 通用
        "coarse": {"max_chunk_size": 1200, "min_chunk_size": 200},  # 流程/指南：步骤连贯
    }

    # 文档分类 → 策略映射
    CATEGORY_STRATEGY = {
        "HR制度": "fine",
        "财务制度": "fine",
        "IT支持": "coarse",
        "行政流程": "coarse",
        "员工手册": "default",
        "综合公告": "default",
    }

    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 150, strategy: str = "default"):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.strategy = strategy

    @classmethod
    def for_category(cls, category: str) -> "SemanticChunker":
        """根据文档分类创建切分器，自动选择策略。"""
        strategy = cls.CATEGORY_STRATEGY.get(category, "default")
        params = cls.STRATEGIES[strategy]
        return cls(
            max_chunk_size=params["max_chunk_size"],
            min_chunk_size=params["min_chunk_size"],
            strategy=strategy,
        )

    def split_text(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        """将文本切分为语义完整的 Document 列表。

        如果 metadata 中包含 category，且当前策略为 default，
        会自动切换为对应分类的策略。
        """
        # 根据 metadata 中的 category 动态切换策略
        category = (metadata or {}).get("category", "")
        if category and self.strategy == "default":
            effective = self.for_category(category)
            effective_max = effective.max_chunk_size
            effective_min = effective.min_chunk_size
        else:
            effective_max = self.max_chunk_size
            effective_min = self.min_chunk_size

        sections = self._split_by_headings(text)
        chunks: list[Document] = []

        for section_title, heading_level, section_body in sections:
            if len(section_body) <= effective_max:
                chunks.append(self._to_document(section_title, section_body, len(chunks), metadata))
            else:
                sub_chunks = self._split_by_paragraphs(
                    section_title, heading_level, section_body, effective_max
                )
                for sub_content in sub_chunks:
                    chunks.append(self._to_document(section_title, sub_content, len(chunks), metadata))

        chunks = self._merge_undersized_chunks(chunks, effective_max, effective_min)
        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str, int, str]]:
        """按标题层级切分为 (标题, 层级, 正文) 元组列表。"""
        lines = text.split("\n")
        sections: list[tuple[str, int, str]] = []
        current_title = "文档"
        current_level = 0
        current_lines: list[str] = []

        for line in lines:
            matched_level, matched_title = self._match_heading(line)
            if matched_level is not None:
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append((current_title, current_level, body))
                current_title = matched_title
                current_level = matched_level
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, current_level, body))

        return sections if sections else [("文档", 0, text)]

    def _match_heading(self, line: str) -> tuple[Optional[int], Optional[str]]:
        line = line.strip()
        if not line:
            return None, None
        for pattern, level in self.HEADING_PATTERNS:
            match = pattern.match(line)
            if match:
                return level, match.group(0)
        return None, None

    def _split_by_paragraphs(self, title: str, level: int, body: str, max_size: int) -> list[str]:
        """在段落边界（双换行）处切分，控制每块大小。"""
        paragraphs = re.split(r"\n\s*\n", body)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) <= max_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                if len(para) > max_size:
                    chunks.extend(self._split_long_paragraph(para, max_size))
                else:
                    current = para

        if current:
            chunks.append(current)
        return chunks

    def _split_long_paragraph(self, para: str, max_size: int) -> list[str]:
        """按句号/问号/感叹号切分超长段落。"""
        sentences = re.split(r"(?<=[。！？.!?])", para)
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= max_size:
                current += sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        return chunks

    def _merge_undersized_chunks(
        self, chunks: list[Document], max_size: int, min_size: int
    ) -> list[Document]:
        """合并过小的碎片到前一个块，保持语义完整。"""
        if len(chunks) <= 1:
            return chunks
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk.page_content) < min_size:
                last = merged[-1]
                combined = (last.page_content + "\n\n" + chunk.page_content).strip()
                if len(combined) <= max_size:
                    merged[-1] = Document(
                        page_content=combined,
                        metadata={**last.metadata, "chunk_index": last.metadata["chunk_index"]},
                    )
                    continue
            merged.append(chunk)
        for idx, chunk in enumerate(merged):
            chunk.metadata["chunk_index"] = idx
        return merged

    def _to_document(self, title: str, content: str, index: int, metadata: Optional[dict]) -> Document:
        return Document(
            page_content=content,
            metadata={
                **(metadata or {}),
                "section_title": title,
                "chunk_index": index,
            },
        )