"""中文文本分析：jieba 分词 + 完整停用词表 + n-gram 保留。

策略：
- 优先使用 jieba 进行中文分词
- 同时保留字符级 n-gram 以捕获复合术语
- 过滤停用词，保留业务实体
"""

import re
from functools import lru_cache

try:
    import jieba
    jieba.initialize()
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

# 中文停用词表
STOP_TOKENS = {
    # 原有
    "公司", "司的", "办公", "公室", "政策", "是什", "什么", "怎么", "如何",
    "哪些", "需要", "应该", "可以", "是否", "这个", "那个", "相关",
    # 新增：制度文档常见噪声词
    "的", "了", "在", "和", "或", "等", "及", "与", "就", "都", "也",
    "将", "被", "让", "使", "对", "从", "到", "关于", "通过", "由于",
    "根据", "按照", "依据", "即", "如", "若", "员", "部", "门",
    "负", "责", "审", "批", "流", "程", "系", "统", "内", "完",
    "成", "提", "交", "申", "请", "进", "行", "操", "作", "人",
    "事", "物", "时", "间", "地", "点", "方", "式", "方", "法",
    "标", "准", "要", "求", "必", "须", "应", "当", "予", "以",
    "予", "以", "不", "予", "不", "得", "未", "经", "通", "过",
    "前", "后", "中", "外", "上", "下", "左", "右", "前", "后",
}


@lru_cache(maxsize=5000)
def tokenize(text: str) -> list[str]:
    """分词：jieba 分词 + 字符级 token，过滤停用词。

    策略：
    - jieba 分词作为主分词（保留 >= 2 字符的词语）
    - 仅保留单字中的数字和 ASCII 字母（中文单字过于常见，会产生大量噪声）
    - 过滤停用词
    """
    normalized = text.lower()
    tokens: list[str] = []

    if JIEBA_AVAILABLE:
        words = jieba.lcut(normalized)
        for word in words:
            word = word.strip()
            if word and word not in STOP_TOKENS and len(word) >= 2:
                tokens.append(word)
        # 仅保留数字和 ASCII 字母作为单字 token
        for char in normalized:
            if char.isdigit() or (char.isascii() and char.isalpha()):
                if char not in STOP_TOKENS:
                    tokens.append(char)
    else:
        # 回退方案：原有 2 字符滑动窗口
        tokens = re.findall(r"[a-z0-9]+", normalized)
        for group in re.findall(r"[\u4e00-\u9fff]+", normalized):
            if len(group) == 1:
                tokens.append(group)
            else:
                tokens.extend(group[i:i + 2] for i in range(len(group) - 1))
        tokens = [t for t in tokens if t not in STOP_TOKENS]

    return tokens


def tokenize_with_ngrams(text: str, n: int = 2) -> list[str]:
    """分词 + n-gram 增强，提升复合术语的匹配能力。"""
    words = tokenize(text)
    ngrams = list(words)
    for i in range(len(words) - n + 1):
        ngram = "".join(words[i:i + n])
        if len(ngram) >= 3:
            ngrams.append(ngram)
    return ngrams