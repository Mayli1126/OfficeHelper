"""Application configuration loaded from environment variables.

This module intentionally contains no credentials. Runtime secrets must be
provided by the environment so the repository can be published safely.
"""

import os
from pathlib import Path

# 加载项目根目录下的 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# 修复 conda 环境可能设置的无效 SSL_CERT_FILE（指向不存在的文件会让 httpx 崩溃）
import ssl as _ssl
_ssl_cert = os.getenv("SSL_CERT_FILE", "")
if _ssl_cert and not Path(_ssl_cert).exists():
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        os.environ.pop("SSL_CERT_FILE", None)

# HuggingFace 国内直连经常 SSL 中断，默认走 hf-mirror 镜像下载模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent
APP_NAME = "OfficeMate"

STORAGE_DIR = BASE_DIR / "storage"
RAW_DOCUMENT_DIR = STORAGE_DIR / "raw_documents"
JSON_STORE_DIR = STORAGE_DIR / "json_store"
SAMPLE_DOC_DIR = BASE_DIR / "sample_docs"

DOCUMENT_INDEX_PATH = JSON_STORE_DIR / "documents.json"
QA_LOG_PATH = JSON_STORE_DIR / "qa_logs.json"
FEEDBACK_PATH = JSON_STORE_DIR / "feedback_logs.json"
LEXICAL_INDEX_PATH = JSON_STORE_DIR / "lexical_index.json"

DOCUMENT_CATEGORIES = [
    "员工手册",
    "HR制度",
    "财务制度",
    "IT支持",
    "行政流程",
    "综合公告",
]
CATEGORY_FILTER_OPTIONS = ["全部"] + DOCUMENT_CATEGORIES
SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx", "xlsx", "csv"]
DEFAULT_VERSION = "v1.0"

collection_name = os.getenv("OFFICEMATE_COLLECTION", "officemate_knowledge_base")
persist_directory = str(STORAGE_DIR / "chroma_db")
chunk_size = int(os.getenv("OFFICEMATE_CHUNK_SIZE", "800"))
chunk_overlap = int(os.getenv("OFFICEMATE_CHUNK_OVERLAP", "120"))
separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " ", ""]
max_split_char_number = chunk_size
chunk_min_size = int(os.getenv("OFFICEMATE_CHUNK_MIN_SIZE", "150"))

# Retrieval first fetches a wider candidate set, then performs local hybrid
# reranking. The minimum score is a real acceptance threshold in [0, 1].
retrieval_candidate_count = int(os.getenv("OFFICEMATE_RETRIEVAL_CANDIDATES", "12"))
retrieval_result_count = int(os.getenv("OFFICEMATE_RETRIEVAL_RESULTS", "4"))
min_relevance_score = float(os.getenv("OFFICEMATE_MIN_RELEVANCE", "0.30"))
vector_score_weight = float(os.getenv("OFFICEMATE_VECTOR_WEIGHT", "0.65"))
embedding2_score_weight = float(os.getenv("OFFICEMATE_EMBEDDING2_WEIGHT", "0.20"))
lexical_score_weight = 1.0 - vector_score_weight - embedding2_score_weight
rrf_rank_constant = int(os.getenv("OFFICEMATE_RRF_K", "60"))
bm25_k1 = float(os.getenv("OFFICEMATE_BM25_K1", "1.5"))
bm25_b = float(os.getenv("OFFICEMATE_BM25_B", "0.75"))
bm25_saturation = float(os.getenv("OFFICEMATE_BM25_SATURATION", "3.0"))
vector_distance_scale = float(os.getenv("OFFICEMATE_VECTOR_DISTANCE_SCALE", "2.0"))

# Cross-Encoder reranker settings
enable_reranker = os.getenv("OFFICEMATE_ENABLE_RERANKER", "true").lower() in ("true", "1", "yes")
reranker_model = os.getenv("OFFICEMATE_RERANKER_MODEL", "BAAI/bge-reranker-v2-zh")
reranker_top_k = int(os.getenv("OFFICEMATE_RERANKER_TOP_K", "8"))

# Query rewriting settings
enable_query_rewrite = os.getenv("OFFICEMATE_ENABLE_QUERY_REWRITE", "true").lower() in ("true", "1", "yes")
max_query_variants = int(os.getenv("OFFICEMATE_MAX_QUERY_VARIANTS", "3"))

# Query decomposition settings
enable_query_decompose = os.getenv("OFFICEMATE_ENABLE_QUERY_DECOMPOSE", "true").lower() in ("true", "1", "yes")
max_sub_queries = int(os.getenv("OFFICEMATE_MAX_SUB_QUERIES", "4"))

# Merged query preprocessing: questions shorter than this (and not complex) skip
# rewrite/HyDE/decompose entirely, saving all preprocessing LLM calls.
preprocess_skip_length = int(os.getenv("OFFICEMATE_PREPROCESS_SKIP_LENGTH", "25"))

# Multi-embedding fusion settings
embedding2_model_name = os.getenv("OFFICEMATE_EMBEDDING2_MODEL", "")
embedding2_score_weight = float(os.getenv("OFFICEMATE_EMBEDDING2_WEIGHT", "0.20"))

# Semantic routing settings
enable_semantic_route = os.getenv("OFFICEMATE_ENABLE_SEMANTIC_ROUTE", "true").lower() in ("true", "1", "yes")
semantic_route_threshold = float(os.getenv("OFFICEMATE_SEMANTIC_ROUTE_THRESHOLD", "0.5"))

# LangGraph workflow settings
enable_langgraph = os.getenv("OFFICEMATE_ENABLE_LANGGRAPH", "true").lower() in ("true", "1", "yes")
max_workflow_retries = int(os.getenv("OFFICEMATE_MAX_WORKFLOW_RETRIES", "2"))

embedding_model_name = os.getenv("OFFICEMATE_EMBEDDING_MODEL", "text-embedding-v4")
chat_model_name = os.getenv("OFFICEMATE_CHAT_MODEL", "qwen-plus")

# ── 多厂商支持（全部走 OpenAI 兼容协议，不依赖任何厂商私有 SDK）──
# 选择厂商，可选：dashscope / openai / deepseek / zhipu / ollama / siliconflow
llm_provider = os.getenv("OFFICEMATE_LLM_PROVIDER", "dashscope").lower()

# 各厂商的候选对话模型（页面下拉可选，也支持手动输入任意模型名）
AVAILABLE_CHAT_MODELS = {
    "dashscope": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-flash", "qwen3-max", "deepseek-v4-pro"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "ollama": ["qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b"],
    "zhipu": ["glm-4-flash", "glm-4-air", "glm-4-plus"],
    "siliconflow": ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-7B-Instruct"],
}

# 各厂商 API Key 环境变量名映射
_API_KEY_ENV_VARS = {
    "dashscope": "DASHSCOPE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "ollama": None,  # 本地，不需要 Key
}


def get_api_key(provider: str = None, required: bool = False):
    """获取指定厂商的 API Key。

    支持的厂商：dashscope / openai / zhipu / deepseek / siliconflow / ollama
    ollama 是本地模型，不需要 Key，返回 None。
    """
    provider = (provider or llm_provider).lower()
    env_var = _API_KEY_ENV_VARS.get(provider)
    if env_var is None:
        return None  # ollama 不需要 Key
    api_key = os.getenv(env_var, "").strip()
    if required and not api_key:
        raise RuntimeError(f"未配置 {env_var}，请通过环境变量提供 {provider} 的密钥。")
    return api_key or None


# ── Base URL 配置（全部为 OpenAI 兼容端点）──────────────────
# 支持全局统一配置和按厂商分别配置
# 优先级：厂商专属 URL > 全局 URL > 默认 URL
base_url = os.getenv("OFFICEMATE_BASE_URL", "").strip() or None

BASE_URL_DEFAULTS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "deepseek": "https://api.deepseek.com",
    "siliconflow": "https://api.siliconflow.cn/v1",
}


def get_base_url(provider: str = None) -> str | None:
    """获取指定厂商的 base_url。

    解析优先级：
    1. OFFICEMATE_{PROVIDER}_BASE_URL（厂商专属，如 OFFICEMATE_OPENAI_BASE_URL）
    2. OFFICEMATE_BASE_URL（全局统一）
    3. BASE_URL_DEFAULTS 中的默认值
    """
    provider = (provider or llm_provider).lower()
    # 厂商专属配置
    env_name = f"OFFICEMATE_{provider.upper()}_BASE_URL"
    custom = os.getenv(env_name, "").strip()
    if custom:
        return custom
    # 全局配置
    if base_url:
        return base_url
    # 默认值
    return BASE_URL_DEFAULTS.get(provider)
max_history_rounds = int(os.getenv("OFFICEMATE_MAX_HISTORY_ROUNDS", "4"))
max_reference_documents = int(os.getenv("OFFICEMATE_MAX_REFERENCES", "4"))
default_session_prefix = "officemate"

NO_EVIDENCE_MESSAGE = (
    "### 最终回答\n"
    "未找到明确依据，当前知识库中没有足够材料支撑这个问题。\n\n"
    "### 操作步骤/材料清单\n"
    "请先切换更准确的分类，或补充上传对应制度、流程和通知文档。\n\n"
    "### 风险提示\n"
    "在没有明确制度依据前，请不要直接执行流程，建议联系对应部门进一步确认。"
)

QUESTION_TYPE_LABELS = {
    "policy_qa": "制度问答",
    "process_guide": "流程指引",
    "material_list": "材料清单",
    "notice_summary": "通知总结",
}

SAMPLE_DOCS = [
    {"file_name": "员工手册.txt", "category": "员工手册", "title": "员工手册（节选）", "version": "v2026.03"},
    {
        "file_name": "请假与考勤制度.txt",
        "category": "HR制度",
        "title": "请假与考勤制度",
        "version": "v2026.01",
    },
    {
        "file_name": "差旅与报销制度.txt",
        "category": "财务制度",
        "title": "差旅与报销制度",
        "version": "v2026.02",
    },
    {"file_name": "采购申请流程.txt", "category": "行政流程", "title": "采购申请流程", "version": "v2026.01"},
    {
        "file_name": "IT服务台常见问题.txt",
        "category": "IT支持",
        "title": "IT服务台常见问题",
        "version": "v2026.02",
    },
    {
        "file_name": "入职与权限开通流程.txt",
        "category": "综合公告",
        "title": "入职与权限开通流程",
        "version": "v2026.01",
    },
]


def get_dashscope_api_key(required=False):
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if required and not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，请通过环境变量提供模型密钥。")
    return api_key or None


def validate_settings():
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("OFFICEMATE_CHUNK_OVERLAP 必须大于等于 0 且小于 CHUNK_SIZE。")
    if chunk_min_size < 0 or chunk_min_size >= chunk_size:
        raise ValueError("OFFICEMATE_CHUNK_MIN_SIZE 必须大于等于 0 且小于 CHUNK_SIZE。")
    if not 0 <= min_relevance_score <= 1:
        raise ValueError("OFFICEMATE_MIN_RELEVANCE 必须位于 0 到 1 之间。")
    if not 0 <= vector_score_weight <= 1:
        raise ValueError("OFFICEMATE_VECTOR_WEIGHT 必须位于 0 到 1 之间。")
    if rrf_rank_constant <= 0 or bm25_k1 <= 0 or not 0 <= bm25_b <= 1:
        raise ValueError("RRF 与 BM25 参数配置无效。")
    if bm25_saturation <= 0 or vector_distance_scale <= 0:
        raise ValueError("BM25 饱和参数和向量距离缩放必须大于 0。")
    if reranker_top_k <= 0:
        raise ValueError("OFFICEMATE_RERANKER_TOP_K 必须大于 0。")
    if max_query_variants <= 0:
        raise ValueError("OFFICEMATE_MAX_QUERY_VARIANTS 必须大于 0。")
    if max_sub_queries <= 0:
        raise ValueError("OFFICEMATE_MAX_SUB_QUERIES 必须大于 0。")
    if max_workflow_retries < 0:
        raise ValueError("OFFICEMATE_MAX_WORKFLOW_RETRIES 必须大于等于 0。")
    if not 0 <= semantic_route_threshold <= 1:
        raise ValueError("OFFICEMATE_SEMANTIC_ROUTE_THRESHOLD 必须位于 0 到 1 之间。")
    if embedding2_score_weight < 0 or embedding2_score_weight > 1:
        raise ValueError("OFFICEMATE_EMBEDDING2_WEIGHT 必须位于 0 到 1 之间。")
    if vector_score_weight + embedding2_score_weight > 1:
        raise ValueError("VECTOR_WEIGHT + EMBEDDING2_WEIGHT 不能大于 1。")


def ensure_runtime_dirs():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_STORE_DIR.mkdir(parents=True, exist_ok=True)
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    for path in (DOCUMENT_INDEX_PATH, QA_LOG_PATH, FEEDBACK_PATH):
        if not path.exists():
            path.write_text("[]", encoding="utf-8")


validate_settings()

# Compatibility values retained for the original demo entry points.
md5_path = str(STORAGE_DIR / "legacy_md5.txt")
session_config = {"configurable": {"session_id": f"{default_session_prefix}_default"}}
