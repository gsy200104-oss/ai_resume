"""
AI Resume Agent 全局配置

作用：
统一管理项目中的模型、检索、Agent、API 等核心参数。

后续：
- Streamlit
- FastAPI
- Agent
- Retrieval
- Docker / Deployment

都可以从这里读取配置。
"""


# =========================================================
# 1. Candidate
# =========================================================

CANDIDATE_NAME = "高颂岩"


# =========================================================
# 2. LLM
# =========================================================

LLM_MODEL = "deepseek-chat"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# =========================================================
# 3. Embedding
# =========================================================

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================================================
# 4. Retrieval
# =========================================================

# 普通简历知识问答召回数量
RESUME_RETRIEVAL_TOP_K = 5

# JD 岗位匹配召回数量
JOB_FIT_TOP_K = 5

# 面试问题生成召回数量
INTERVIEW_RETRIEVAL_TOP_K = 5


# =========================================================
# 5. Vector Store
# =========================================================

CHROMA_DIR = "chroma_db"

CHROMA_COLLECTION_NAME = "resume_knowledge"

KNOWLEDGE_DIR = "knowledge"

INDEX_VERSION = (
    "contextual-chunking-v3-knowledge-structure"
)


# =========================================================
# 6. Agent
# =========================================================

# Full Agent 最大循环次数
AGENT_MAX_STEPS = 5


# =========================================================
# 7. API
# =========================================================

API_TITLE = "AI Resume Agent API"

API_VERSION = "1.1.0"

API_HOST = "127.0.0.1"

API_PORT = 8000


# =========================================================
# 8. Performance
# =========================================================

# 概览型问题使用 Compact Context
ENABLE_COMPACT_CONTEXT = True

# 每个 Chunk 在 Compact Context 中
# 最多保留几条关键信息
COMPACT_CONTEXT_MAX_LINES = 2

# 单条关键信息最大字符数
COMPACT_CONTEXT_MAX_CHARS = 220