"""
简历知识库：
Contextual Chunking、自动同步与意图感知向量检索。
"""

from functools import lru_cache
from pathlib import Path

import hashlib
import re

import chromadb

from config import (
    CANDIDATE_NAME,
    EMBEDDING_MODEL,
    CHROMA_DIR as CHROMA_DIR_CONFIG,
    CHROMA_COLLECTION_NAME,
    KNOWLEDGE_DIR as KNOWLEDGE_DIR_CONFIG,
    INDEX_VERSION,
)


# =========================================================
# 1. 基础配置
# =========================================================

KNOWLEDGE_DIR = Path(
    KNOWLEDGE_DIR_CONFIG
)

CHROMA_DIR = Path(
    CHROMA_DIR_CONFIG
)

HASH_FILE = (
    CHROMA_DIR
    / "knowledge_hash.txt"
)

MODEL_NAME = EMBEDDING_MODEL

COLLECTION_NAME = (
    CHROMA_COLLECTION_NAME
)


SOURCE_TYPE_MAP = {
    "education.md": "教育背景",
    "internships.md": "实习经历",
    "research.md": "科研经历",
    "projects.md": "项目经历",
    "skills.md": "技能与能力",
}


# =========================================================
# 2. Chroma
# =========================================================

# Chroma 客户端本身比较轻，
# 可以在模块导入时初始化。
#
# Embedding Model 则继续使用 Lazy Loading，
# 避免网页 / API 启动时加载 MiniLM。

chroma_client = (
    chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )
)

collection = (
    chroma_client
    .get_or_create_collection(
        name=COLLECTION_NAME
    )
)


# =========================================================
# 3. Embedding Model Lazy Loading
# =========================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    懒加载 Embedding Model。

    启动 Streamlit / FastAPI 时：
        不加载 MiniLM。

    第一次真正需要进行：
        - Query Embedding
        - Knowledge 重建

    时才加载。

    同一 Python 进程后续直接复用模型。
    """

    print(
        "首次需要 Embedding Model，"
        f"正在加载：{MODEL_NAME}"
    )

    # 放到函数内部导入，
    # 避免 import vector_store 时提前加载
    # sentence-transformers。
    from sentence_transformers import (
        SentenceTransformer,
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding Model 加载完成。"
    )

    return model


# =========================================================
# 4. Knowledge Source Type
# =========================================================

def get_source_type(
    file_name: str
) -> str:
    """
    根据 Knowledge 文件名判断资料类型。
    """

    return SOURCE_TYPE_MAP.get(
        file_name,
        "个人资料",
    )


# =========================================================
# 5. Query Normalize
# =========================================================

def normalize_query(
    query: str
) -> str:
    """
    仅用于 Intent Detection：

    - 转小写
    - 去掉空白

    不改变真正送入 Embedding Model 的原始问题。
    """

    return re.sub(
        r"\s+",
        "",
        query.lower(),
    )


# =========================================================
# 6. Intent Detection
# =========================================================

def detect_query_source_type(
    query: str
):
    """
    判断 Query 是否明确属于某一种资料类别。

    如果明确：
        返回 source_type，
        后续使用 Metadata Filter。

    如果不明确或涉及多个类别：
        返回 None，
        使用全库向量检索。
    """

    q = normalize_query(
        query
    )


    # =====================================================
    # 实习
    # =====================================================

    # “实习”属于强类别信号。
    if "实习" in q:

        return "实习经历"


    # =====================================================
    # 管理 / 协作
    # =====================================================

    # 例如：
    # “项目管理能力怎么样”
    #
    # 这种问题不能简单路由到 projects.md，
    # 因为还可能需要 skills.md。

    if (
        (
            "项目" in q
            and any(
                pattern in q
                for pattern in (
                    "管理",
                    "统筹",
                    "协作",
                    "协调",
                )
            )
        )
        or any(
            pattern in q
            for pattern in (
                "团队协作",
                "团队合作",
            )
        )
    ):

        return None


    # =====================================================
    # Research
    # =====================================================

    research_patterns = (
        "硕士课题",
        "科研经历",
        "科研工作",
        "科研项目",
        "研究经历",
        "研究工作",
        "研究方向",
        "科研成果",
        "做过哪些科研",
        "做过什么科研",
    )


    # =====================================================
    # Education
    # =====================================================

    education_patterns = (
        "教育背景",
        "专业背景",
        "什么专业",
        "哪个专业",
        "所学专业",
        "学历背景",
        "什么学历",
        "毕业院校",
        "哪个学校",
        "什么学校",
        "本科教育",
        "硕士教育",
        "本科专业",
        "硕士专业",
    )


    # =====================================================
    # Projects
    # =====================================================

    project_patterns = (
        "项目经历",
        "项目经验",
        "ai项目",
        "人工智能项目",
        "大模型项目",
        "软件项目",
        "工程项目",
        "ai简历",
        "resumeagent",
        "知识库助手",
        "科研文献智能分析agent",
        "文献智能分析agent",
    )


    # =====================================================
    # Skills
    # =====================================================

    skill_patterns = (
        "技能有哪些",
        "有哪些技能",
        "技术栈",
        "会哪些技术",
        "会什么技术",
        "掌握哪些技术",
        "软件开发技术",
        "ai开发技术",
        "ai技术能力",
        "会哪些软件",
        "会什么软件",
        "编程能力",
    )


    # =====================================================
    # Collect Matches
    # =====================================================

    matches = set()


    for source_type, patterns in (
        (
            "科研经历",
            research_patterns,
        ),
        (
            "教育背景",
            education_patterns,
        ),
        (
            "项目经历",
            project_patterns,
        ),
        (
            "技能与能力",
            skill_patterns,
        ),
    ):

        if any(
            pattern in q
            for pattern in patterns
        ):

            matches.add(
                source_type
            )


    # =====================================================
    # Publication
    # =====================================================

    # “论文”本身不能直接认为是科研经历。
    #
    # 例如：
    # “会不会进行论文分析？”
    #
    # 可能是在问 AI / 技能。
    #
    # 只有同时出现：
    # 发表、DOI、作者、成果等信息时，
    # 才判断为科研经历。

    publication_patterns = (
        "发表",
        "成果",
        "作者",
        "doi",
        "论文题目",
        "论文名称",
        "论文标题",
    )


    if (
        "论文" in q
        and any(
            pattern in q
            for pattern
            in publication_patterns
        )
    ):

        matches.add(
            "科研经历"
        )


    # =====================================================
    # Only One Category
    # =====================================================

    if len(matches) == 1:

        return next(
            iter(matches)
        )


    return None


# =========================================================
# 7. Markdown Chunk Helpers
# =========================================================

def get_section_title(
    chunk: str
) -> str:
    """
    从 Chunk 中寻找第一个 Markdown 标题。
    """

    for line in chunk.splitlines():

        line = line.strip()

        if line.startswith("#"):

            return (
                line
                .lstrip("#")
                .strip()
            )


    return "概览"


def has_meaningful_content(
    chunk: str
) -> bool:
    """
    过滤只有标题、没有正文内容的空壳 Chunk。
    """

    return any(

        line.strip()
        and not line
        .strip()
        .startswith("#")

        for line
        in chunk.splitlines()
    )


# =========================================================
# 8. Contextual Chunk
# =========================================================

def build_contextual_chunk(
    source: str,
    section_title: str,
    raw_content: str,
) -> str:
    """
    给原始 Knowledge Chunk 添加：

    - 候选人
    - 资料类型
    - 章节标题

    提高 Embedding 检索语义。
    """

    return (

        f"候选人："
        f"{CANDIDATE_NAME}\n"

        f"资料类型："
        f"{get_source_type(source)}\n"

        f"章节："
        f"{section_title}\n\n"

        f"{raw_content}"

    ).strip()


# =========================================================
# 9. Markdown Split
# =========================================================

def split_markdown_file(
    file_path: Path
) -> list:
    """
    以 Markdown 二级标题 ## 为主要 Chunk 边界。
    """

    content = (
        file_path.read_text(
            encoding="utf-8"
        )
    )

    chunks = []


    for index, section in enumerate(
        content.split("\n## ")
    ):

        section = (
            section.strip()
        )


        if not section:

            continue


        raw_chunk = (

            section

            if index == 0

            else "## " + section
        )


        # 跳过只有标题的 Chunk
        if not has_meaningful_content(
            raw_chunk
        ):

            print(
                "跳过空壳 Chunk："
                f"{file_path.name}"
                " → "
                f"{get_section_title(raw_chunk)}"
            )

            continue


        chunks.append(
            raw_chunk
        )


    return chunks


# =========================================================
# 10. Load Knowledge
# =========================================================

def load_knowledge():
    """
    加载 knowledge/*.md，
    并转换为结构化 Documents。
    """

    documents = []


    markdown_files = sorted(
        KNOWLEDGE_DIR.glob(
            "*.md"
        )
    )


    if not markdown_files:

        print(
            "没有找到 Knowledge 文件。"
        )

        return documents


    print(
        "\n" + "=" * 60
    )

    print(
        "Loading Knowledge Base"
    )

    print(
        "=" * 60
    )


    for file_path in markdown_files:

        source = (
            file_path.name
        )


        source_type = (
            get_source_type(
                source
            )
        )


        if source not in SOURCE_TYPE_MAP:

            print(
                f"注意：{source} "
                "未配置资料类型，"
                "暂标记为「个人资料」。"
            )


        chunks = (
            split_markdown_file(
                file_path
            )
        )


        print(
            f"{source}: "
            f"{len(chunks)} 个有效 Chunk"
        )


        for raw_chunk in chunks:

            section_title = (
                get_section_title(
                    raw_chunk
                )
            )


            documents.append(
                {
                    "source":
                        source,

                    "source_type":
                        source_type,

                    "section_title":
                        section_title,

                    "raw_content":
                        raw_chunk,

                    "content":
                        build_contextual_chunk(
                            source,
                            section_title,
                            raw_chunk,
                        ),
                }
            )


    print(
        "\nKnowledge 总 Chunk 数："
        f"{len(documents)}"
    )


    return documents


# =========================================================
# 11. Knowledge Hash
# =========================================================

def calculate_knowledge_hash():
    """
    计算 Knowledge 当前版本的 SHA256。

    Hash 内容包括：

    - INDEX_VERSION
    - Knowledge 文件名
    - Knowledge 文件原始内容

    如果内容改变，则自动重建 Chroma。
    """

    hasher = (
        hashlib.sha256()
    )


    hasher.update(
        INDEX_VERSION.encode(
            "utf-8"
        )
    )


    for file_path in sorted(
        KNOWLEDGE_DIR.glob(
            "*.md"
        )
    ):

        hasher.update(
            file_path.name.encode(
                "utf-8"
            )
        )

        hasher.update(
            file_path.read_bytes()
        )


    return hasher.hexdigest()


def load_saved_hash():
    """
    读取上一次 Knowledge Hash。
    """

    if not HASH_FILE.exists():

        return None


    return (
        HASH_FILE.read_text(
            encoding="utf-8"
        )
        .strip()
    )


def save_hash(
    hash_value: str
):
    """
    保存 Knowledge Hash。
    """

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    HASH_FILE.write_text(
        hash_value,
        encoding="utf-8",
    )


# =========================================================
# 12. Build Vector Store
# =========================================================

def build_vector_store():
    """
    根据 Knowledge 重建 Chroma Vector Store。
    """

    documents = (
        load_knowledge()
    )


    if not documents:

        print(
            "Knowledge Base 为空，"
            "停止构建 Vector Store。"
        )

        return False


    texts = [

        document["content"]

        for document
        in documents
    ]


    # =====================================================
    # 这里才真正加载 Embedding Model
    # =====================================================

    embedding_model = (
        get_embedding_model()
    )


    print(
        "\n正在生成 Knowledge Embedding..."
    )


    embeddings = (

        embedding_model
        .encode(
            texts,
            show_progress_bar=True,
        )
        .tolist()
    )


    ids = [

        f"resume_chunk_{index}"

        for index
        in range(
            len(documents)
        )
    ]


    metadatas = [

        {
            "source":
                document["source"],

            "source_type":
                document["source_type"],

            "section_title":
                document["section_title"],

            "raw_content":
                document["raw_content"],
        }

        for document
        in documents
    ]


    # =====================================================
    # Embedding 成功生成后，
    # 才删除旧 Chroma 数据。
    #
    # 防止模型出错时旧索引直接丢失。
    # =====================================================

    existing_ids = (

        collection
        .get()
        .get(
            "ids",
            [],
        )
    )


    if existing_ids:

        print(
            "\n正在删除旧索引："
            f"{len(existing_ids)} 条"
        )

        collection.delete(
            ids=existing_ids
        )


    collection.upsert(

        ids=ids,

        documents=texts,

        embeddings=embeddings,

        metadatas=metadatas,
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "Chroma Vector Store 构建完成"
    )

    print(
        "写入 Chunk 数量："
        f"{len(documents)}"
    )

    print(
        "索引版本："
        f"{INDEX_VERSION}"
    )

    print(
        "=" * 60
    )


    return True


# =========================================================
# 13. Auto Sync
# =========================================================

def ensure_vector_store():
    """
    检查 Knowledge 是否变化。

    如果：
    - Knowledge 内容变化
    - INDEX_VERSION 变化
    - Chroma 为空

    则自动重建索引。
    """

    current_hash = (
        calculate_knowledge_hash()
    )


    saved_hash = (
        load_saved_hash()
    )


    if (
        current_hash
        != saved_hash

        or collection.count()
        == 0
    ):

        print(
            "\n检测到 Knowledge "
            "或索引策略更新。"
        )

        print(
            "正在重新构建 Chroma..."
        )


        if build_vector_store():

            save_hash(
                current_hash
            )


    else:

        print(
            "Knowledge 与索引策略未变化，"
            "直接使用现有 Chroma 向量库。"
        )


# =========================================================
# 14. Metadata Filter Count
# =========================================================

def get_filtered_count(
    source_type: str
) -> int:
    """
    查询某一 source_type
    在 Chroma 中包含多少 Chunk。
    """

    results = (
        collection.get(
            where={
                "source_type":
                    source_type
            }
        )
    )


    return len(
        results.get(
            "ids",
            [],
        )
    )


# =========================================================
# 15. Chroma Query
# =========================================================

def query_chroma(
    query_embedding,
    top_k: int,
    source_type=None,
):
    """
    执行 Chroma 查询。

    source_type 存在：
        Metadata Filter

    source_type=None：
        全库向量检索
    """

    query_args = {

        "query_embeddings":
            [query_embedding],

        "n_results":
            top_k,
    }


    if source_type:

        query_args[
            "where"
        ] = {

            "source_type":
                source_type
        }


    return collection.query(
        **query_args
    )


# =========================================================
# 16. Search Vector Store
# =========================================================

def search_vector_store(
    query: str,
    top_k: int = 3,
    use_intent_filter: bool = True,
    debug: bool = False,
):
    """
    AI Resume Retrieval 主入口。

    Pipeline：

    Query
        ↓
    Knowledge Auto Sync
        ↓
    Intent Detection
        ↓
    MiniLM Query Embedding
        ↓
    Metadata Filter（如果明确）
        ↓
    Chroma Vector Search
        ↓
    Top-K Results
    """

    if (
        not query.strip()
        or top_k <= 0
    ):

        return []


    # =====================================================
    # Knowledge Auto Sync
    # =====================================================

    ensure_vector_store()


    collection_count = (
        collection.count()
    )


    if collection_count == 0:

        return []


    # =====================================================
    # Intent Detection
    # =====================================================

    detected_source_type = (

        detect_query_source_type(
            query
        )

        if use_intent_filter

        else None
    )


    # =====================================================
    # Lazy Load Embedding Model
    # =====================================================

    embedding_model = (
        get_embedding_model()
    )


    # 保留用户原始问题，
    # normalize_query 只用于规则匹配。

    query_embedding = (

        embedding_model
        .encode(
            [query]
        )[0]
        .tolist()
    )


    # =====================================================
    # Metadata Filter
    # =====================================================

    applied_source_type = None

    retrieval_mode = (
        "global_vector_search"
    )

    available_count = (
        collection_count
    )


    if detected_source_type:

        filtered_count = (
            get_filtered_count(
                detected_source_type
            )
        )


        if filtered_count > 0:

            applied_source_type = (
                detected_source_type
            )

            available_count = (
                filtered_count
            )

            retrieval_mode = (
                "metadata_filtered_"
                "vector_search"
            )


    # =====================================================
    # Chroma Search
    # =====================================================

    results = (
        query_chroma(

            query_embedding=
                query_embedding,

            top_k=min(
                top_k,
                available_count,
            ),

            source_type=
                applied_source_type,
        )
    )


    # =====================================================
    # Debug
    # =====================================================

    if debug:

        print(
            "\n" + "=" * 60
        )

        print(
            "Retrieval Routing"
        )

        print(
            "=" * 60
        )

        print(
            "Query：",
            query,
        )

        print(
            "Intent Detection：",
            detected_source_type
            or "无明确单一类别",
        )

        print(
            "Retrieval Mode：",
            retrieval_mode,
        )

        print(
            "Metadata Filter：",
            applied_source_type
            or "未启用",
        )


    # =====================================================
    # Format Results
    # =====================================================

    documents = (

        results
        .get(
            "documents",
            [[]],
        )[0]
    )


    metadatas = (

        results
        .get(
            "metadatas",
            [[]],
        )[0]
    )


    distances = (

        results
        .get(
            "distances",
            [[]],
        )[0]
    )


    search_results = []


    for index, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[index]
            or {}
        )


        search_results.append(
            {
                "source":
                    metadata.get(
                        "source",
                        "",
                    ),

                "source_type":
                    metadata.get(
                        "source_type",
                        "",
                    ),

                "section_title":
                    metadata.get(
                        "section_title",
                        "",
                    ),

                "content":
                    document,

                "raw_content":
                    metadata.get(
                        "raw_content",
                        "",
                    ),

                "distance":
                    distances[index],

                "retrieval_mode":
                    retrieval_mode,

                "metadata_filter":
                    applied_source_type,
            }
        )


    return search_results


# =========================================================
# 17. Local Test
# =========================================================

if __name__ == "__main__":

    ensure_vector_store()


    print(
        "\n" + "=" * 60
    )

    print(
        "AI Resume Intent-Aware Retrieval Test"
    )

    print(
        "=" * 60
    )


    question = input(
        "\n请输入测试问题："
    )


    results = (
        search_vector_store(
            query=question,
            top_k=5,
            use_intent_filter=True,
            debug=True,
        )
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "Retrieval Results"
    )

    print(
        "=" * 60
    )


    if not results:

        print(
            "没有检索到结果。"
        )


    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            "\n" + "-" * 60
        )

        print(
            f"第 {index} 条结果"
        )

        print(
            "来源：",
            result["source"],
        )

        print(
            "资料类型：",
            result["source_type"],
        )

        print(
            "章节：",
            result["section_title"],
        )

        print(
            "距离：",
            result["distance"],
        )

        print(
            "Retrieval Mode：",
            result["retrieval_mode"],
        )

        print(
            "Metadata Filter：",
            result["metadata_filter"],
        )

        print(
            "\n原始 Knowledge："
        )

        print(
            result["raw_content"]
        )