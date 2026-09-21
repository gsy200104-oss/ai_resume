"""简历知识库：Contextual Chunking、自动同步与意图感知向量检索。"""

from functools import lru_cache
from pathlib import Path

import hashlib
import re

import chromadb


# =========================================================
# 基础配置
# =========================================================

# 基础配置与索引格式保持兼容；
# 仅修改模型加载方式，不改变已有 Embedding 索引格式。

KNOWLEDGE_DIR = Path("knowledge")

CHROMA_DIR = Path("chroma_db")

HASH_FILE = CHROMA_DIR / "knowledge_hash.txt"

CANDIDATE_NAME = "高颂岩"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAME = "resume_knowledge"

INDEX_VERSION = "contextual-chunking-v3-knowledge-structure"


SOURCE_TYPE_MAP = {
    "education.md": "教育背景",
    "internships.md": "实习经历",
    "research.md": "科研经历",
    "projects.md": "项目经历",
    "skills.md": "技能与能力",
}


# =========================================================
# Chroma
# =========================================================

# Chroma 本身保留启动时初始化。
# 它远比 SentenceTransformer 模型轻，
# 暂时不需要额外复杂化。

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================================================
# Embedding Model Lazy Loading
# =========================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    懒加载 Embedding 模型。

    启动 Streamlit / import vector_store 时：
        不加载 MiniLM。

    第一次真正需要：
        - 构建 Knowledge Embedding
        - 用户执行向量检索

    时才加载模型。

    后续在同一个 Python 进程中直接复用，
    不会重复加载。
    """

    print(
        f"首次需要 Embedding Model，正在加载：{MODEL_NAME}"
    )

    # 放到函数内部，
    # 避免 import vector_store 时连
    # sentence-transformers 都提前加载。
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding Model 加载完成。"
    )

    return model


# =========================================================
# Knowledge 类型
# =========================================================

def get_source_type(file_name: str) -> str:
    return SOURCE_TYPE_MAP.get(
        file_name,
        "个人资料",
    )


# =========================================================
# Query Normalize
# =========================================================

def normalize_query(query: str) -> str:
    """
    仅为意图检测统一大小写和空白，
    不改变送入 Embedding 的原问题。
    """

    return re.sub(
        r"\s+",
        "",
        query.lower(),
    )


# =========================================================
# Intent Detection
# =========================================================

def detect_query_source_type(query: str):
    """
    只对明确的单一资料类别路由。

    不确定或跨类别时返回 None。
    """

    q = normalize_query(query)


    # -----------------------------------------------------
    # 实习
    # -----------------------------------------------------

    # “实习”是强类别信号，
    # 优先直接路由。
    if "实习" in q:
        return "实习经历"


    # -----------------------------------------------------
    # 管理 / 协作
    # -----------------------------------------------------

    # 管理和协作能力需要结合项目、技能等多来源，
    # 避免“项目”子串把它误判为单一项目经历。

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


    # -----------------------------------------------------
    # 各类别关键词
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # 论文
    # -----------------------------------------------------

    # “论文”本身和“有哪些/有什么”都不够明确：
    # 论文阅读、写作工具也可能属于技能或项目。
    #
    # 只有同时出现明确发表/成果/署名信号，
    # 才路由科研经历。

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


    # -----------------------------------------------------
    # 单一类别才启用 Metadata Filter
    # -----------------------------------------------------

    # 例如：
    # “教育背景、项目经历、科研经历”
    # 必须保留全库召回。

    if len(matches) == 1:
        return next(
            iter(matches)
        )

    return None


# =========================================================
# Markdown Chunk
# =========================================================

def get_section_title(chunk: str) -> str:

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
    过滤只有标题或空白的空壳 Chunk。
    """

    return any(

        line.strip()
        and not line.strip().startswith("#")

        for line
        in chunk.splitlines()
    )


def build_contextual_chunk(
    source: str,
    section_title: str,
    raw_content: str,
) -> str:
    """
    给原文添加：
    - 候选人
    - 资料类型
    - 章节

    保持 V3 索引内容格式。
    """

    return (

        f"候选人：{CANDIDATE_NAME}\n"

        f"资料类型："
        f"{get_source_type(source)}\n"

        f"章节："
        f"{section_title}\n\n"

        f"{raw_content}"

    ).strip()


def split_markdown_file(
    file_path: Path
) -> list:

    content = file_path.read_text(
        encoding="utf-8"
    )

    chunks = []


    for index, section in enumerate(
        content.split("\n## ")
    ):

        section = section.strip()

        if not section:
            continue


        raw_chunk = (
            section
            if index == 0
            else "## " + section
        )


        if not has_meaningful_content(
            raw_chunk
        ):

            print(
                f"跳过空壳 Chunk："
                f"{file_path.name}"
                f" → "
                f"{get_section_title(raw_chunk)}"
            )

            continue


        chunks.append(
            raw_chunk
        )


    return chunks


# =========================================================
# Load Knowledge
# =========================================================

def load_knowledge():

    documents = []

    markdown_files = sorted(
        KNOWLEDGE_DIR.glob("*.md")
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

        source = file_path.name

        source_type = (
            get_source_type(
                source
            )
        )


        if source not in SOURCE_TYPE_MAP:

            print(
                f"注意：{source} "
                f"未配置资料类型，"
                f"暂标记为「个人资料」。"
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
        f"\nKnowledge 总 Chunk 数："
        f"{len(documents)}"
    )


    return documents


# =========================================================
# Knowledge Hash
# =========================================================

def calculate_knowledge_hash():
    """
    以索引版本、文件名和文件原始字节
    计算 SHA256。

    沿用 V3 算法。
    """

    hasher = hashlib.sha256()

    hasher.update(
        INDEX_VERSION.encode(
            "utf-8"
        )
    )


    for file_path in sorted(
        KNOWLEDGE_DIR.glob("*.md")
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

    if not HASH_FILE.exists():
        return None


    return HASH_FILE.read_text(
        encoding="utf-8"
    ).strip()


def save_hash(
    hash_value: str
):

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HASH_FILE.write_text(
        hash_value,
        encoding="utf-8",
    )


# =========================================================
# Build Vector Store
# =========================================================

def build_vector_store():

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


    # -----------------------------------------------------
    # 到这里才真正需要 Embedding Model
    # -----------------------------------------------------

    embedding_model = (
        get_embedding_model()
    )


    print(
        "\n正在生成 Knowledge Embedding..."
    )


    embeddings = (
        embedding_model.encode(
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


    # -----------------------------------------------------
    # 新 Embedding 成功后，
    # 才删除旧 Chroma 记录。
    #
    # 避免模型异常时丢失已有索引。
    # -----------------------------------------------------

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
            f"\n正在删除旧索引："
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
        f"写入 Chunk 数量："
        f"{len(documents)}"
    )

    print(
        f"索引版本："
        f"{INDEX_VERSION}"
    )

    print(
        "=" * 60
    )


    return True


# =========================================================
# Auto Sync
# =========================================================

def ensure_vector_store():

    current_hash = (
        calculate_knowledge_hash()
    )

    saved_hash = (
        load_saved_hash()
    )


    if (
        current_hash != saved_hash
        or collection.count() == 0
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
# Metadata Filter Count
# =========================================================

def get_filtered_count(
    source_type: str
) -> int:

    results = collection.get(
        where={
            "source_type":
                source_type
        }
    )


    return len(
        results.get(
            "ids",
            [],
        )
    )


# =========================================================
# Chroma Query
# =========================================================

def query_chroma(
    query_embedding,
    top_k: int,
    source_type=None,
):
    """
    传入资料类别时使用 Metadata Filter，
    否则搜索全库。
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
# Search Vector Store
# =========================================================

def search_vector_store(
    query: str,
    top_k: int = 3,
    use_intent_filter: bool = True,
    debug: bool = False,
):
    """
    自动同步后检索。

    意图不明确或对应类别无数据时，
    回退全库。

    Embedding Model 只有执行到真正的
    Query Embedding 时才第一次加载。
    """

    if (
        not query.strip()
        or top_k <= 0
    ):
        return []


    # -----------------------------------------------------
    # Knowledge Auto Sync
    # -----------------------------------------------------

    ensure_vector_store()


    collection_count = (
        collection.count()
    )


    if collection_count == 0:
        return []


    # -----------------------------------------------------
    # Intent Detection
    # -----------------------------------------------------

    detected_source_type = (

        detect_query_source_type(
            query
        )

        if use_intent_filter

        else None
    )


    # -----------------------------------------------------
    # Lazy Load Embedding Model
    # -----------------------------------------------------

    embedding_model = (
        get_embedding_model()
    )


    # 保留原问题的空白和大小写；
    # normalize_query 只用于规则匹配。

    query_embedding = (
        embedding_model
        .encode(
            [query]
        )[0]
        .tolist()
    )


    # -----------------------------------------------------
    # Metadata Filter
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Vector Search
    # -----------------------------------------------------

    results = query_chroma(

        query_embedding=
            query_embedding,

        top_k=min(
            top_k,
            available_count,
        ),

        source_type=
            applied_source_type,
    )


    # -----------------------------------------------------
    # Debug
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Format Results
    # -----------------------------------------------------

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
# Local Test
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


    results = search_vector_store(

        query=question,

        top_k=5,

        use_intent_filter=True,

        debug=True,
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