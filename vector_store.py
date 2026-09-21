from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# 基础配置
# =========================================================

KNOWLEDGE_DIR = Path("knowledge")

CHROMA_DIR = Path("chroma_db")

HASH_FILE = CHROMA_DIR / "knowledge_hash.txt"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAME = "resume_knowledge"


# =========================================================
# Embedding 模型
# =========================================================

model = SentenceTransformer(MODEL_NAME)


# =========================================================
# Chroma Client
# =========================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================================================
# 读取并切分知识库
# =========================================================

def load_knowledge():

    documents = []

    for file_path in sorted(
        KNOWLEDGE_DIR.glob("*.md")
    ):

        content = file_path.read_text(
            encoding="utf-8"
        )

        sections = content.split("\n## ")

        for i, section in enumerate(sections):

            if i == 0:
                chunk = section
            else:
                chunk = "## " + section

            if chunk.strip():

                documents.append(
                    {
                        "source": file_path.name,
                        "content": chunk.strip()
                    }
                )

    return documents


# =========================================================
# 计算 Knowledge 指纹
# =========================================================

def calculate_knowledge_hash():
    """
    根据所有 Markdown 文件内容计算 SHA256。

    只要 knowledge 中任何文件发生变化，
    得到的 hash 就会变化。
    """

    hasher = hashlib.sha256()

    for file_path in sorted(
        KNOWLEDGE_DIR.glob("*.md")
    ):

        hasher.update(
            file_path.name.encode("utf-8")
        )

        hasher.update(
            file_path.read_bytes()
        )

    return hasher.hexdigest()


# =========================================================
# 读取上一次 Knowledge 指纹
# =========================================================

def load_saved_hash():

    if not HASH_FILE.exists():
        return None

    return HASH_FILE.read_text(
        encoding="utf-8"
    ).strip()


# =========================================================
# 保存 Knowledge 指纹
# =========================================================

def save_hash(hash_value):

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HASH_FILE.write_text(
        hash_value,
        encoding="utf-8"
    )


# =========================================================
# 建立 / 重建向量数据库
# =========================================================

def build_vector_store():

    documents = load_knowledge()

    if not documents:
        print("知识库为空。")
        return


    # -----------------------------------------------------
    # 清除旧向量
    # -----------------------------------------------------

    existing_data = collection.get()

    existing_ids = existing_data.get(
        "ids",
        []
    )

    if existing_ids:

        collection.delete(
            ids=existing_ids
        )


    # -----------------------------------------------------
    # 文本
    # -----------------------------------------------------

    texts = [
        item["content"]
        for item in documents
    ]


    # -----------------------------------------------------
    # Embedding
    # -----------------------------------------------------

    embeddings = model.encode(
        texts
    ).tolist()


    # -----------------------------------------------------
    # ID
    # -----------------------------------------------------

    ids = [
        f"doc_{i}"
        for i in range(len(documents))
    ]


    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadatas = [
        {
            "source": item["source"]
        }
        for item in documents
    ]


    # -----------------------------------------------------
    # 写入 Chroma
    # -----------------------------------------------------

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )


    print(
        f"Chroma 向量库更新完成，"
        f"共写入 {len(documents)} 个知识片段。"
    )


# =========================================================
# 自动同步 Knowledge → Chroma
# =========================================================

def ensure_vector_store():

    current_hash = (
        calculate_knowledge_hash()
    )

    saved_hash = load_saved_hash()


    # Chroma 本身是否为空
    collection_count = (
        collection.count()
    )


    if (
        current_hash != saved_hash
        or collection_count == 0
    ):

        print(
            "检测到 Knowledge 更新，"
            "正在同步 Chroma..."
        )

        build_vector_store()

        save_hash(
            current_hash
        )

    else:

        print(
            "Knowledge 未变化，"
            "直接使用现有 Chroma 向量库。"
        )


# =========================================================
# 向量检索
# =========================================================

def search_vector_store(
    query: str,
    top_k: int = 3
):

    # 每次检索前确保数据库有效
    ensure_vector_store()


    query_embedding = model.encode(
        [query]
    )[0].tolist()


    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k
    )


    search_results = []


    for i in range(
        len(results["documents"][0])
    ):

        search_results.append(
            {
                "source":
                    results["metadatas"][0][i][
                        "source"
                    ],

                "content":
                    results["documents"][0][i],

                "distance":
                    results["distances"][0][i]
            }
        )


    return search_results


# =========================================================
# 本地测试
# =========================================================

if __name__ == "__main__":

    ensure_vector_store()

    question = input(
        "\n请输入测试问题："
    )

    results = search_vector_store(
        question,
        top_k=3
    )


    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n第 {i} 条结果："
        )

        print(
            "来源：",
            result["source"]
        )

        print(
            "距离：",
            result["distance"]
        )

        print("内容：")

        print(
            result["content"]
        )