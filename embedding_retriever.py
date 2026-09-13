from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_DIR = Path("knowledge")

# 加载 Embedding 模型
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def load_knowledge():
    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.md"):
        content = file_path.read_text(encoding="utf-8")

        # 按 Markdown 二级标题切块
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


def retrieve(query, top_k=3):
    documents = load_knowledge()

    contents = [
        document["content"]
        for document in documents
    ]

    # 把知识片段变成向量
    document_vectors = model.encode(contents)

    # 把用户问题变成向量
    query_vector = model.encode([query])

    # 计算问题和每个知识片段的相似度
    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    results = []

    for document, score in zip(
        documents,
        similarities
    ):
        results.append(
            {
                "source": document["source"],
                "content": document["content"],
                "score": float(score)
            }
        )

    # 按相似度从高到低排序
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":
    question = input("请输入问题：")

    results = retrieve(
        question,
        top_k=3
    )

    for i, result in enumerate(
        results,
        start=1
    ):
        print(f"\n第 {i} 条资料：")
        print("来源：", result["source"])
        print("相似度：", result["score"])
        print("内容：")
        print(result["content"])