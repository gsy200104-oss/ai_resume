from pathlib import Path

KNOWLEDGE_DIR = Path("knowledge")

SOURCE_KEYWORDS = {
    "education.md": [
        "教育", "学校", "大学", "专业", "学历", "本科", "硕士", "毕业"
    ],

    "research.md": [
        "科研", "研究", "课题", "实验", "论文", "研究方法", "研究成果"
    ],

    "projects.md": [
        "项目", "开发", "负责", "功能", "实现", "技术", "难点", "成果"
    ],

    "skills.md": [
        "技能", "Python", "Streamlit", "API", "LLM", "RAG",
        "数据分析", "技术栈"
    ]
}


def load_knowledge():
    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.md"):
        content = file_path.read_text(encoding="utf-8")

        sections = content.split("\n## ")

        chunks = []

        for i, section in enumerate(sections):
            if i == 0:
                chunks.append(section)
            else:
                chunks.append("## " + section)

        for chunk in chunks:
            if chunk.strip():
               documents.append({
                  "source": file_path.name,
                   "content": chunk.strip()
               })
    return documents


def retrieve(query, top_k=3):
    documents = load_knowledge()
    scored_documents = []

    for document in documents:
        source = document["source"]
        content = document["content"]
        keywords = SOURCE_KEYWORDS.get(source, [])

        score = 0

        for keyword in keywords:
            keyword_lower = keyword.lower()
            query_lower = query.lower()
            content_lower = content.lower()

            if (
                keyword_lower in query_lower
                and keyword_lower in content_lower
            ):
                score += 3

        if score > 0:
            scored_documents.append({
                "source": source,
                "content": content,
                "score": score
            })

    # 按分数从高到低排序
    scored_documents.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # 只返回前 top_k 个
    return scored_documents[:top_k]

if __name__ == "__main__":
    question = input("请输入问题：")

    results = retrieve(question)

    if results:
        for i, result in enumerate(results, start=1):
            print(f"\n第 {i} 条资料：")
            print("来源：", result["source"])
            print("相关度分数：", result["score"])
            print("内容：")
            print(result["content"])
    else:
        print("没有找到相关资料。")