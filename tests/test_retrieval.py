"""
AI Resume Agent Retrieval 集成测试

验证目标：

1. 科研 + 实习
   → 必须同时召回 research.md 和 internships.md

2. AI 项目 + 技术能力
   → 必须同时召回 projects.md 和 skills.md

3. 综合候选人介绍
   → 必须覆盖教育、科研、实习、项目、技能五类资料

注意：
这些测试会真正调用 Embedding Model 和 Chroma，
因此比 test_routing.py 慢。
"""

from vector_store import (
    search_vector_store,
)


def get_sources(results):
    """
    提取检索结果中的 source 文件名。
    """

    return {
        item["source"]
        for item in results
    }


# =========================================================
# Research + Internship
# =========================================================

def test_research_and_internship_retrieval():
    question = (
        "请介绍高颂岩主要的科研经历和实习经历，"
        "并说明这些经历体现了哪些能力。"
    )

    results = search_vector_store(
        question,
        top_k=5,
    )

    sources = get_sources(
        results
    )

    assert "research.md" in sources
    assert "internships.md" in sources


# =========================================================
# AI Projects + Skills
# =========================================================

def test_ai_project_and_skill_retrieval():
    question = (
        "高颂岩有哪些AI项目和工程能力？"
    )

    results = search_vector_store(
        question,
        top_k=5,
    )

    sources = get_sources(
        results
    )

    assert "projects.md" in sources
    assert "skills.md" in sources


# =========================================================
# Full Candidate Introduction
# =========================================================

def test_full_candidate_intro_retrieval():
    question = (
        "请从招聘方视角，用大约1分钟的篇幅介绍候选人高颂岩，"
        "重点说明教育背景、科研与实习经历、"
        "AI项目、技术能力和求职方向。"
    )

    results = search_vector_store(
        question,
        top_k=5,
    )

    sources = get_sources(
        results
    )

    expected_sources = {
        "education.md",
        "research.md",
        "internships.md",
        "projects.md",
        "skills.md",
    }

    assert expected_sources.issubset(
        sources
    )