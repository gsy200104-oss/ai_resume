"""
AI Resume Agent 路由测试

测试目标：

1. 普通简历问题
   → Fast Path

2. 具体项目问题
   → Fast Path

3. JD / 岗位匹配
   → Full Agent

4. 面试问题生成
   → Full Agent

5. 概览问题
   → Compact Context

6. Intent Router
   → 正确识别单意图和多意图
"""

from agent_engine import (
    should_use_full_agent,
    is_overview_question,
)

from vector_store import (
    detect_query_source_types,
)


# =========================================================
# Fast Path
# =========================================================

def test_ai_projects_use_fast_path():
    question = "高颂岩做过哪些AI项目？"

    assert (
        should_use_full_agent(question)
        is False
    )


def test_ai_resume_detail_use_fast_path():
    question = (
        "高颂岩的AI Resume Agent是怎么实现的？"
    )

    assert (
        should_use_full_agent(question)
        is False
    )


def test_research_question_use_fast_path():
    question = "高颂岩的硕士课题是什么？"

    assert (
        should_use_full_agent(question)
        is False
    )


# =========================================================
# Full Agent
# =========================================================

def test_job_fit_use_full_agent():
    question = (
        "根据这个岗位JD分析高颂岩的匹配情况，"
        "要求熟悉Python、RAG和大模型应用开发。"
    )

    assert (
        should_use_full_agent(question)
        is True
    )


def test_job_match_use_full_agent():
    question = (
        "分析高颂岩和这个岗位的匹配度。"
    )

    assert (
        should_use_full_agent(question)
        is True
    )


def test_interview_question_use_full_agent():
    question = (
        "根据这个岗位生成针对性的面试问题。"
    )

    assert (
        should_use_full_agent(question)
        is True
    )


# =========================================================
# Compact Context
# =========================================================

def test_overview_question_detected():
    question = "高颂岩做过哪些AI项目？"

    assert (
        is_overview_question(question)
        is True
    )


def test_skill_overview_detected():
    question = "高颂岩有哪些技能？"

    assert (
        is_overview_question(question)
        is True
    )


# =========================================================
# Full Context
# =========================================================

def test_detail_question_not_overview():
    question = (
        "高颂岩的AI Resume Agent是怎么实现的？"
    )

    assert (
        is_overview_question(question)
        is False
    )


# =========================================================
# Intent Router
# =========================================================

def test_single_internship_intent():
    question = (
        "高颂岩有哪些实习经历？"
    )

    assert (
        detect_query_source_types(question)
        == ["实习经历"]
    )


def test_research_and_internship_multi_intent():
    question = (
        "请介绍高颂岩的科研经历和实习经历。"
    )

    intents = (
        detect_query_source_types(
            question
        )
    )

    assert "科研经历" in intents
    assert "实习经历" in intents
    assert len(intents) == 2


def test_ai_project_and_skill_multi_intent():
    question = (
        "他有哪些AI项目和工程能力？"
    )

    intents = (
        detect_query_source_types(
            question
        )
    )

    assert "项目经历" in intents
    assert "技能与能力" in intents


def test_rag_project_intent():
    question = (
        "RAG系统是怎么实现和优化的？"
    )

    assert (
        detect_query_source_types(question)
        == ["项目经历"]
    )


def test_agent_tool_project_intent():
    question = (
        "Agent如何调用多个Tool？"
    )

    assert (
        detect_query_source_types(question)
        == ["项目经历"]
    )


def test_full_candidate_intro_multi_intent():
    question = (
        "请从招聘方视角介绍高颂岩，"
        "重点说明教育背景、科研与实习经历、"
        "AI项目和技术能力。"
    )

    intents = set(
        detect_query_source_types(
            question
        )
    )

    assert intents == {
        "教育背景",
        "科研经历",
        "实习经历",
        "项目经历",
        "技能与能力",
    }