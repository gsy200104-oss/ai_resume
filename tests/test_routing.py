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
"""

from agent_engine import (
    should_use_full_agent,
    is_overview_question,
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
        "根据这个岗位JD分析高颂岩的匹配情况："
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