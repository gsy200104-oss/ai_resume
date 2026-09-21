import json

from vector_store import search_vector_store


# =========================================================
# 1. Tool 1
# 搜索个人简历知识库
# =========================================================

def search_resume_knowledge(query: str):
    """
    搜索候选人高颂岩的个人知识库。

    适用于：
    - 教育背景
    - 实习经历
    - 科研经历
    - 项目经历
    - 技术能力

    在线问答不使用 Jina Reranker。
    直接使用 Intent-Aware Chroma Retrieval，
    返回 Top-3 相关内容给 LLM。
    """

    results = search_vector_store(
        query=query,
        top_k=5,
        use_intent_filter=True,
        debug=False,
    )

    if not results:
        return json.dumps(
            {
                "message": "没有找到相关候选人资料。",
                "results": [],
            },
            ensure_ascii=False,
        )

    formatted_results = []

    for result in results:
        formatted_results.append(
            {
                "source": result["source"],
                "source_type": result["source_type"],
                "section": result["section_title"],
                "content": result["raw_content"],
            }
        )

    return json.dumps(
        {
            "query": query,
            "results": formatted_results,
        },
        ensure_ascii=False,
    )


# =========================================================
# 2. Tool 2
# JD 岗位匹配分析
# =========================================================

def analyze_job_fit(job_description: str):
    """
    根据岗位 JD，
    从候选人知识库中寻找最相关的匹配证据。

    JD 往往同时涉及：
    - 教育
    - 技能
    - 项目
    - 科研
    - 实习

    因此使用 Top-5，
    给 LLM 提供更完整的候选人背景。
    """

    retrieval_query = (
        "请寻找候选人高颂岩与下面岗位要求最相关的"
        "教育背景、技术能力、项目经历、科研经历和实习经历。"
        "\n\n"
        f"岗位描述：\n{job_description}"
    )

    results = search_vector_store(
        query=retrieval_query,
        top_k=5,
        use_intent_filter=True,
        debug=False,
    )

    if not results:
        return json.dumps(
            {
                "message": "没有找到足够的岗位匹配资料。",
                "results": [],
            },
            ensure_ascii=False,
        )

    formatted_results = []

    for result in results:
        formatted_results.append(
            {
                "source": result["source"],
                "source_type": result["source_type"],
                "section": result["section_title"],
                "content": result["raw_content"],
            }
        )

    return json.dumps(
        {
            "job_description": job_description,
            "matching_evidence": formatted_results,
        },
        ensure_ascii=False,
    )


# =========================================================
# 3. Tool 3
# 面试问题生成
# =========================================================

def generate_interview_questions(job_description: str):
    """
    根据岗位 JD 和候选人经历，
    检索适合生成面试问题的个人经历证据。

    在线链路同样不使用 Jina，
    使用 Chroma Top-5 提供上下文。
    """

    retrieval_query = (
        "请寻找最适合用于面试追问的候选人经历，"
        "重点关注项目中的个人职责、技术实现、"
        "问题解决、实验设计、结果数据、"
        "项目管理和岗位相关能力。"
        "\n\n"
        f"岗位描述：\n{job_description}"
    )

    results = search_vector_store(
        query=retrieval_query,
        top_k=5,
        use_intent_filter=True,
        debug=False,
    )

    if not results:
        return json.dumps(
            {
                "message": "没有找到足够的面试问题生成依据。",
                "results": [],
            },
            ensure_ascii=False,
        )

    formatted_results = []

    for result in results:
        formatted_results.append(
            {
                "source": result["source"],
                "source_type": result["source_type"],
                "section": result["section_title"],
                "content": result["raw_content"],
            }
        )

    return json.dumps(
        {
            "job_description": job_description,
            "candidate_evidence": formatted_results,
        },
        ensure_ascii=False,
    )


# =========================================================
# 4. OpenAI / DeepSeek Tool Schemas
# =========================================================

TOOLS = [

    # -----------------------------------------------------
    # Tool 1
    # -----------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "search_resume_knowledge",

            "description": (
                "搜索候选人高颂岩的个人知识库，"
                "获取教育、实习、科研、项目、技能等真实资料。"
                "当用户询问候选人的具体个人经历或能力时使用。"
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要从候选人知识库中检索的问题。",
                    }
                },

                "required": ["query"],
            },
        },
    },


    # -----------------------------------------------------
    # Tool 2
    # -----------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "analyze_job_fit",

            "description": (
                "根据招聘岗位 JD 检索候选人与岗位最相关的"
                "教育、技能、科研、项目和实习证据，"
                "用于进行岗位匹配分析。"
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "job_description": {
                        "type": "string",
                        "description": "完整或主要岗位描述 JD。",
                    }
                },

                "required": ["job_description"],
            },
        },
    },


    # -----------------------------------------------------
    # Tool 3
    # -----------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "generate_interview_questions",

            "description": (
                "根据岗位 JD 和候选人真实经历，"
                "检索适合进行技术面试、项目追问和行为面试的证据。"
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "job_description": {
                        "type": "string",
                        "description": "用于生成针对性面试问题的岗位描述 JD。",
                    }
                },

                "required": ["job_description"],
            },
        },
    },
]