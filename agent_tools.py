from embedding_retriever import retrieve


# =========================================================
# Tool 1：搜索个人简历知识库
# =========================================================

def search_resume_knowledge(query: str) -> str:
    """
    搜索个人简历知识库。
    """

    results = retrieve(query, top_k=3)

    if not results:
        return "知识库中没有找到相关信息。"

    context_parts = []

    for result in results:
        context_parts.append(
            f"""
来源：{result['source']}
相似度：{result['score']:.4f}
内容：
{result['content']}
"""
        )

    return "\n".join(context_parts)


# =========================================================
# Tool 2：岗位匹配分析
# =========================================================

def analyze_job_fit(job_description: str) -> str:
    """
    根据岗位 JD 检索与岗位要求最相关的候选人经历和技能。
    """

    results = retrieve(job_description, top_k=5)

    if not results:
        return "没有找到与该岗位要求相关的个人经历。"

    context_parts = [
        "以下是根据岗位 JD 检索到的候选人相关经历："
    ]

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
【相关资料 {i}】
来源：{result['source']}
相似度：{result['score']:.4f}
内容：
{result['content']}
"""
        )

    return "\n".join(context_parts)


# =========================================================
# Tool 3：根据岗位生成面试追问素材
# =========================================================

def generate_interview_questions(job_description: str) -> str:
    """
    根据岗位 JD 检索候选人的相关经历，
    为面试官生成针对性的面试追问提供真实素材。
    """

    query = (
        f"{job_description}\n"
        "请检索与该岗位最相关的教育背景、项目经历、科研经历、"
        "技术能力、项目成果和能力短板，"
        "用于帮助面试官设计针对性的面试问题。"
    )

    results = retrieve(query, top_k=5)

    if not results:
        return "没有找到适合生成面试问题的候选人资料。"

    context_parts = [
        "以下是与该岗位相关、可供面试官设计追问的候选人真实资料："
    ]

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
【候选人资料 {i}】
来源：{result['source']}
相似度：{result['score']:.4f}
内容：
{result['content']}
"""
        )

    return "\n".join(context_parts)


# =========================================================
# 提供给大模型的 Tool 说明书
# =========================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_resume_knowledge",
            "description": (
                "搜索高颂岩的个人简历知识库。"
                "当招聘方询问候选人的教育背景、科研经历、项目经历、"
                "AI项目、技术能力、技能或其他个人履历信息时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要在个人简历知识库中检索的问题或关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "analyze_job_fit",
            "description": (
                "当招聘方提供岗位JD、岗位职责或任职要求，"
                "并希望分析高颂岩与该岗位的匹配情况时使用。"
                "该工具会根据岗位要求检索最相关的候选人项目、技能和科研经历。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_description": {
                        "type": "string",
                        "description": "招聘方提供的完整岗位JD、岗位职责或任职要求"
                    }
                },
                "required": ["job_description"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "generate_interview_questions",
            "description": (
                "当招聘方或面试官提供岗位JD，"
                "并希望根据高颂岩的真实经历生成针对性的面试问题、"
                "追问方向或考察重点时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_description": {
                        "type": "string",
                        "description": "目标岗位JD、岗位职责或任职要求"
                    }
                },
                "required": ["job_description"]
            }
        }
    }
]