import json

from config import (
    CANDIDATE_NAME,
    LLM_MODEL,
    AGENT_MAX_STEPS,
    ENABLE_COMPACT_CONTEXT,
    COMPACT_CONTEXT_MAX_LINES,
    COMPACT_CONTEXT_MAX_CHARS,
)

from agent_tools import (
    TOOLS,
    search_resume_knowledge,
    analyze_job_fit,
    generate_interview_questions,
)


# =========================================================
# 1. System Prompt
# =========================================================

SYSTEM_PROMPT = f"""
你是{CANDIDATE_NAME}的 AI 简历 Agent，
主要服务对象是招聘方、HR 和面试官。

你的任务是帮助招聘方快速了解候选人{CANDIDATE_NAME}的教育背景、
科研经历、项目经历、技术能力，以及他与目标岗位之间的匹配情况。

你可以根据用户任务自主选择一个或多个工具：

1. search_resume_knowledge
   用于查询候选人的教育背景、科研经历、项目经历、
   AI 项目和技术能力。

2. analyze_job_fit
   用于根据岗位 JD 分析候选人与岗位之间的
   相关经历和能力匹配情况。

3. generate_interview_questions
   用于根据目标岗位和候选人的真实经历，
   为招聘方生成针对性的面试问题、追问方向和考察重点。

涉及候选人个人经历的信息必须以工具返回的真实资料为依据，
不得自行编造。

如果工具返回的信息不足，请明确说明目前资料不足，
不要为了完成回答而虚构经历。

回答风格应专业、客观、简洁，
避免夸大候选人的能力。
"""


# =========================================================
# 2. Fast Path Prompt
# =========================================================

FAST_PATH_SYSTEM_PROMPT = f"""
你是{CANDIDATE_NAME}的 AI 简历助手，
主要服务对象是招聘方、HR 和面试官。

系统已经提前从候选人知识库中检索出了
与用户问题相关的真实资料。

请直接根据提供的知识库资料回答用户问题。

要求：

1. 只能依据提供的候选人资料回答，不得编造。
2. 如果资料不足，请明确说明资料不足。
3. 优先直接回答问题，不需要解释检索过程。
4. 回答专业、客观、简洁。
5. 用户询问“有哪些”“做过哪些”等问题时，
   应尽量完整列出相关内容。
"""


# =========================================================
# 3. Tool Dispatcher
# =========================================================

def execute_tool(
    tool_name,
    tool_arguments,
):

    if tool_name == "search_resume_knowledge":

        return search_resume_knowledge(
            tool_arguments["query"]
        )

    elif tool_name == "analyze_job_fit":

        return analyze_job_fit(
            tool_arguments["job_description"]
        )

    elif tool_name == "generate_interview_questions":

        return generate_interview_questions(
            tool_arguments["job_description"]
        )

    else:

        return f"未知工具：{tool_name}"


# =========================================================
# 4. Conversation History
# =========================================================

def append_conversation_history(
    messages,
    conversation_history,
):

    if not conversation_history:
        return

    for message in conversation_history:

        if message.get("role") not in (
            "user",
            "assistant",
        ):
            continue

        content = message.get(
            "content"
        )

        if not content:
            continue

        messages.append(
            {
                "role":
                    message["role"],

                "content":
                    content,
            }
        )


# =========================================================
# 5. Full Agent Route Detection
# =========================================================

def should_use_full_agent(
    user_question: str
) -> bool:

    q = (
        user_question
        .lower()
        .replace(" ", "")
        .replace("\n", "")
    )

    complex_patterns = (
        "岗位匹配",
        "匹配度",
        "jd",
        "jobdescription",
        "职位描述",
        "岗位要求",
        "职位要求",
        "任职要求",
        "面试问题",
        "面试题",
        "面试追问",
        "追问方向",
        "考察重点",
        "生成面试",
    )

    if any(
        pattern in q
        for pattern in complex_patterns
    ):
        return True

    if (
        "岗位" in q
        and any(
            pattern in q
            for pattern in (
                "匹配",
                "适合",
                "胜任",
                "分析",
                "评价",
            )
        )
    ):
        return True

    return False


# =========================================================
# 6. Overview Question Detection
# =========================================================

def is_overview_question(
    user_question: str
) -> bool:
    """
    判断是否属于概览 / 列表型问题。
    """

    q = (
        user_question
        .lower()
        .replace(" ", "")
        .replace("\n", "")
    )

    overview_patterns = (
        "有哪些",
        "有什么",
        "都有哪些",
        "都有什么",
        "做过哪些",
        "做过什么",
        "包括哪些",
        "列出",
        "列一下",
        "介绍一下经历",
        "经历有哪些",
        "项目有哪些",
        "技能有哪些",
        "科研有哪些",
        "实习有哪些",
    )

    return any(
        pattern in q
        for pattern in overview_patterns
    )


# =========================================================
# 7. Compact Context Builder
# =========================================================

def build_compact_context(
    knowledge_result: str
) -> str:
    """
    为概览型问题构建短上下文。

    参数由 config.py 控制：

    COMPACT_CONTEXT_MAX_LINES
        每个 Chunk 最多保留几条信息

    COMPACT_CONTEXT_MAX_CHARS
        每条信息最多保留多少字符
    """

    try:

        data = json.loads(
            knowledge_result
        )

    except json.JSONDecodeError:

        return knowledge_result

    results = data.get(
        "results",
        []
    )

    if not results:

        return "没有检索到相关候选人资料。"

    compact_blocks = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        source_type = result.get(
            "source_type",
            "",
        )

        section = result.get(
            "section",
            "",
        )

        content = result.get(
            "content",
            "",
        )

        useful_lines = []

        for line in content.splitlines():

            line = line.strip()

            if not line:
                continue

            # Markdown 标题已经有 section，
            # 不重复发送。
            if line.startswith("#"):
                continue

            if line.startswith("-"):

                clean_line = (
                    line
                    .lstrip("-")
                    .strip()
                )

                if clean_line:

                    useful_lines.append(
                        clean_line
                    )

            if (
                len(useful_lines)
                >= COMPACT_CONTEXT_MAX_LINES
            ):
                break

        block = (
            f"{index}. "
            f"资料类型：{source_type}\n"
            f"章节：{section}"
        )

        if useful_lines:

            block += "\n关键信息："

            for line in useful_lines:

                if (
                    len(line)
                    > COMPACT_CONTEXT_MAX_CHARS
                ):

                    line = (
                        line[
                            :COMPACT_CONTEXT_MAX_CHARS
                        ]
                        + "..."
                    )

                block += (
                    f"\n- {line}"
                )

        compact_blocks.append(
            block
        )

    return "\n\n".join(
        compact_blocks
    )


# =========================================================
# 8. Fast Path
# =========================================================

def run_fast_path(
    client,
    user_question,
    conversation_history=None,
):
    """
    普通简历问答。

    Overview：
        Retrieval
        → Compact Context
        → DeepSeek ×1

    Detail：
        Retrieval
        → Full Context
        → DeepSeek ×1
    """

    print(
        "Agent Route：Fast Path"
    )

    # =====================================================
    # Retrieval
    # =====================================================

    knowledge_result = (
        search_resume_knowledge(
            user_question
        )
    )

    # =====================================================
    # Context Mode
    # =====================================================

    overview_mode = (
        ENABLE_COMPACT_CONTEXT
        and is_overview_question(
            user_question
        )
    )

    if overview_mode:

        print(
            "Fast Context：Compact"
        )

        context = (
            build_compact_context(
                knowledge_result
            )
        )

    else:

        print(
            "Fast Context：Full"
        )

        context = (
            knowledge_result
        )

    # =====================================================
    # Messages
    # =====================================================

    messages = [
        {
            "role":
                "system",

            "content":
                FAST_PATH_SYSTEM_PROMPT,
        }
    ]

    append_conversation_history(
        messages,
        conversation_history,
    )

    messages.append(
        {
            "role":
                "user",

            "content":
                (
                    f"用户问题：\n"
                    f"{user_question}"
                    f"\n\n"
                    f"候选人知识库资料：\n"
                    f"{context}"
                    f"\n\n"
                    f"请直接回答用户问题。"
                ),
        }
    )

    # =====================================================
    # DeepSeek ×1
    # =====================================================

    response = (
        client
        .chat
        .completions
        .create(
            model=LLM_MODEL,
            messages=messages,
        )
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return {
        "answer":
            answer,

        "tool_trace": [
            {
                "tool":
                    "search_resume_knowledge",

                "arguments": {
                    "query":
                        user_question
                },

                "context_mode":
                    (
                        "compact"
                        if overview_mode
                        else "full"
                    ),
            }
        ],
    }


# =========================================================
# 9. Full Multi-Tool Agent
# =========================================================

def run_full_agent(
    client,
    user_question,
    conversation_history=None,
    max_steps=AGENT_MAX_STEPS,
):
    """
    完整 Multi-Tool Agent。

    用于：
    - JD 分析
    - 岗位匹配
    - 面试问题
    - 多工具复杂任务
    """

    print(
        "Agent Route：Full Agent"
    )

    messages = [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        }
    ]

    append_conversation_history(
        messages,
        conversation_history,
    )

    messages.append(
        {
            "role":
                "user",

            "content":
                user_question,
        }
    )

    tool_trace = []

    # 防止相同 Tool + 参数重复执行
    tool_cache = {}

    # =====================================================
    # Agent Loop
    # =====================================================

    for step in range(
        max_steps
    ):

        response = (
            client
            .chat
            .completions
            .create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        )

        assistant_message = (
            response
            .choices[0]
            .message
        )

        messages.append(
            assistant_message.model_dump(
                exclude_none=True
            )
        )

        # -------------------------------------------------
        # 没有 Tool Call
        # -------------------------------------------------

        if not assistant_message.tool_calls:

            return {
                "answer":
                    assistant_message.content,

                "tool_trace":
                    tool_trace,
            }

        # -------------------------------------------------
        # Execute Tools
        # -------------------------------------------------

        for tool_call in (
            assistant_message.tool_calls
        ):

            tool_name = (
                tool_call
                .function
                .name
            )

            tool_arguments = (
                json.loads(
                    tool_call
                    .function
                    .arguments
                )
            )

            cache_key = (
                tool_name,
                json.dumps(
                    tool_arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )

            if cache_key in tool_cache:

                tool_result = (
                    tool_cache[
                        cache_key
                    ]
                )

            else:

                tool_result = (
                    execute_tool(
                        tool_name,
                        tool_arguments,
                    )
                )

                tool_cache[
                    cache_key
                ] = tool_result

            tool_trace.append(
                {
                    "tool":
                        tool_name,

                    "arguments":
                        tool_arguments,
                }
            )

            messages.append(
                {
                    "role":
                        "tool",

                    "tool_call_id":
                        tool_call.id,

                    "content":
                        tool_result,
                }
            )

    # =====================================================
    # 防止 Agent 无限循环
    # =====================================================

    return {
        "answer":
            (
                "Agent 已达到最大执行轮数，"
                "请尝试重新描述问题。"
            ),

        "tool_trace":
            tool_trace,
    }


# =========================================================
# 10. Public Agent Entry
# =========================================================

def run_agent(
    client,
    user_question,
    conversation_history=None,
    max_steps=AGENT_MAX_STEPS,
):
    """
    AI Resume Agent 统一入口。

    Fast Path：
        普通简历问答

    Full Agent：
        JD / 岗位匹配 / 面试等复杂任务
    """

    if should_use_full_agent(
        user_question
    ):

        return run_full_agent(
            client=client,
            user_question=user_question,
            conversation_history=conversation_history,
            max_steps=max_steps,
        )

    return run_fast_path(
        client=client,
        user_question=user_question,
        conversation_history=conversation_history,
    )