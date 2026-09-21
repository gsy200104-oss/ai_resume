import json

from agent_tools import (
    TOOLS,
    search_resume_knowledge,
    analyze_job_fit,
    generate_interview_questions,
)


# =========================================================
# System Prompt
# =========================================================

SYSTEM_PROMPT = """
你是高颂岩的 AI 简历 Agent，主要服务对象是招聘方、HR 和面试官。

你的任务是帮助招聘方快速了解候选人高颂岩的教育背景、科研经历、
项目经历、技术能力，以及他与目标岗位之间的匹配情况。

你可以根据用户任务自主选择一个或多个工具：

1. search_resume_knowledge
   用于查询候选人的教育背景、科研经历、项目经历、AI 项目和技术能力。

2. analyze_job_fit
   用于根据岗位 JD 分析候选人与岗位之间的相关经历和能力匹配情况。

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
# Fast Path Prompt
# =========================================================

FAST_PATH_SYSTEM_PROMPT = """
你是高颂岩的 AI 简历助手，主要服务对象是招聘方、HR 和面试官。

系统已经提前从候选人知识库中检索出了与用户问题相关的真实资料。

请直接根据提供的知识库资料回答用户问题。

要求：

1. 只能依据提供的候选人资料回答，不得编造。
2. 如果资料不足，请明确说明资料不足。
3. 优先直接回答问题，不需要解释检索过程。
4. 回答专业、客观、简洁。
5. 用户询问“有哪些”“做过哪些”等问题时，应完整列出相关内容。
"""


# =========================================================
# Tool Dispatcher
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
# Conversation History
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
# Full Agent Route Detection
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
# Overview Question Detection
# =========================================================

def is_overview_question(
    user_question: str
) -> bool:
    """
    判断是不是“概览 / 列表型”问题。

    这类问题通常不需要完整长正文，
    章节标题和少量关键信息已经足够。
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
# Compact Context Builder
# =========================================================

def build_compact_context(
    knowledge_result: str
) -> str:
    """
    把 search_resume_knowledge 返回的完整 JSON
    压缩成适合概览问题使用的短上下文。

    每个 Chunk 只保留：
    - 资料类型
    - 章节标题
    - 最多 2 条关键正文

    避免把 Top-5 的完整几千字正文
    全部发送给 DeepSeek。
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

        source_type = (
            result.get(
                "source_type",
                ""
            )
        )

        section = (
            result.get(
                "section",
                ""
            )
        )

        content = (
            result.get(
                "content",
                ""
            )
        )


        # -------------------------------------------------
        # 从正文中抽取少量有效信息
        # -------------------------------------------------

        useful_lines = []


        for line in content.splitlines():

            line = line.strip()


            # 跳过空行
            if not line:
                continue


            # 跳过 Markdown 标题，
            # 因为 section 已经单独提供。
            if line.startswith("#"):
                continue


            # 优先保留列表正文
            if line.startswith("-"):

                clean_line = (
                    line.lstrip("-")
                    .strip()
                )


                if clean_line:

                    useful_lines.append(
                        clean_line
                    )


            # 每个 Chunk 最多两条
            if len(useful_lines) >= 2:
                break


        block = (
            f"{index}. "
            f"资料类型：{source_type}\n"
            f"章节：{section}"
        )


        if useful_lines:

            block += (
                "\n关键信息："
            )


            for line in useful_lines:

                # 防止单条内容异常过长
                if len(line) > 220:

                    line = (
                        line[:220]
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
# Fast Path
# =========================================================

def run_fast_path(
    client,
    user_question,
    conversation_history=None,
):
    """
    普通简历问答快速路径。

    Overview：
        Retrieval
        → Compact Context
        → DeepSeek 1 次

    Detail：
        Retrieval
        → Full Context
        → DeepSeek 1 次
    """

    print(
        "Agent Route：Fast Path"
    )


    # =====================================================
    # 1. Retrieval
    # =====================================================

    knowledge_result = (
        search_resume_knowledge(
            user_question
        )
    )


    # =====================================================
    # 2. Context Mode
    # =====================================================

    overview_mode = (
        is_overview_question(
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
    # 3. Messages
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
    # 4. DeepSeek
    # =====================================================

    response = (
        client
        .chat
        .completions
        .create(
            model="deepseek-chat",
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
# Full Multi-Tool Agent
# =========================================================

def run_full_agent(
    client,
    user_question,
    conversation_history=None,
    max_steps=5,
):

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
                model=
                    "deepseek-chat",

                messages=
                    messages,

                tools=
                    TOOLS,

                tool_choice=
                    "auto",
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
        # Agent 已完成
        # -------------------------------------------------

        if not assistant_message.tool_calls:

            return {
                "answer":
                    assistant_message.content,

                "tool_trace":
                    tool_trace,
            }


        # -------------------------------------------------
        # 执行 Tool
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
# Public Agent Entry
# =========================================================

def run_agent(
    client,
    user_question,
    conversation_history=None,
    max_steps=5,
):

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