import json

from agent_tools import (
    TOOLS,
    search_resume_knowledge,
    analyze_job_fit,
    generate_interview_questions
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
   用于查询候选人的教育背景、科研经历、项目经历、AI项目和技术能力。

2. analyze_job_fit
   用于根据岗位 JD 分析候选人与岗位之间的相关经历和能力匹配情况。

3. generate_interview_questions
   用于根据目标岗位和候选人的真实经历，
   为招聘方生成针对性的面试问题、追问方向和考察重点。

你可以根据任务需要连续调用多个工具。

涉及候选人个人经历的信息必须以工具返回的真实资料为依据，
不得自行编造。

如果工具返回的信息不足，请明确说明目前资料不足，
不要为了完成回答而虚构经历。

当已经获得足够的信息后，请停止调用工具并直接生成最终回答。

回答风格应专业、客观、简洁，
避免夸大候选人的能力。
"""


# =========================================================
# Tool Dispatcher
# =========================================================

def execute_tool(tool_name, tool_arguments):
    """
    根据 Agent 选择的工具名称，
    执行对应的 Python 函数。
    """

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
# Agent
# =========================================================

def run_agent(
    client,
    user_question,
    conversation_history=None,
    max_steps=5
):
    """
    运行 AI Resume Agent。

    参数：
    client：
        OpenAI / DeepSeek 客户端

    user_question：
        用户当前问题

    conversation_history：
        历史对话，可选

    max_steps：
        Agent 最大执行轮数

    返回：
        answer：最终回答
        tool_trace：本次调用过的 Tool
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


    # -----------------------------------------------------
    # 加入历史对话
    # -----------------------------------------------------

    if conversation_history:

        for message in conversation_history:

            if message["role"] in ["user", "assistant"]:

                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                )


    # -----------------------------------------------------
    # 加入当前用户问题
    # -----------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    tool_trace = []


    # =====================================================
    # Agent Loop
    # =====================================================

    for step in range(max_steps):

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message


        messages.append(
            assistant_message.model_dump(
                exclude_none=True
            )
        )


        # -------------------------------------------------
        # 没有 Tool Call
        # Agent 完成任务
        # -------------------------------------------------

        if not assistant_message.tool_calls:

            return {
                "answer": assistant_message.content,
                "tool_trace": tool_trace
            }


        # -------------------------------------------------
        # 执行这一轮所有 Tool
        # -------------------------------------------------

        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            tool_arguments = json.loads(
                tool_call.function.arguments
            )


            tool_result = execute_tool(
                tool_name,
                tool_arguments
            )


            # 保存 Tool 调用轨迹
            tool_trace.append(
                {
                    "tool": tool_name,
                    "arguments": tool_arguments
                }
            )


            # Tool Result 返回给模型
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                }
            )


    # =====================================================
    # 防止 Agent 无限循环
    # =====================================================

    return {
        "answer": "Agent 已达到最大执行轮数，请尝试重新描述问题。",
        "tool_trace": tool_trace
    }