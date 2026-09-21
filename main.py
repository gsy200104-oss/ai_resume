import streamlit as st
from openai import OpenAI

from agent_engine import run_agent


# =========================================================
# 页面设置
# =========================================================

st.set_page_config(
    page_title="高颂岩｜AI Resume Agent",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# API Key
# =========================================================

api_key = st.secrets["DEEPSEEK_API_KEY"]

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# =========================================================
# 页面布局
# =========================================================

left_col, right_col = st.columns([1, 2])


# =========================================================
# 左侧：候选人信息
# =========================================================

with left_col:

    st.title("高颂岩")

    st.write("**求职方向：AI 应用开发**")

    st.write("中国海洋大学")

    st.write("材料 / 生物工程交叉背景")

    st.divider()


    st.subheader("核心能力")

    st.write("🐍 Python")
    st.write("🤖 LLM API")
    st.write("📚 RAG")
    st.write("🧠 Embedding")
    st.write("🔍 向量检索")
    st.write("🛠️ Tool Calling")
    st.write("🤖 Agent")
    st.write("🌐 Streamlit")


    st.divider()


    st.subheader("项目亮点")

    st.write("• AI Resume Agent")
    st.write("• RAG 知识库问答")
    st.write("• 多工具 Tool Calling")
    st.write("• JD 岗位匹配分析")
    st.write("• 智能面试问题生成")
    st.write("• Streamlit Cloud 在线部署")


# =========================================================
# 右侧：面试官交互区
# =========================================================

with right_col:

    st.title("AI Resume Agent")

    st.write(
        "👋 你好，我是高颂岩的 AI 简历 Agent。"
    )

    st.write(
        "你可以向我询问候选人的教育背景、科研经历、"
        "AI 项目和技术能力，也可以粘贴岗位 JD，"
        "让我分析候选人与岗位的匹配情况，"
        "并生成建议重点追问的面试问题。"
    )


    st.divider()


    # =====================================================
    # 推荐问题
    # =====================================================

    st.subheader("你可以问我")

    col1, col2 = st.columns(2)


    with col1:

        if st.button("介绍一下这个候选人"):
            st.session_state.suggested_question = (
                "请从招聘方视角介绍一下候选人高颂岩。"
            )

        if st.button("他有哪些 AI 项目？"):
            st.session_state.suggested_question = (
                "高颂岩做过哪些 AI 项目？"
            )

        if st.button("他的 RAG 项目怎么实现？"):
            st.session_state.suggested_question = (
                "请详细介绍高颂岩的 RAG 项目是如何实现的。"
            )


    with col2:

        if st.button("他有哪些科研经历？"):
            st.session_state.suggested_question = (
                "高颂岩有哪些科研经历？"
            )

        if st.button("为什么转向 AI 应用开发？"):
            st.session_state.suggested_question = (
                "高颂岩为什么希望进入 AI 应用开发方向？"
            )

        if st.button("我应该重点追问什么？"):
            st.session_state.suggested_question = (
                "假设我是一名面试官，请结合高颂岩的真实经历，"
                "给我一些值得重点追问的面试问题，"
                "并说明每个问题主要想考察什么。"
            )


    # =====================================================
    # 清空聊天
    # =====================================================

    if st.button("🗑️ 清空聊天"):

        st.session_state.messages = []

        if "suggested_question" in st.session_state:
            del st.session_state.suggested_question

        st.rerun()


    st.divider()


    # =====================================================
    # 显示历史消息
    # =====================================================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("tools")
            ):

                st.caption(
                    "Agent Tools："
                    + " → ".join(message["tools"])
                )


    # =====================================================
    # 用户输入
    # =====================================================

    question = st.chat_input(
        "向 AI Resume Agent 提问..."
    )


    # 推荐问题按钮
    if "suggested_question" in st.session_state:

        question = st.session_state.suggested_question

        del st.session_state.suggested_question


    # =====================================================
    # Agent 执行
    # =====================================================

    if question:

        conversation_history = list(
            st.session_state.messages
        )


        # -----------------------------
        # 显示用户问题
        # -----------------------------

        with st.chat_message("user"):

            st.write(question)


        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # -----------------------------
        # 调用 Agent
        # -----------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Agent 正在分析任务并调用工具..."
            ):

                result = run_agent(
                    client=client,
                    user_question=question,
                    conversation_history=conversation_history
                )


            answer = result["answer"]

            tool_trace = result["tool_trace"]


            st.write(answer)


            # -------------------------
            # 展示调用过的 Tool
            # -------------------------

            tool_names = [
                item["tool"]
                for item in tool_trace
            ]


            if tool_names:

                st.caption(
                    "Agent Tools："
                    + " → ".join(tool_names)
                )


        # -----------------------------
        # 保存回答
        # -----------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "tools": tool_names
            }
        )