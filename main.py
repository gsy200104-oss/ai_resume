import streamlit as st
from openai import OpenAI

from embedding_retriever import retrieve


# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="高颂岩｜AI Interactive Resume",
    page_icon="🤖",
    layout="wide"
)


# =========================
# 初始化聊天记录
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# API Key
# 现在先用于本地测试
# 后面部署时会改成 Secrets
# =========================
api_key = st.secrets["DEEPSEEK_API_KEY"]


# =========================
# 页面两栏布局
# =========================
left_col, right_col = st.columns([1, 2])


# =========================
# 左侧：个人简历信息
# =========================
with left_col:
    st.title("高颂岩")

    st.write("**求职方向：AI 应用开发**")
    st.write("中国海洋大学")
    st.write("材料 / 生物工程相关背景")

    st.divider()

    st.subheader("核心能力")

    st.write("🐍 Python")
    st.write("🌐 Streamlit")
    st.write("🤖 DeepSeek API")
    st.write("📚 RAG")
    st.write("📊 数据分析")

    st.divider()

    st.subheader("项目亮点")

    st.write("• AI 简历知识库助手")
    st.write("• 科研与材料项目经历")
    st.write("• 数据分析与实验设计")
    st.write("• 项目管理与团队协作")


# =========================
# 右侧：AI 简历助手
# =========================
with right_col:
    st.title("AI Interactive Resume")

    st.write(
        "👋 你好，我是高颂岩的 AI 简历助手。"
    )

    st.write(
        "你可以直接向我提问，了解教育背景、科研经历、"
        "项目经验和技术能力。"
    )

    st.divider()

    # =========================
    # 推荐问题
    # =========================
    st.subheader("你可以问我")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "介绍一下你自己",
            use_container_width=True
        ):
            st.session_state.suggested_question = (
                "介绍一下你自己"
            )

        if st.button(
            "你有哪些科研经历？",
            use_container_width=True
        ):
            st.session_state.suggested_question = (
                "你有哪些科研经历？"
            )

        if st.button(
            "你的 Python 能力怎么样？",
            use_container_width=True
        ):
            st.session_state.suggested_question = (
                "你的 Python 能力怎么样？"
            )

    with col2:
        if st.button(
            "你做过哪些项目？",
            use_container_width=True
        ):
            st.session_state.suggested_question = (
                "你做过哪些项目？"
            )

        if st.button(
            "你在项目中主要负责什么？",
            use_container_width=True
        ):
            st.session_state.suggested_question = (
                "你在项目中主要负责什么？"
            )

        if st.button(
            "为什么想进入 AI 应用开发？",
            use_container_width=True
        ):
            st.session_state.suggested_question = (
                "为什么想进入 AI 应用开发？"
            )

    # =========================
    # 清空聊天
    # =========================
    if st.button("🗑️ 清空聊天"):
        st.session_state.messages = []

        if "suggested_question" in st.session_state:
            del st.session_state.suggested_question

        st.rerun()

    st.divider()

    # =========================
    # 显示历史聊天
    # =========================
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if (
                message["role"] == "assistant"
                and "source" in message
            ):
                st.caption(
                    f"资料来源：{message['source']}"
                )

    # =========================
    # 用户输入
    # =========================
    question = st.chat_input(
        "你想了解我的什么？"
    )

    # 如果用户点击了推荐问题按钮
    if "suggested_question" in st.session_state:
        question = st.session_state.suggested_question
        del st.session_state.suggested_question

    # =========================
    # 开始处理用户问题
    # =========================
    if question:

        # -------------------------
        # 显示用户问题
        # -------------------------
        with st.chat_message("user"):
            st.write(question)

        # -------------------------
        # 保存用户问题
        # -------------------------
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # -------------------------
        # RAG：检索 Top-K 资料
        # -------------------------
        retrieved_documents = retrieve(
            question,
            top_k=3
        )

        # -------------------------
        # 合并检索结果
        # -------------------------
        if retrieved_documents:

            context_parts = []

            for document in retrieved_documents:
                context_parts.append(
                    f"""
来源：{document["source"]}

{document["content"]}
"""
                )

            context = "\n\n".join(
                context_parts
            )

            # 去除重复来源
            sources = []

            for document in retrieved_documents:
                if document["source"] not in sources:
                    sources.append(
                        document["source"]
                    )

            source = "、".join(sources)

        else:
            context = (
                "没有检索到相关资料。"
            )
            source = "无"

        # -------------------------
        # 检查 API Key
        # -------------------------
        if not api_key:

            st.warning(
                "请先输入 DeepSeek API Key。"
            )

        else:

            # -------------------------
            # 创建 DeepSeek 客户端
            # -------------------------
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

            # -------------------------
            # System Prompt
            # -------------------------
            messages = [
                {
                    "role": "system",
                    "content": f"""
你是高颂岩的 AI 简历助手。

你的任务是帮助面试官了解高颂岩的教育背景、
科研经历、项目经历、技术能力以及求职方向。

下面是系统根据面试官当前问题，
从高颂岩个人知识库中检索到的相关资料：

========================

{context}

========================

请严格遵守以下规则：

1. 只能根据上面检索到的资料回答。

2. 不得编造高颂岩没有经历过的事情。

3. 不得虚构公司、项目、数据、技能、
学历、奖项或工作经历。

4. 如果检索到的资料不足以回答问题，
请明确说明：
“目前检索到的资料中没有相关信息。”

5. 不要根据常识猜测或补充资料中不存在的信息。

6. 如果资料中明确存在答案，
请直接、自然地回答，不要过度保守。

7. 回答应当专业、自然、简洁，
像候选人在真实面试中回答问题。

8. 如果问题涉及多个方面，
可以根据检索到的多条资料进行综合回答。
"""
                }
            ]

            # -------------------------
            # 加入历史对话
            # -------------------------
            messages.extend(
                st.session_state.messages
            )

            # -------------------------
            # 调用 DeepSeek
            # -------------------------
            response = (
                client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages
                )
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

            # -------------------------
            # 显示 AI 回答
            # -------------------------
            with st.chat_message(
                "assistant"
            ):
                st.write(answer)

                st.caption(
                    f"资料来源：{source}"
                )

            # -------------------------
            # 保存 AI 回答
            # -------------------------
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "source": source
                }
            )