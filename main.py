
import textwrap

import streamlit as st
from openai import OpenAI

from agent_engine import run_agent


# =========================================================
# 1. 页面设置
# =========================================================

st.set_page_config(
    page_title="高颂岩｜AI Resume Agent",
    page_icon="🤖",
    layout="wide",
)



# =========================================================
# 2. HTML Helper
# =========================================================

def render_html(content: str):
    """
    直接渲染 HTML，避免经过 Markdown 解析器。
    """
    st.html(
        textwrap.dedent(content)
    )


# =========================================================
# 3. CSS
# =========================================================

render_html(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    /* Hero */

    .hero-card {
        padding: 2rem 2.2rem;
        border: 1px solid #e6ebf2;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #f7faff 0%,
            #ffffff 70%
        );
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 760;
        color: #172033;
        margin-bottom: 0.55rem;
    }

    .hero-subtitle {
        font-size: 1.03rem;
        color: #606b7d;
        line-height: 1.8;
        max-width: 950px;
    }

    /* Deployment Status */

    .status-row {
        margin-top: 1.1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .status-badge {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: #ecfdf3;
        border: 1px solid #abefc6;
        color: #067647;
        font-size: 0.8rem;
        font-weight: 650;
    }

    .version-badge {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: #eef4ff;
        border: 1px solid #d5e2ff;
        color: #3157c8;
        font-size: 0.8rem;
        font-weight: 650;
    }

    /* Candidate */

    .candidate-card {
        padding: 1.55rem;
        border: 1px solid #e6ebf2;
        border-radius: 18px;
        background: #ffffff;
        margin-bottom: 1rem;
    }

    .candidate-name {
        font-size: 1.8rem;
        font-weight: 760;
        color: #172033;
        margin-bottom: 0.35rem;
    }

    .candidate-role {
        font-size: 0.98rem;
        font-weight: 650;
        color: #356ae6;
        margin-bottom: 1rem;
    }

    .candidate-meta {
        color: #687386;
        line-height: 1.9;
        font-size: 0.92rem;
    }

    /* Section Card */

    .section-card {
        padding: 1.2rem 1.3rem;
        border: 1px solid #e6ebf2;
        border-radius: 17px;
        background: #ffffff;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 0.97rem;
        font-weight: 720;
        color: #1e293b;
        margin-bottom: 0.85rem;
    }

    .highlight-item {
        color: #596579;
        line-height: 1.95;
        font-size: 0.9rem;
    }

    /* Tags */

    .tag-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 0.4rem;
    }

    .tag {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #f4f6fa;
        border: 1px solid #e4e8ef;
        font-size: 0.8rem;
        color: #394457;
    }

    /* Questions */

    .question-title {
        font-size: 1.35rem;
        font-weight: 730;
        color: #1d2939;
        margin-bottom: 0.2rem;
    }

    .question-desc {
        color: #7b8596;
        margin-bottom: 1rem;
    }

    /* Buttons */

    div.stButton > button {
        min-height: 3.05rem;
        border-radius: 12px;
        border: 1px solid #e0e6ee;
        background: #ffffff;
        font-weight: 540;
    }

    div.stButton > button:hover {
        border-color: #9db3ef;
        background: #f7f9ff;
    }

    /* Chat */

    [data-testid="stChatMessage"] {
        border-radius: 15px;
    }

    .small-muted {
        color: #8b94a5;
        font-size: 0.8rem;
        line-height: 1.65;
    }

    </style>
    """
)


# =========================================================
# 4. Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# 5. DeepSeek Client
# =========================================================

try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]

except KeyError:
    st.error(
        "未检测到 DEEPSEEK_API_KEY，"
        "请在 Streamlit Secrets 中配置。"
    )
    st.stop()


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# =========================================================
# 6. 页面布局
# =========================================================

left_col, right_col = st.columns(
    [0.85, 2.15],
    gap="large",
)


# =========================================================
# 7. 左侧
# =========================================================

with left_col:

    render_html(
        """
        <div class="candidate-card">

            <div class="candidate-name">
                高颂岩
            </div>

            <div class="candidate-role">
                AI 应用开发 / 大模型应用开发
            </div>

            <div class="candidate-meta">
                🎓 中国海洋大学<br>
                🧪 材料 / 生物工程 × AI 应用开发<br>
                🤖 RAG / Agent / LLM Application
            </div>

        </div>
        """
    )


    render_html(
        """
        <div class="section-card">

            <div class="section-title">
                核心技术
            </div>

            <div class="tag-wrap">
                <span class="tag">Python</span>
                <span class="tag">DeepSeek API</span>
                <span class="tag">RAG</span>
                <span class="tag">Agent</span>
                <span class="tag">Tool Calling</span>
                <span class="tag">Chroma</span>
                <span class="tag">Embedding</span>
                <span class="tag">Contextual Chunking</span>
                <span class="tag">FastAPI</span>
                <span class="tag">Docker</span>
                <span class="tag">Streamlit</span>
            </div>

        </div>
        """
    )


    render_html(
        """
        <div class="section-card">

            <div class="section-title">
                项目亮点
            </div>

            <div class="highlight-item">
                • AI Resume Agent<br>
                • Intent-Aware Retrieval<br>
                • Contextual Chunking<br>
                • RAG Retrieval Evaluation<br>
                • Bad Case Optimization<br>
                • Multi-Tool Agent<br>
                • Agent Loop<br>
                • JD 岗位匹配分析<br>
                • 智能面试问题生成<br>
                • FastAPI 服务化<br>
                • Docker 容器化
            </div>

        </div>
        """
    )


    render_html(
        """
        <div class="section-card">

            <div class="section-title">
                工程状态
            </div>

            <div class="highlight-item">
                ✅ Agent V2 公网部署<br>
                ✅ FastAPI REST API<br>
                ✅ Docker Containerization<br>
                ✅ Git / GitHub<br>
                ✅ RAG Evaluation<br>
                ✅ Multi-Tool Agent
            </div>

            <br>

            <div class="small-muted">
                Agent V2 已支持公网访问与完整 RAG / Agent 交互
            </div>

        </div>
        """
    )


# =========================================================
# 8. 右侧
# =========================================================

with right_col:

    # -----------------------------------------------------
    # Hero
    # -----------------------------------------------------

    render_html(
        """
        <div class="hero-card">

            <div class="hero-title">
                AI Resume Agent
            </div>

            <div class="hero-subtitle">
                面向招聘场景的候选人智能 Agent。
                招聘方可以自然语言查询高颂岩的教育背景、
                科研与实习经历、AI 项目和技术能力；
                也可以输入岗位 JD，
                由 Agent 检索真实候选人经历进行岗位匹配分析，
                并生成针对性的技术面试问题。
            </div>

            <div class="status-row">

                <span class="status-badge">
                    ● Agent V2 已公网部署
                </span>

                <span class="version-badge">
                    RAG + Multi-Tool Agent
                </span>

                <span class="version-badge">
                    FastAPI + Docker
                </span>

            </div>

        </div>
        """
    )


    # =====================================================
    # 推荐问题
    # =====================================================

    render_html(
        """
        <div class="question-title">
            你可以这样问
        </div>

        <div class="question-desc">
            从候选人背景、AI 项目、RAG 架构到岗位匹配，
            选择一个问题开始了解。
        </div>
        """
    )


    q_col1, q_col2 = st.columns(2)


    with q_col1:

        if st.button(
            "👤 1 分钟介绍一下高颂岩",
            use_container_width=True,
        ):
            st.session_state.suggested_question = (
                "请从招聘方视角，用大约 1 分钟的篇幅介绍候选人高颂岩，"
                "重点说明教育背景、科研与实习经历、AI 项目、"
                "技术能力和求职方向。"
            )


        if st.button(
            "🧠 AI Resume Agent 是怎么实现的？",
            use_container_width=True,
        ):
            st.session_state.suggested_question = (
                "请详细介绍高颂岩的 AI Resume Agent 项目，"
                "重点说明项目定位、技术架构、Knowledge Base、RAG、"
                "Agent、Tool Calling、FastAPI、Docker 和工程化实现。"
            )


        if st.button(
            "📚 RAG 系统是怎么实现和优化的？",
            use_container_width=True,
        ):
            st.session_state.suggested_question = (
                "请详细介绍高颂岩的 RAG 项目是如何实现和优化的，"
                "包括 Knowledge Base、Contextual Chunking、Embedding、"
                "Chroma、Intent-Aware Retrieval、RAG Evaluation "
                "和 Bad Case Optimization。"
            )


        if st.button(
            "🛠️ Agent 如何调用多个 Tool？",
            use_container_width=True,
        ):
            st.session_state.suggested_question = (
                "请详细介绍高颂岩的 AI Resume Agent "
                "是如何实现 Tool Calling、Multi-Tool Agent、"
                "Tool Dispatcher 和 Agent Loop 的。"
            )


    with q_col2:

        if st.button(
            "💻 他有哪些 AI 项目和工程能力？",
            use_container_width=True,
        ):
            st.session_state.suggested_question = (
                "请介绍高颂岩的 AI 项目经历和 AI 工程能力，"
                "重点说明实际使用的技术栈、系统功能和工程化工作。"
            )


        if st.button(
            "🔬 他有哪些科研和实习经历？",
            use_container_width=True,
        ):
            st.session_state.suggested_question = (
                "请介绍高颂岩主要的科研经历和实习经历，"
                "并说明这些经历体现了哪些能力。"
            )


        if st.button(
            "🔄 为什么从材料方向转向 AI？",
            use_container_width=True,
        ):
            st.session_state.suggested_question = (
                "高颂岩为什么希望从材料与生物工程背景"
                "转向 AI 应用开发方向？"
                "请结合他的真实经历进行说明。"
            )


        if st.button(
            "🎯 给我 5 个技术面试问题",
            use_container_width=True,
        ):
            st.session_state.suggested_question = (
                "假设你是一名 AI 应用开发岗位的技术面试官，"
                "请根据高颂岩的真实项目经历生成 5 个"
                "值得重点追问的技术面试问题，"
                "并说明每个问题主要考察什么能力。"
            )


    # =====================================================
    # JD Matching
    # =====================================================

    with st.expander(
        "📄 粘贴岗位 JD，分析候选人匹配情况"
    ):

        jd_text = st.text_area(
            "岗位 JD",
            height=180,
            placeholder=(
                "将岗位职责、任职要求或完整 JD 粘贴到这里..."
            ),
        )

        if st.button(
            "分析岗位匹配",
            type="primary",
            use_container_width=True,
        ):

            if jd_text.strip():

                st.session_state.suggested_question = (
                    "请分析高颂岩与下面这个岗位 JD 的匹配情况。"
                    "请基于候选人的真实教育、科研、实习、项目和技能经历，"
                    "分别说明：\n\n"
                    "1. 核心匹配点\n"
                    "2. 能力证据\n"
                    "3. 可能的不足\n"
                    "4. 建议面试重点追问的问题\n\n"
                    f"岗位 JD：\n{jd_text}"
                )

            else:
                st.warning(
                    "请先粘贴岗位 JD。"
                )


    # =====================================================
    # Clear
    # =====================================================

    clear_col, _ = st.columns([1, 5])

    with clear_col:

        if st.button(
            "🗑️ 清空对话",
            use_container_width=True,
        ):

            st.session_state.messages = []

            if "suggested_question" in st.session_state:
                del st.session_state[
                    "suggested_question"
                ]

            st.rerun()


    st.divider()


    # =====================================================
    # 历史消息
    # =====================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("tools")
            ):

                st.caption(
                    "Agent Tools："
                    + " → ".join(
                        message["tools"]
                    )
                )


    # =====================================================
    # Chat Input
    # =====================================================

    question = st.chat_input(
        "向 AI Resume Agent 提问..."
    )


    if "suggested_question" in st.session_state:

        question = st.session_state[
            "suggested_question"
        ]

        del st.session_state[
            "suggested_question"
        ]


    # =====================================================
    # Agent
    # =====================================================

    if question:

        conversation_history = list(
            st.session_state.messages
        )


        with st.chat_message("user"):
            st.markdown(question)


        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )


        with st.chat_message("assistant"):

            with st.spinner(
                "Agent 正在检索候选人资料并分析..."
            ):

                result = run_agent(
                    client=client,
                    user_question=question,
                    conversation_history=(
                        conversation_history
                    ),
                )


            answer = result["answer"]

            tool_trace = result[
                "tool_trace"
            ]


            st.markdown(answer)


            tool_names = [
                item["tool"]
                for item in tool_trace
            ]


            if tool_names:

                st.caption(
                    "Agent Tools："
                    + " → ".join(
                        tool_names
                    )
                )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "tools": tool_names,
            }
        )