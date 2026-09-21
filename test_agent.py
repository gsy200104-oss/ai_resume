import tomllib
from openai import OpenAI

from agent_engine import run_agent


# =========================================================
# 1. 读取 API Key
# =========================================================

with open(".streamlit/secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

api_key = secrets["DEEPSEEK_API_KEY"]


# =========================================================
# 2. 创建 DeepSeek 客户端
# =========================================================

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# =========================================================
# 3. 测试正式 Agent 模块
# =========================================================

question = """
我准备面试一个 AI 应用开发工程师岗位。

岗位要求包括：
Python、大模型 API、RAG、Embedding、向量检索、
Agent、Tool Calling、Git 和 AI 应用部署。

请先分析高颂岩与这个岗位的匹配点，
然后准备一段一分钟左右的针对性自我介绍。
"""


result = run_agent(
    client=client,
    user_question=question
)


# =========================================================
# 4. 输出结果
# =========================================================

print("\n" + "=" * 50)
print("Agent 最终回答")
print("=" * 50)

print(result["answer"])


print("\n" + "=" * 50)
print("本次 Tool 调用轨迹")
print("=" * 50)

for i, item in enumerate(result["tool_trace"], start=1):

    print(f"\n第 {i} 次调用：")
    print("Tool：", item["tool"])
    print("参数：", item["arguments"])