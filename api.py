"""
AI Resume Agent FastAPI Service

作用：
- 将现有 AI Resume Agent 封装为 HTTP API
- 复用 agent_engine.py
- 复用 agent_tools.py
- 复用 vector_store.py
- 不改变现有 Streamlit 应用
"""

import os
import tomllib

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from agent_engine import run_agent


# =========================================================
# 1. FastAPI App
# =========================================================

app = FastAPI(
    title="AI Resume Agent API",
    description=(
        "面向招聘场景的 AI Resume Agent API，"
        "支持候选人知识问答、岗位匹配和面试辅助。"
    ),
    version="1.0.0",
)


# =========================================================
# 2. Request Model
# =========================================================

class ChatRequest(BaseModel):
    """
    /chat 接口请求格式
    """

    question: str = Field(
        ...,
        min_length=1,
        description="用户向 AI Resume Agent 提出的问题",
    )

    conversation_history: list[dict[str, Any]] | None = Field(
        default=None,
        description="可选的历史对话",
    )


# =========================================================
# 3. DeepSeek API Key
# =========================================================

def load_deepseek_api_key() -> str:
    """
    优先从系统环境变量读取：

        DEEPSEEK_API_KEY

    如果没有，则尝试读取现有：

        .streamlit/secrets.toml

    因此暂时不需要复制或暴露 API Key。
    """

    # -----------------------------------------------------
    # A. Environment Variable
    # -----------------------------------------------------

    env_key = os.getenv(
        "DEEPSEEK_API_KEY"
    )

    if env_key:
        return env_key


    # -----------------------------------------------------
    # B. Streamlit Secrets
    # -----------------------------------------------------

    secrets_path = Path(
        ".streamlit/secrets.toml"
    )


    if secrets_path.exists():

        with secrets_path.open(
            "rb"
        ) as file:

            secrets = tomllib.load(
                file
            )


        # 常见写法：
        #
        # DEEPSEEK_API_KEY = "xxx"

        possible_keys = (
            "DEEPSEEK_API_KEY",
            "deepseek_api_key",
            "DEEPSEEK_KEY",
            "deepseek_key",
        )


        for key_name in possible_keys:

            value = secrets.get(
                key_name
            )

            if value:
                return str(value)


        # 兼容：
        #
        # [deepseek]
        # api_key = "xxx"

        deepseek_config = secrets.get(
            "deepseek"
        )


        if isinstance(
            deepseek_config,
            dict,
        ):

            for key_name in (
                "api_key",
                "key",
            ):

                value = deepseek_config.get(
                    key_name
                )

                if value:
                    return str(value)


    # -----------------------------------------------------
    # C. 没找到 Key
    # -----------------------------------------------------

    raise RuntimeError(
        "没有找到 DeepSeek API Key。"
        "请检查 DEEPSEEK_API_KEY 环境变量"
        "或 .streamlit/secrets.toml。"
    )


# =========================================================
# 4. DeepSeek Client
# =========================================================

@lru_cache(maxsize=1)
def get_deepseek_client():
    """
    Client 懒加载。

    启动 FastAPI 时不会重复创建 Client，
    第一次真正调用 Agent 时才初始化。
    """

    api_key = (
        load_deepseek_api_key()
    )


    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


    return client


# =========================================================
# 5. Root
# =========================================================

@app.get("/")
def root():

    return {
        "service":
            "AI Resume Agent API",

        "status":
            "running",

        "version":
            "1.0.0",
    }


# =========================================================
# 6. Health Check
# =========================================================

@app.get("/health")
def health():

    return {
        "status":
            "ok",

        "service":
            "AI Resume Agent",
    }


# =========================================================
# 7. Chat API
# =========================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):
    """
    调用现有 Agent Engine。

    Pipeline：

    HTTP Request
        ↓
    FastAPI
        ↓
    Agent Engine
        ↓
    Tool Calling
        ↓
    Retrieval / Chroma
        ↓
    DeepSeek
        ↓
    JSON Response
    """

    try:

        client = (
            get_deepseek_client()
        )


        result = run_agent(
            client=client,
            user_question=request.question,
            conversation_history=(
                request.conversation_history
            ),
        )


        return {
            "success": True,
            "question": request.question,
            "result": result,
        }


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc