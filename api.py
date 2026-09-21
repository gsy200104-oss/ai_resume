"""
AI Resume Agent FastAPI Service

功能：
- 提供 AI Resume Agent HTTP API
- 复用 agent_engine.py
- 记录 API 请求日志与耗时
- 提供统一异常处理
"""

import logging
import os
import time
import tomllib
import uuid

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from agent_engine import run_agent


# =========================================================
# 1. Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "ai_resume_api"
)


# =========================================================
# 2. FastAPI App
# =========================================================

app = FastAPI(
    title="AI Resume Agent API",
    description=(
        "面向招聘场景的 AI Resume Agent API，"
        "支持候选人知识问答、岗位匹配和面试辅助。"
    ),
    version="1.1.0",
)


# =========================================================
# 3. Request Model
# =========================================================

class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="用户向 AI Resume Agent 提出的问题",
    )

    conversation_history: (
        list[dict[str, Any]] | None
    ) = Field(
        default=None,
        description="可选的历史对话",
    )


# =========================================================
# 4. DeepSeek API Key
# =========================================================

def load_deepseek_api_key() -> str:

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
                return str(
                    value
                )


        deepseek_config = (
            secrets.get(
                "deepseek"
            )
        )


        if isinstance(
            deepseek_config,
            dict,
        ):

            for key_name in (
                "api_key",
                "key",
            ):

                value = (
                    deepseek_config.get(
                        key_name
                    )
                )

                if value:
                    return str(
                        value
                    )


    raise RuntimeError(
        "没有找到 DeepSeek API Key。"
    )


# =========================================================
# 5. DeepSeek Client
# =========================================================

@lru_cache(maxsize=1)
def get_deepseek_client():

    logger.info(
        "Initializing DeepSeek client"
    )

    api_key = (
        load_deepseek_api_key()
    )


    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


    return client


# =========================================================
# 6. Request Logging Middleware
# =========================================================

@app.middleware("http")
async def request_logging(
    request: Request,
    call_next,
):

    request_id = (
        uuid.uuid4()
        .hex[:8]
    )

    start_time = (
        time.perf_counter()
    )


    logger.info(
        "[%s] %s %s START",
        request_id,
        request.method,
        request.url.path,
    )


    try:

        response = await call_next(
            request
        )


    except Exception:

        elapsed = (
            time.perf_counter()
            - start_time
        )


        logger.exception(
            "[%s] %s %s FAILED | %.3fs",
            request_id,
            request.method,
            request.url.path,
            elapsed,
        )

        raise


    elapsed = (
        time.perf_counter()
        - start_time
    )


    logger.info(
        "[%s] %s %s %s | %.3fs",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )


    response.headers[
        "X-Request-ID"
    ] = request_id


    return response


# =========================================================
# 7. Global Exception Handler
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):

    logger.exception(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
    )


    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": (
                "AI Resume Agent "
                "服务内部错误。"
            ),
        },
    )


# =========================================================
# 8. Root
# =========================================================

@app.get("/")
def root():

    return {
        "service":
            "AI Resume Agent API",

        "status":
            "running",

        "version":
            "1.1.0",
    }


# =========================================================
# 9. Health Check
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
# 10. Chat API
# =========================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    logger.info(
        "Agent question received | %s",
        request.question,
    )


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