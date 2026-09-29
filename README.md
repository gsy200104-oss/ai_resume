# AI Resume Agent

> 面向招聘与技术面试场景的候选人智能 Agent  
> 基于 **RAG + Multi-Intent Retrieval + Tool Calling + FastAPI + Docker** 构建。

AI Resume Agent 将传统静态简历升级为可交互的候选人知识系统。

招聘方可以通过自然语言快速了解候选人的教育、科研、实习、AI 项目与技术能力，也可以输入岗位 JD，由 Agent 分析岗位匹配情况并生成针对性的技术面试问题。

**Agent V2 已完成公网部署。**

---

## ✨ Core Features

- 🤖 RAG 候选人知识问答
- 🔍 Intent-Aware Retrieval
- 🧠 Multi-Intent Retrieval
- 🛠️ Tool Calling / Multi-Tool Agent
- 📄 JD 岗位匹配分析
- 🎯 技术面试问题生成
- 📊 RAG Retrieval Evaluation
- ⚡ Fast Path / Full Agent Routing
- 🌐 FastAPI REST API
- 🐳 Docker Containerization
- 🧪 Pytest Automated Tests

---

## 🎯 What Can It Do?

招聘方可以直接询问：

```text
请用 1 分钟介绍一下高颂岩

他有哪些 AI 项目和工程能力？

AI Resume Agent 是怎么实现的？

RAG 系统是怎么实现和优化的？

他有哪些科研和实习经历？

为什么从材料 / 生物工程方向转向 AI？

Agent 如何调用多个 Tool？

为什么不用在线 Reranker？

如果有 10 万份简历，这个系统应该怎么扩展？
```

也可以粘贴岗位 JD，让 Agent：

```text
分析岗位匹配
→ 找出能力证据
→ 说明可能的不足
→ 生成针对性面试问题
```

---

# 🏗️ Architecture

```text
                 Recruiter / Interviewer
                          │
                          ▼
                 Streamlit Frontend
                          │
                          ▼
                     Agent Engine
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
          Fast Path              Full Agent
              │                       │
              │                 Tool Calling
              │                       │
              └───────────┬───────────┘
                          ▼
                  Retrieval Router
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        Single Intent           Multi Intent
              │                       │
       Metadata Filter      Multi Metadata Search
              │                       │
              └───────────┬───────────┘
                          ▼
                Sentence-Transformers
                          │
                          ▼
                 Chroma Vector DB
                          │
                          ▼
                  Relevant Context
                          │
                          ▼
                    DeepSeek API
                          │
                          ▼
                   Grounded Answer
```

---

# 🧠 Knowledge Base

候选人信息使用 Markdown 结构化管理：

```text
knowledge/
├── education.md
├── internships.md
├── research.md
├── projects.md
├── skills.md
├── career.md
└── interview_cases.md
```

当前知识库包含约：

```text
65 Knowledge Chunks
```

覆盖：

```text
教育背景
科研经历
实习经历
AI 项目
技术能力
求职方向
职业动机
项目技术面试案例
```

Knowledge 内容发生变化时，系统通过 Hash 自动检测，并重新同步 Chroma Vector Database。

---

# 🔍 RAG Retrieval

基础链路：

```text
User Query
↓
Query Embedding
↓
Chroma Vector Search
↓
Relevant Context
↓
DeepSeek
↓
Grounded Answer
```

Embedding Model：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Vector Database：

```text
Chroma
```

---

# 🧠 Multi-Intent Retrieval

普通 RAG Top-K 在多类别问题中可能出现检索竞争。

例如：

```text
请介绍高颂岩的科研经历和实习经历
```

旧方案可能只返回：

```text
skills.md
research.md
skills.md
...
```

导致 `internships.md` 没有进入最终 Context。

因此项目实现了 Multi-Intent Retrieval：

```text
Query
↓
Intent Detection
↓
Detect Multiple Intents
↓
Query Embedding × 1
↓
Metadata Search for Each Intent
↓
Round-Robin Merge
↓
Deduplication
↓
Context Limit
↓
LLM
```

例如：

```text
科研 + 实习
→ research.md + internships.md
```

```text
AI 项目 + 技术能力
→ projects.md + skills.md
```

Query Embedding 只计算一次，因此不会因为 Intent 数量增加而重复执行 Embedding。

### Warm Retrieval Benchmark

```text
Single Intent   ≈ 0.0161 s
2 Intents       ≈ 0.0229 s
5 Intents       ≈ 0.0310 s
```

---

# ⚡ Fast Path / Full Agent

并不是所有问题都需要完整 Agent Loop。

普通候选人问答走：

```text
Fast Path

Question
↓
Retrieval
↓
Context
↓
DeepSeek
↓
Answer
```

复杂任务，例如：

```text
JD Matching
Interview Question Generation
Multi-Step Recruitment Task
```

进入：

```text
Full Agent

User Task
↓
LLM Decision
↓
Tool Calling
↓
Tool Result
↓
LLM
↓
Final Answer
```

这样可以减少普通问题中不必要的 LLM Tool Decision 调用。

---

# 🛠️ Tool Calling

当前主要 Tool：

### `search_resume_knowledge`

检索候选人的教育、科研、实习、AI 项目和技术能力。

### `analyze_job_fit`

根据岗位 JD 检索候选人相关经历，分析岗位匹配情况。

### `generate_interview_questions`

结合 JD 和候选人的真实项目经历生成技术面试问题与追问方向。

Agent 支持在一个任务中连续调用多个 Tool：

```text
JD Analysis
↓
analyze_job_fit
↓
generate_interview_questions
↓
Final Answer
```

---

# 📊 RAG Evaluation

项目建立了独立 Retrieval Evaluation，而不是只根据最终回答主观判断效果。

主要指标：

```text
Hit@1
Hit@3
Hit@5
```

优化流程：

```text
Evaluation
↓
Bad Case Analysis
↓
Locate Root Cause
↓
Retrieval Optimization
↓
Re-Evaluation
```

真实 Bad Case 也直接推动了 Multi-Intent Retrieval 的实现。

---

# 🧪 Reranker Experiment

项目测试过：

```text
Jina Multilingual Cross-Encoder Reranker
```

离线流程：

```text
Vector Retrieval
↓
Candidate Chunks
↓
Cross-Encoder Reranker
↓
Reordered Results
↓
Hit@K Evaluation
```

实验表明 Reranker 能改善部分 Top-1 排序问题。

但 CPU 环境下在线推理延迟较高，因此：

```text
Online Runtime
→ 不使用 Reranker
```

Reranker 当前作为：

```text
Offline Retrieval Optimization Experiment
```

保留。

---

# 🌐 FastAPI

Agent 核心逻辑已经封装为 FastAPI 服务。

主要接口：

```text
GET  /health
POST /chat
```

同时实现：

```text
Pydantic Request / Response
Request ID
Logging
Latency Tracking
Global Exception Handler
Swagger Docs
```

本地启动：

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

---

# 🐳 Docker

项目已经完成 Docker 容器化，包括：

```text
Python Runtime
Project Code
Dependencies
FastAPI Service
Embedding Runtime
```

使用：

```text
Dockerfile
.dockerignore
Healthcheck
Environment Variables
Port Mapping
```

保证本地和云部署环境尽可能一致。

---

# 🧪 Automated Tests

当前包括两类自动化测试。

### Routing Tests

```text
15 passed
```

覆盖：

```text
Fast Path
Full Agent
Single Intent
Multi Intent
Overview Query
RAG Query
Agent / Tool Query
```

### Retrieval Integration Tests

```text
3 passed
```

验证：

```text
Research + Internship
→ research.md + internships.md
```

```text
AI Project + Skill
→ projects.md + skills.md
```

```text
Candidate Overview
→ education + research + internships + projects + skills
```

当前相关测试：

```text
18 passed
```

运行：

```bash
python -m pytest tests/test_routing.py tests/test_retrieval.py -v
```

---

# 🧰 Tech Stack

### AI / LLM

```text
DeepSeek API
OpenAI SDK
RAG
Prompt Engineering
Tool Calling
Function Calling
Multi-Tool Agent
```

### Retrieval

```text
Sentence-Transformers
Multilingual MiniLM
Chroma
Contextual Chunking
Metadata Filtering
Intent-Aware Retrieval
Multi-Intent Retrieval
Hit@K Evaluation
Cross-Encoder Reranker
```

### Backend

```text
Python
FastAPI
Pydantic
REST API
```

### Frontend

```text
Streamlit
Session State
HTML / CSS
```

### Engineering

```text
Git / GitHub
Docker
Pytest
Logging
Secrets Management
Cloud Deployment
Performance Testing
```

---

# 📁 Project Structure

```text
ai_resume/
│
├── main.py
├── api.py
├── agent_engine.py
├── agent_tools.py
├── vector_store.py
├── config.py
├── evaluation/reranker.py
├── evaluation/rag_eval.py
├── evaluation/section_eval.py
│
├── Dockerfile
├── requirements.txt
│
├── knowledge/
│   ├── education.md
│   ├── internships.md
│   ├── research.md
│   ├── projects.md
│   ├── skills.md
│   ├── career.md
│   └── interview_cases.md
│
├── tests/
│   ├── test_routing.py
│   └── test_retrieval.py
│
└── README.md
```

---

# 🚀 Local Run

安装依赖：

```bash
pip install -r requirements.txt
```

配置：

```text
.streamlit/secrets.toml
```

```toml
DEEPSEEK_API_KEY = "your_api_key"
```

启动 Streamlit：

```bash
python -m streamlit run main.py
```

访问：

```text
http://localhost:8501
```

---

# ✅ Current Status

```text
✅ Structured Knowledge Base
✅ Contextual Chunking
✅ Chroma Vector Database
✅ Intent-Aware Retrieval
✅ Multi-Intent Retrieval
✅ RAG
✅ Retrieval Evaluation
✅ Tool Calling
✅ Multi-Tool Agent
✅ Fast Path / Full Agent
✅ JD Matching
✅ Interview Question Generation
✅ FastAPI
✅ Streamlit
✅ Pytest
✅ Docker
✅ Git / GitHub
✅ Public Deployment
```

---

# 🎯 Project Goal

这个项目的目标不是简单调用一次 LLM API，而是实践一个完整的大模型应用开发流程：

```text
Product Requirement
↓
Knowledge Base
↓
Embedding
↓
Vector Database
↓
Retrieval
↓
RAG
↓
Evaluation
↓
Agent
↓
Tool Calling
↓
FastAPI
↓
Testing
↓
Docker
↓
Deployment
↓
Optimization
```

核心实践重点是：

> 通过真实 Bad Case、Retrieval Evaluation、自动化测试和部署问题，不断定位系统瓶颈，并完成可验证的工程优化。