# AI Resume Agent

面向招聘与技术面试场景开发的智能候选人 Agent。

项目将候选人的教育背景、科研经历、实习经历、AI 项目、技术能力、求职方向以及高频技术面试问题构建为结构化 Knowledge Base，并结合 **RAG、Multi-Intent Retrieval、Tool Calling、Multi-Tool Agent、FastAPI 和 Docker**，使招聘方能够通过自然语言快速了解候选人、分析岗位匹配情况，并生成针对性的技术面试问题。

目前 **Agent V2 已完成公网部署**，支持完整的 RAG / Agent 交互。

---

## 项目简介

传统 PDF 简历信息密度较高，招聘方通常需要在较短时间内判断：

- 候选人的教育背景
- 科研与实习经历
- AI 项目能力
- 技术栈
- 工程实践能力
- 岗位匹配程度
- 值得重点追问的技术问题

AI Resume Agent 将传统静态简历升级为一个可以直接对话的候选人智能 Agent。

招聘方可以直接询问：

- 请用 1 分钟介绍一下高颂岩
- 他有哪些 AI 项目和工程能力？
- AI Resume Agent 是怎么实现的？
- RAG 系统是怎么实现和优化的？
- Agent 如何调用多个 Tool？
- 他有哪些科研和实习经历？
- 为什么从材料 / 生物工程方向转向 AI？
- 为什么不用 LangChain？
- 为什么不用在线 Reranker？
- Multi-Intent Retrieval 是怎么实现的？
- 如果有 10 万份简历，这个系统应该怎么扩展？
- 根据岗位 JD 分析候选人的匹配情况
- 根据候选人项目生成技术面试问题

---

# Agent V2

Agent V2 当前已完成：

- Structured Knowledge Base
- Contextual Chunking
- Sentence-Transformers Embedding
- Chroma Persistent Vector Database
- Knowledge Auto Sync
- Intent-Aware Retrieval
- Multi-Intent Retrieval
- RAG
- Retrieval Evaluation
- Tool Calling
- Multi-Tool Agent
- Agent Loop
- Fast Path / Full Agent Routing
- JD Matching
- Interview Question Generation
- FastAPI 服务化
- Streamlit 招聘交互界面
- Pytest 自动化测试
- Docker 容器化
- Git / GitHub 版本管理
- 公网部署

Agent V2 已在 **subscrib** 完成公网部署。

---

# 1. Structured Knowledge Base

使用 Markdown 文件对候选人信息进行模块化管理。

当前 Knowledge Base：

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

当前共构建：

```text
65 Knowledge Chunks
```

主要覆盖：

- 教育背景
- 科研经历
- 实习经历
- AI 项目
- 技术能力
- 工程能力
- 求职方向
- 转 AI 动机
- 职业规划
- 个人优势
- AI 项目技术面试案例

其中：

### `career.md`

用于保存：

- 1 分钟自我介绍
- 求职方向
- 为什么从材料 / 生物工程转向 AI
- 为什么选择 AI 应用开发
- 原专业背景对 AI 开发的帮助
- 个人优势
- 职业规划

### `interview_cases.md`

用于保存：

- AI Resume Agent 项目深挖
- RAG 技术问题
- Retrieval Bad Case
- Multi-Intent Retrieval
- Reranker 工程取舍
- Fast Path
- Tool Calling
- Multi-Tool Agent
- FastAPI
- Docker
- 部署问题
- 系统扩展性
- 跨专业转 AI 面试问题

---

# 2. Contextual Chunking

项目不是简单按照固定字符数切分 Knowledge。

而是基于 Markdown Section 进行结构化 Chunking，并保留必要的上下文信息。

每个 Chunk 包含：

```text
Source
Source Type
Section Title
Content
Raw Content
Context Information
```

同时过滤：

- 空 Chunk
- 只有标题没有正文的 Chunk
- 缺乏有效语义信息的 Chunk

使每个 Chunk 即使独立进入向量库，也能保留较完整的语义上下文。

---

# 3. Embedding + Chroma Vector Database

使用：

- Sentence-Transformers
- `paraphrase-multilingual-MiniLM-L12-v2`
- Chroma Vector Database

完成知识向量化与持久化存储。

基本流程：

```text
Knowledge
↓
Contextual Chunking
↓
Embedding
↓
Chroma
↓
Persistent Vector Store
```

Knowledge 发生变化时，通过 Hash 自动检测更新。

```text
Knowledge Change
↓
SHA256 Detection
↓
Generate New Embedding
↓
Rebuild Chroma Index
```

同时使用：

```text
INDEX_VERSION
```

管理索引策略版本。

因此即使 Knowledge 原文没有变化，只要 Chunking 或 Retrieval 策略发生改变，也可以触发索引升级。

---

# 4. Intent-Aware Retrieval

早期版本对所有问题直接进行 Global Top-K Vector Search。

实际测试发现，不同资料类别之间可能发生检索竞争。

例如：

```text
用户询问 AI 项目
↓
材料科研 Chunk 也可能具有一定语义相似度
↓
占据 Top-K
↓
真正需要的 AI 项目证据被挤出
```

因此增加 Query Intent Detection。

当前主要支持：

```text
教育问题
→ 教育背景

科研问题
→ 科研经历

实习问题
→ 实习经历

AI / 软件项目问题
→ 项目经历

技术能力问题
→ 技能与能力
```

对于明确的单类别问题：

```text
Query
↓
Intent Detection
↓
Metadata Filter
↓
Vector Search
↓
Relevant Context
```

通过缩小检索范围减少跨类别噪声。

---

# 5. Multi-Intent Retrieval

在真实使用中发现，很多招聘问题并不是单一 Intent。

例如：

```text
请介绍高颂岩的科研经历和实习经历
```

这个问题同时涉及：

```text
科研经历
+
实习经历
```

旧版本 Router 只能返回：

```text
单一 Source Type
or
None
```

当检测到多个类别时，会退回：

```text
Global Vector Search
```

实际出现过：

```text
skills.md
research.md
skills.md
skills.md
skills.md
```

而：

```text
internships.md
```

完全没有进入 Top-K。

最终导致 LLM 错误回答：

```text
“当前资料中没有实习经历”
```

但实际上 Knowledge 中存在完整实习资料。

---

## Multi-Intent Retrieval 升级方案

当前流程：

```text
User Query
↓
Intent Detection
↓
Detect Multiple Intents
↓
Query Embedding
↓
每个 Intent 分别执行 Metadata Filter Search
↓
Round-Robin Merge
↓
Deduplication
↓
Context Limit
↓
DeepSeek
↓
Grounded Answer
```

关键设计：

### Query Embedding 只计算一次

即使问题包含多个 Intent：

```text
科研 + 实习 + 项目 + 技能
```

也不会重复计算四次 Embedding。

而是：

```text
Query
↓
Embedding × 1
↓
复用同一个 Query Vector
```

---

## 多类别独立召回

例如：

```text
科研 + 实习
```

系统分别执行：

```text
research.md
→ Metadata Filter Retrieval
```

以及：

```text
internships.md
→ Metadata Filter Retrieval
```

之后合并结果。

---

## Round-Robin Merge

为了避免某一个类别占满 Context：

```text
类别 A Top-1
类别 B Top-1
类别 A Top-2
类别 B Top-2
```

交替加入最终 Context。

---

## Context Limit

对于多 Intent 问题：

```text
2～3 个 Intent
→ 每类最多召回 2 个 Chunk

4 个以上 Intent
→ 每类优先召回 1 个 Chunk
```

最终 Context：

```text
最多约 6 个 Chunk
```

避免为了增加召回覆盖率导致 Prompt 过长。

---

# 6. Multi-Intent Retrieval 性能测试

对 Warm Retrieval 进行实测：

```text
Single Intent
≈ 0.0161 s

2 Intents
≈ 0.0229 s

5 Intents
≈ 0.0310 s
```

说明：

Multi-Intent Retrieval 虽然增加了多次轻量 Chroma Metadata Search，但：

```text
Query Embedding 仍只计算一次
LLM 调用次数没有增加
```

因此 Retrieval 层增加的延迟非常有限。

---

# 7. RAG Pipeline

当前在线 RAG 主链路：

```text
User Query
↓
Intent Detection
↓
Query Embedding
↓
Single Intent / Multi-Intent Retrieval
↓
Chroma Vector Database
↓
Relevant Context
↓
DeepSeek
↓
Grounded Answer
```

System Prompt 要求模型：

- 基于真实 Knowledge 回答
- 不虚构候选人经历
- 对未知信息明确说明
- 尽量提供与问题直接相关的事实依据

这样可以降低 LLM Hallucination。

---

# 8. RAG Evaluation

项目没有只通过：

```text
“感觉回答不错”
```

判断 RAG 效果。

而是建立独立 Retrieval Evaluation。

主要指标：

- Hit@1
- Hit@3
- Hit@5

基本优化流程：

```text
Evaluation
↓
Bad Case Analysis
↓
Inspect Retrieved Chunks
↓
Locate Root Cause
↓
Optimize Knowledge / Chunk / Router / Retrieval
↓
Re-Evaluation
```

---

## RAG Debug 方法

遇到错误回答时，不首先修改 Prompt。

而是检查完整链路：

```text
Query
↓
Intent
↓
Metadata Filter
↓
Retrieved Chunks
↓
Context
↓
LLM
```

典型案例：

```text
科研 + 实习问题
↓
最终回答错误
↓
检查 Retrieval Result
↓
发现 internships.md 没进入 Context
↓
定位为 Retrieval Routing 问题
↓
升级 Multi-Intent Retrieval
```

这也是项目从普通 RAG Demo 向可调试 RAG 系统演进的重要过程。

---

# 9. Cross-Encoder Reranker Experiment

项目开展过：

```text
Jina Multilingual Cross-Encoder Reranker
```

离线实验。

实验流程：

```text
Vector Retrieval
↓
Candidate Recall
↓
Cross-Encoder Reranker
↓
Reordered Results
↓
Hit@K Evaluation
```

离线实验中，Reranker 能改善部分 Bad Case 的 Top-1 排序。

但是实际测试发现：

```text
CPU Environment
↓
Cross-Encoder Inference
↓
明显增加在线延迟
```

因此当前在线 Runtime：

```text
不使用 Reranker
```

Reranker 当前仅作为：

```text
Offline Retrieval Optimization Experiment
```

保留。

这一决策主要综合考虑：

- Retrieval Accuracy
- Response Latency
- CPU Resource
- Memory Usage
- Cloud Deployment Cost
- User Experience

---

# 10. Tool Calling

不同业务能力被封装为独立 Tool。

当前主要 Tool 包括：

## `search_resume_knowledge`

用于查询候选人的：

- 教育背景
- 科研经历
- 实习经历
- AI 项目
- 技术能力
- 工程能力
- 求职信息

---

## `analyze_job_fit`

接收岗位 JD。

结合候选人真实 Knowledge 分析：

- 核心匹配点
- 能力证据
- 潜在不足
- 面试重点追问方向

---

## `generate_interview_questions`

根据：

```text
Job Description
+
Candidate Knowledge
```

生成：

- 技术面试问题
- 深挖方向
- 考察重点

---

# 11. Multi-Tool Agent

通过 Function Calling 构建 Multi-Tool Agent。

Agent Loop：

```text
User Task
↓
LLM Decision
↓
Tool Selection
↓
Generate Tool Arguments
↓
Python Tool Execution
↓
Tool Result
↓
Return Result to LLM
↓
LLM Replanning
↓
Continue / Final Answer
```

Agent 可以在一个任务中连续调用多个 Tool。

例如：

```text
分析候选人与岗位的匹配情况
并告诉我面试时应该重点追问什么
```

可以执行：

```text
analyze_job_fit
↓
generate_interview_questions
↓
Combine Tool Results
↓
Final Answer
```

同时设置最大 Agent Step，避免无限循环。

---

# 12. Fast Path / Full Agent

项目并不是所有问题都进入完整 Agent Loop。

为了降低普通问题的响应时间，增加：

```text
Fast Path
+
Full Agent Path
```

---

## Fast Path

普通候选人问答：

```text
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

例如：

```text
高颂岩做过哪些 AI 项目？
```

这种问题不需要先让 LLM 判断调用哪个 Tool。

因此直接执行 RAG。

---

## Full Agent

复杂任务，例如：

```text
JD Matching
Interview Question Generation
Multi-Step Recruitment Task
```

走完整 Agent：

```text
User Task
↓
LLM Tool Decision
↓
Tool
↓
Tool Result
↓
LLM
↓
Final Answer
```

Fast Path 的主要价值不是让 Chroma 更快，而是：

```text
减少不必要的 LLM 调用
```

---

# 13. FastAPI 服务化

Agent Engine 已封装为 FastAPI 服务。

主要接口：

```text
GET /health
POST /chat
```

并实现：

- Pydantic Request / Response
- Request ID
- Logging
- Request Latency
- Global Exception Handler
- Swagger API Docs

FastAPI 为 Agent 提供独立 API 服务能力。

---

# 14. Streamlit Frontend

Streamlit 用于招聘场景 Demo 和候选人交互界面。

主要功能：

- Candidate Profile
- AI 技术标签
- Agent V2 状态
- 推荐招聘问题
- AI 项目介绍
- RAG 技术介绍
- 科研与实习查询
- 转 AI 动机查询
- JD 输入
- 岗位匹配
- 多轮聊天
- Agent Tool Trace
- 清空对话
- 自定义 HTML / CSS UI

---

# 15. 双入口架构

当前项目支持两种主要使用方式。

## Streamlit

```text
Recruiter
↓
Streamlit
↓
Agent Engine
```

用于 Web Demo。

## FastAPI

```text
External Client
↓
FastAPI
↓
Agent Engine
```

用于 API 服务。

两种入口共享同一套 Agent / RAG 核心逻辑。

---

# 16. Docker Containerization

项目已完成 Docker 容器化。

Docker Image 包含：

- Python Runtime
- Project Code
- Dependencies
- FastAPI Service
- Sentence-Transformers Runtime

使用：

- Dockerfile
- `.dockerignore`
- Healthcheck
- Environment Variables
- Port Mapping

提高本地与部署环境之间的一致性。

---

# 17. Deployment Debugging

云部署过程中实际遇到过：

```text
/health
→ 正常

/chat
→ 502
```

排查过程：

```text
FastAPI Log
↓
Deployment Log
↓
Container Log
↓
Docker Stats
↓
Memory Analysis
```

最终确认部分低内存云环境无法稳定支撑当前：

```text
Sentence-Transformers
+
Agent
+
FastAPI
```

运行。

这一过程用于实践：

- Container Debugging
- Memory Profiling
- Cold Start Analysis
- Cloud Runtime Difference
- Model Resource Usage
- Deployment Troubleshooting

Agent V2 后续已在 **subscrib** 完成公网部署。

---

# 18. Lazy Loading

Embedding Model 使用 Lazy Loading。

启动 Streamlit / FastAPI 时不会立刻加载 MiniLM。

第一次真正需要 Retrieval 时：

```text
Load Embedding Model
↓
Keep Model in Memory
↓
Reuse for Later Requests
```

项目也尝试过：

```text
Embedding Warm-Up
```

即服务启动时提前加载模型。

但实际测试发现：

- 网站启动时间增加
- 当前环境下整体体验没有改善

因此最终保留：

```text
Lazy Loading
```

作为当前方案。

---

# 19. Automated Testing

项目建立了两层自动化测试。

---

## Routing Unit Tests

当前：

```text
15 passed
```

主要覆盖：

- Fast Path
- Full Agent
- Overview Query
- Detail Query
- Single Intent
- Multi-Intent
- AI Project Query
- RAG Query
- Agent / Tool Query
- Candidate Overview Query

---

## Retrieval Integration Tests

当前：

```text
3 passed
```

验证真实 Chroma Retrieval。

### Research + Internship

要求：

```text
research.md
+
internships.md
```

必须同时进入结果。

### AI Project + Skill

要求：

```text
projects.md
+
skills.md
```

必须同时进入结果。

### Candidate Overview

要求同时覆盖：

```text
education.md
research.md
internships.md
projects.md
skills.md
```

---

## 当前完整相关测试

```text
18 passed
```

---

# 20. 技术栈

## AI / LLM

- DeepSeek API
- OpenAI SDK
- Prompt Engineering
- RAG
- Grounded Generation
- Tool Calling
- Function Calling
- Multi-Tool Agent
- Agent Loop

---

## Retrieval

- Sentence-Transformers
- Multilingual MiniLM
- Embedding
- Chroma Vector Database
- Contextual Chunking
- Vector Search
- Top-K Retrieval
- Metadata Filtering
- Intent-Aware Retrieval
- Multi-Intent Retrieval
- Hit@K Evaluation
- Bad Case Analysis
- Cross-Encoder Reranker（Offline Experiment）

---

## Backend

- Python
- FastAPI
- REST API
- Pydantic

---

## Frontend

- Streamlit
- Session State
- HTML
- CSS

---

## Engineering

- Git
- GitHub
- Docker
- Pytest
- Logging
- Request ID
- Exception Handling
- Secrets Management
- Cloud Deployment
- Performance Testing
- Debugging

---

# 21. 系统架构

```text
                    Recruiter / Interviewer
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
      Streamlit Frontend                FastAPI API
              │                               │
              └───────────────┬───────────────┘
                              ▼
                         Agent Engine
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
          Fast Path                      Full Agent
              │                               │
              │                        LLM Tool Decision
              │                               │
              │                          Tool Dispatcher
              │                               │
              └───────────────┬───────────────┘
                              ▼
                       Retrieval Router
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
        Single Intent                  Multi Intent
              │                               │
       Metadata Filter             Multi Metadata Search
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    Sentence-Transformers
                              │
                              ▼
                    Chroma Vector Database
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

# 22. Reranker 离线实验链路

Reranker 当前不属于在线 Runtime。

实验链路：

```text
Query
↓
Chroma Retrieval
↓
Candidate Chunks
↓
Cross-Encoder Reranker
↓
Reordered Results
↓
Hit@K Evaluation
```

主要用于研究：

```text
Cross-Encoder Reranking
```

对：

```text
Top-1 Retrieval Accuracy
```

的影响。

---

# 23. 项目结构

```text
ai_resume/
│
├── main.py
├── api.py
├── agent_engine.py
├── agent_tools.py
├── vector_store.py
├── config.py
├── reranker.py
├── section_eval.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
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

# 24. 本地运行

## 创建 / 激活 Python 环境

进入项目目录：

```bash
cd ai_resume
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 配置 DeepSeek API Key

Streamlit 本地运行时，在：

```text
.streamlit/secrets.toml
```

配置：

```toml
DEEPSEEK_API_KEY = "your_api_key"
```

注意：

```text
不要将真实 API Key 提交到 GitHub
```

---

## 启动 Streamlit

```bash
python -m streamlit run main.py
```

本地访问：

```text
http://localhost:8501
```

---

## 启动 FastAPI

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Health Check：

```text
http://127.0.0.1:8000/health
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

---

# 25. 运行测试

Routing Tests：

```bash
python -m pytest tests/test_routing.py -v
```

Retrieval Integration Tests：

```bash
python -m pytest tests/test_retrieval.py -v
```

全部执行：

```bash
python -m pytest tests/test_routing.py tests/test_retrieval.py -v
```

当前结果：

```text
18 passed
```

---

# 26. 当前项目状态

- [x] Structured Knowledge Base
- [x] Contextual Chunking
- [x] Sentence-Transformers Embedding
- [x] Chroma Persistent Vector Database
- [x] Knowledge Auto Sync
- [x] Intent-Aware Retrieval
- [x] Multi-Intent Retrieval
- [x] RAG
- [x] RAG Evaluation
- [x] Bad Case Analysis
- [x] Reranker Offline Experiment
- [x] Tool Calling
- [x] Multi-Tool Agent
- [x] Agent Loop
- [x] Fast Path / Full Agent
- [x] JD Matching
- [x] Interview Question Generation
- [x] Career Knowledge
- [x] Interview Case Knowledge
- [x] FastAPI
- [x] Streamlit
- [x] Pytest
- [x] Docker
- [x] Git / GitHub
- [x] Agent V2 Public Deployment

---

# 27. 项目目标

本项目的目标不是简单调用一次 LLM API，而是实践一个较完整的大模型应用开发流程：

```text
Product Requirement
↓
Knowledge Base
↓
Contextual Chunking
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
Bad Case Analysis
↓
Agent
↓
Tool Calling
↓
API Service
↓
Testing
↓
Docker
↓
Deployment
↓
Performance Optimization
```

通过真实 Bad Case、Retrieval Evaluation、自动化测试和部署问题不断迭代系统。

---

# 28. 项目特点

这个项目重点实践的不是某一个单独框架，而是：

> 从真实业务问题出发，完成 Knowledge、RAG、Agent、Evaluation、Backend、Docker、Testing 和 Deployment 的完整 AI 应用开发闭环。

同时通过实际错误案例不断分析：

```text
问题到底来自：
LLM？
Prompt？
Knowledge？
Chunking？
Embedding？
Retrieval？
Router？
Context？
Deployment？
```

并通过最小验证和自动测试完成迭代。

这也是 AI Resume Agent 从早期 RAG Demo 逐步升级为 Agent V2 的主要过程。