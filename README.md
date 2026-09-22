# AI Resume Agent

面向招聘与面试场景开发的智能简历 Agent。

项目将候选人的教育背景、科研经历、项目经历和技术能力构建为结构化知识库，并结合 RAG、向量数据库、Reranker 实验、Tool Calling 与 Agent Loop，使招聘方能够通过自然语言对话快速了解候选人信息、分析岗位匹配情况，并生成针对性的面试追问。

项目已完成 Web 应用开发、FastAPI 服务化及 Docker 容器化；最新 Agent V2 在线部署正在进行中。

---

## 在线演示

[点击体验 AI Resume Agent](https://airesume-psqiujjucpzak8egn2nccd.streamlit.app/)

> 当前在线版本主要用于 AI 简历问答展示。  
> 最新 Agent V2 / FastAPI / Docker 架构将在后续完成正式线上部署。

---

## 项目背景

传统 PDF 简历的信息密度较高，招聘方通常需要在较短时间内快速判断候选人的教育背景、项目能力和岗位匹配程度。

本项目尝试将传统静态简历升级为可交互的 AI Resume Agent。

招聘方可以直接提出问题，例如：

- 介绍一下这个候选人
- 他做过哪些 AI 项目？
- 他的 RAG 项目是如何实现的？
- 他有哪些科研经历？
- 为什么从材料 / 生物工程方向转向 AI 应用开发？
- 这个候选人与当前岗位有哪些匹配点？
- 面试时应该重点追问哪些问题？

系统根据问题自主判断任务类型，并调用不同 Tool 获取候选人的真实资料，再结合 DeepSeek 生成最终回答。

---

## 核心功能

### 1. AI 简历知识库

使用 Markdown 文件对候选人的个人信息进行模块化管理，包括：

- 教育背景
- 科研经历
- 项目经历
- AI 与软件开发能力
- 数据分析能力
- 求职方向

按照 Markdown 二级标题进行文本切块，使每个知识片段保持相对完整的语义信息。

---

### 2. Embedding 与向量数据库

使用 Sentence-Transformers 将知识片段转换为文本 Embedding。

向量数据写入 Chroma Vector Database，实现知识片段的持久化存储与 Top-K 语义检索。

知识库文件发生变化时，通过内容 Hash 自动检测更新并重新同步 Chroma 向量库。

---

### 3. RAG 检索增强生成

当前在线 RAG 流程：

```text
用户问题
→ Query Embedding
→ Chroma Vector Search
→ Top-K Candidate Retrieval
→ Relevant Context
→ DeepSeek LLM
→ Grounded Answer
```

通过检索真实个人资料作为模型上下文，提高回答的事实一致性。

当前在线版本采用向量检索与 Top-K Context 作为主要检索链路。

---

### 4. Reranker 重排序实验

为了进一步优化向量检索的排序质量，项目实验性引入 Cross-Encoder Reranker，对 Chroma 初步召回的候选知识片段进行二次相关性打分。

实验流程：

```text
Chroma Top-K Recall
→ Cross-Encoder Rerank
→ Reordered Context
→ Retrieval Evaluation
```

通过“向量召回 + 重排序”的两阶段检索方式，验证 Reranker 对高相关知识片段排序质量的改善效果。

实验结果表明，Reranker 能够改善部分 Top-1 排序错误。

但由于 CPU 环境下 Reranker 推理延迟较高，因此当前在线运行链路未启用 Reranker。

Reranker 相关代码与评估结果保留在项目中，作为 Retrieval Optimization 实验。

---

### 5. RAG Evaluation

构建独立检索测试集，对 RAG Retrieval 进行量化评估。

主要指标：

- Hit@1
- Hit@3
- Hit@5

同时记录 Top-1 未命中的 Bad Case，对以下问题进行分析：

- 文本切块粒度
- Query 与知识片段语义差异
- 多文件信息重复
- Embedding 排序偏差
- Reranker 优化效果

形成：

```text
Evaluation
→ Bad Case Analysis
→ Retrieval Optimization
→ Re-Evaluation
```

的 RAG 优化闭环。

---

### 6. Tool Calling

将不同业务能力封装为独立 Tool，并通过 Function Schema 向 LLM 描述工具用途和参数。

当前主要 Tool：

#### search_resume_knowledge

查询候选人的：

- 教育背景
- 科研经历
- 项目经历
- AI 技能
- 软件开发能力
- 其他个人履历信息

#### analyze_job_fit

接收招聘岗位 JD，并从候选人知识库中检索相关项目和技能，为模型进行岗位匹配分析提供事实依据。

#### generate_interview_questions

根据岗位 JD 和候选人的真实经历，为招聘方生成针对性的面试问题、追问方向及考察重点。

---

## Agent Loop

项目实现基础 Agent 执行循环。

流程：

```text
用户任务
→ LLM 判断任务
→ 自主选择 Tool
→ 生成 Tool Arguments
→ Python 执行 Tool
→ Tool Result 返回 LLM
→ LLM 再次判断
→ 必要时继续调用 Tool
→ 信息充分后生成最终答案
```

Agent 支持在一个任务中连续调用多个 Tool。

例如：

> 分析候选人与这个 AI 岗位的匹配情况，并告诉我应该重点追问什么。

Agent 可以自主执行：

```text
analyze_job_fit
→ generate_interview_questions
→ 综合 Tool Result
→ Final Answer
```

---

## FastAPI 服务化

将 Agent 核心逻辑封装为 FastAPI 后端服务，实现前端与 AI 业务逻辑解耦。

系统架构：

```text
Streamlit Frontend
→ FastAPI REST API
→ Agent Engine
→ Tool Dispatcher
→ RAG / Vector Database
→ DeepSeek API
```

通过 API 服务化后，Agent 能力可以进一步接入其他 Web 前端、移动端或第三方应用。

---

## 工程化设计

项目进一步增加：

- 模块化代码结构
- Agent Engine
- Tool Dispatcher
- API Key 环境配置
- Streamlit Secrets
- 异常处理
- Logging
- 请求耗时记录
- Agent Tool 调用轨迹
- Git Branch 开发管理
- RAG Evaluation
- Bad Case 分析

提高项目的可维护性与可调试性。

---

## Docker 容器化

使用 Docker 对 AI Resume Agent 进行容器化封装。

将：

- Python Runtime
- 项目代码
- Python Dependencies
- FastAPI Service

统一构建为 Docker Image，提高不同环境中的部署一致性。

---

## 技术栈

### AI / LLM

- DeepSeek API
- OpenAI SDK
- Prompt Engineering
- RAG
- Agent
- Tool Calling / Function Calling
- Agent Loop

### Retrieval

- Sentence-Transformers
- Embedding
- Chroma Vector Database
- Vector Search
- Top-K Retrieval
- Cross-Encoder Reranker（离线实验）
- Hit@K Evaluation
- Bad Case Analysis

### Backend

- Python
- FastAPI
- REST API

### Frontend

- Streamlit
- Session State

### Engineering

- Git
- GitHub
- Docker
- Logging
- Secrets Management
- Streamlit Community Cloud

---

## 项目架构

```text
                     Recruiter / Interviewer
                               │
                               ▼
                     Streamlit Frontend
                               │
                               ▼
                         FastAPI API
                               │
                               ▼
                         Agent Engine
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
         Resume Search      JD Analysis    Interview
             Tool              Tool       Question Tool
                │              │              │
                └──────────────┼──────────────┘
                               │
                               ▼
                              RAG
                               │
                               ▼
                    Sentence-Transformers
                               │
                               ▼
                    Chroma Vector Database
                               │
                               ▼
                       Top-K Retrieval
                               │
                               ▼
                       Relevant Context
                               │
                               ▼
                         DeepSeek LLM
                               │
                               ▼
                          Final Answer
```

---

## Reranker 实验链路

Reranker 当前不属于在线 Runtime，而作为独立 Retrieval Optimization 实验保留。

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

该实验用于评估 Cross-Encoder Reranking 对 Top-1 Retrieval Accuracy 的影响。