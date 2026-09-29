# AI 项目技术面试案例

## AI Resume Agent 项目定位

AI Resume Agent 是面向招聘场景开发的候选人智能问答系统。

目标用户主要是 HR 和技术面试官，可以通过自然语言查询候选人的教育背景、科研经历、实习、AI 项目和技能，也可以输入岗位 JD 进行匹配分析，并生成针对性的技术面试问题。

系统核心技术包括：
Python、DeepSeek API、Sentence-Transformers、Chroma、RAG、Contextual Chunking、Intent-Aware Retrieval、Multi-Intent Retrieval、Tool Calling、Multi-Tool Agent、FastAPI、Streamlit 和 Docker。


## 这个项目是不是独立完成的？

AI Resume Agent 的核心设计和开发由本人独立完成。

主要工作包括：
- Knowledge Base 结构设计
- Chunking 与 Contextual Chunking
- Embedding 与 Chroma 向量库
- RAG 检索
- Intent Routing
- Multi-Intent Retrieval
- Retrieval Evaluation
- Tool Calling
- Multi-Tool Agent
- Agent Loop
- FastAPI 服务化
- Streamlit 前端
- Docker 容器化
- 自动化测试
- 公网部署
- 性能与 Bad Case 优化


## RAG 是怎么实现的？

系统首先将教育、科研、实习、项目和技能整理成结构化 Markdown Knowledge Base。

随后通过 Contextual Chunking 将知识拆分成带有上下文信息的 Chunk，并使用 Sentence-Transformers 的 multilingual MiniLM 生成 Embedding，存入 Chroma 向量数据库。

用户提问后：
1. 对 Query 进行 Intent Detection；
2. 计算 Query Embedding；
3. 根据 Intent 使用 Metadata Filtering 或 Multi-Intent Retrieval；
4. 从 Chroma 召回相关 Chunk；
5. 构建 Context；
6. 将检索内容与问题一起发送给 DeepSeek；
7. 通过 System Prompt 要求模型严格依据候选人资料回答，降低幻觉。


## 为什么要做 Contextual Chunking？

最初普通 Chunking 会出现一个问题：

Chunk 中可能只有具体内容，但缺少它属于哪个项目、哪个经历或哪个主题的信息。

这样 Embedding 时容易丢失语义上下文。

因此我在 Chunk 中保留 Section Title 和必要的上级上下文，使每个 Chunk 即使独立存在也能表达“这段内容属于什么”。

这样可以提高检索准确性，同时减少一些标题为空或上下文不足的 Bad Case。


## 为什么要做 Intent-Aware Retrieval？

单纯做全库 Top-K 时，我发现不同类别之间会互相竞争。

例如用户询问 AI 项目时，材料科研相关 Chunk 可能因为部分语义相似被召回。

因此我增加了 Query Intent Detection，将问题识别为：
- 教育背景
- 科研经历
- 实习经历
- 项目经历
- 技能与能力

如果问题明确属于单一类别，就先使用 Metadata Filter 限制检索范围，再进行向量检索。

这样可以减少跨类别噪声。


## 为什么后来又做 Multi-Intent Retrieval？

实际测试中发现，单一 Intent Router 还有一个问题。

例如用户问：

“请介绍高颂岩的科研经历和实习经历。”

这个问题同时包含两个类别。

旧逻辑发现不是单一类别后，会退回全库 Top-K。

结果 Top-5 中可能被 skills.md 和 research.md 占满，导致 internships.md 完全没有召回，最终 LLM 错误回答“没有实习资料”。

因此我升级为 Multi-Intent Retrieval：

1. Router 保留所有命中的 Intent；
2. Query Embedding 只计算一次；
3. 每个 Intent 分别进行 Metadata Filter 检索；
4. 使用 Round-Robin 合并结果；
5. 去重；
6. 限制最终 Chunk 数量。

这样既保证多类别覆盖，又不会明显增加 Context。


## Multi-Intent Retrieval 会不会很慢？

我专门进行了 Warm Retrieval 测试：

- 单意图：约 0.0161 秒
- 双意图：约 0.0229 秒
- 五意图：约 0.0310 秒

Multi-Intent 增加的主要只是几次轻量 Chroma Metadata 查询。

Query Embedding 仍然只计算一次，也没有增加额外 LLM 调用，因此总体延迟增加非常小。


## RAG 如何评估？

我没有只通过“感觉回答好不好”判断 RAG 效果，而是建立了 Retrieval Evaluation。

主要指标包括：
- Hit@1
- Hit@3
- Hit@5

优化过程为：

Evaluation
→ 找出 Bad Case
→ 分析错误原因
→ 调整 Knowledge Structure / Chunking / Metadata / Retrieval
→ 再次评测。

通过这种方式量化判断检索优化是否真正有效。


## Reranker 做过什么实验？

我测试过 Jina multilingual Cross-Encoder Reranker。

流程为：
Vector Retrieval
→ Candidate Recall
→ Cross-Encoder Rerank
→ Final Ranking。

离线实验中，Reranker 对部分 Bad Case 的 Top-1 排序效果有明显改善。

但实际测试发现 CPU 在线推理延迟较高，因此最终没有将 Reranker 放入线上主链路，而是保留为离线实验和效果评估方案。

这个选择主要是在准确率、延迟和部署资源之间做工程权衡。


## 为什么选择 Chroma？

这个项目的数据规模不大，核心需求是：
- 本地持久化
- Metadata Filter
- Python 接入简单
- 方便快速迭代
- 不需要复杂的分布式部署

Chroma 能比较好地满足当前项目规模和开发效率要求。

如果未来知识规模显著增长或需要更高并发，可以进一步评估 Milvus、Qdrant、Elasticsearch 等方案。


## 为什么选择 Sentence-Transformers / MiniLM？

主要考虑三个因素：

1. 支持中英文语义表示；
2. 可以本地运行，不依赖额外 Embedding API；
3. 模型规模相对可控，适合个人项目和 CPU 部署。

实际部署中也发现 Embedding Model 的冷启动和内存占用需要关注，因此最终保留 Lazy Loading，而没有强制在 Streamlit 启动时 Warm-Up。


## 为什么不用 Warm-Up？

尝试过启动阶段提前加载 Embedding Model。

实际测试发现：
- Streamlit 启动明显变慢；
- 当前机器和部署资源下没有改善整体体验；
- 原有 Lazy Loading 在第一次调用后能够正常复用模型。

因此最终选择保留 Lazy Loading：

第一次真正需要 RAG 时加载模型，后续请求复用。


## Fast Path 是什么？

早期所有请求都会进入完整 Agent Loop。

但普通候选人问答其实并不需要 LLM 先做一次 Tool Decision。

因此设计了 Fast Path：

普通问题：
Query
→ Retrieval
→ Context
→ DeepSeek。

复杂任务，例如：
- JD 匹配
- 面试问题生成

才进入 Full Agent：

Query
→ LLM Tool Decision
→ Tool
→ Tool Result
→ LLM Final Answer。

这样减少了一次不必要的 LLM 调用。


## Multi-Tool Agent 怎么实现？

系统定义多个业务 Tool，例如：
- search_resume_knowledge
- analyze_job_fit
- generate_interview_questions

通过 Function Schema 描述 Tool 名称、功能和参数。

LLM 根据任务选择 Tool 后：
1. 解析 Tool Call；
2. Tool Dispatcher 查找对应 Python Function；
3. 执行 Tool；
4. 将 Tool Result 返回给 LLM；
5. LLM 判断是否继续调用其他 Tool；
6. 最终生成答案。

同时设置最大 Agent Step，避免无限循环。


## FastAPI 在项目中起什么作用？

早期 Agent 与 Streamlit 耦合在一起。

后来将 Agent Engine 封装成 FastAPI 服务，提供：
- /health
- /chat

并增加：
- Pydantic Request / Response
- Request ID
- Logging
- 请求耗时统计
- 全局异常处理

这样 UI 和 AI 后端可以分离，后续其他前端也可以直接调用 Agent API。


## Docker 的作用是什么？

Docker 主要用于保证运行环境一致。

项目包含 Python、Sentence-Transformers、Chroma、FastAPI 等依赖，本地和云端环境容易出现差异。

通过 Dockerfile 固定运行环境，可以实现：
代码 + 依赖 + 启动命令统一打包。

同时使用 .dockerignore 排除：
- API Key
- 本地 Chroma
- 缓存
- Git 文件
等不应该进入镜像的内容。


## 遇到过什么部署问题？

一次 Railway 部署中出现：

/health 正常
/chat 返回 502。

排查过程：
1. 检查 FastAPI 日志；
2. 确认请求已经进入应用；
3. 发现 MiniLM 加载阶段后服务进程异常；
4. 结合 Docker Stats 和平台资源限制分析；
5. 判断 1GB 内存不足以稳定支撑当前模型和应用。

这个问题让我认识到：

本地能运行不代表云环境一定能运行，AI 应用部署还必须关注模型内存、CPU、冷启动和容器资源限制。


## 项目中最难的问题是什么？

一个比较典型的问题是 RAG 的检索错误不是程序报错，而是“系统正常运行但答案不正确”。

例如“科研 + 实习”问题中，系统完全没有异常日志，但实际召回结果缺少 internships.md。

解决这类问题不能只看最终 LLM 答案，而需要把系统拆开：

Query
→ Intent
→ Metadata Filter
→ Retrieved Chunks
→ Context
→ LLM。

通过 Debug Retrieval Result 才最终定位到 Multi-Intent 缺失问题。

这个过程让我对 RAG Debug 的理解从“调 Prompt”转向“先检查检索证据是否正确”。


## 项目目前有什么局限？

目前主要有几个局限：

1. Intent Detection 主要采用规则方式，对非常复杂或模糊的 Query 仍有提升空间。
2. Knowledge Base 当前规模较小，还没有验证大规模知识场景。
3. Reranker 因 CPU 延迟没有进入在线主链路。
4. 回答质量仍受到 DeepSeek API 响应速度影响。
5. Multi-Intent Retrieval 目前以类别覆盖为重点，未来可以进一步研究动态 Top-K 和 Query Decomposition。


## 如果重新做一次，会怎么优化？

如果继续迭代，我会重点做：

1. 更完善的 Query Routing 和 Multi-Intent / Query Decomposition；
2. 建立更大的 RAG Evaluation Dataset；
3. 增加 Retrieval 和 Answer 两层评测；
4. 根据部署资源尝试更轻量 Embedding / Reranker；
5. 增加缓存和 API Streaming；
6. 优化招聘方第一次使用时的响应体验；
7. 继续补充真实招聘问题 Knowledge。


## 和直接把简历上传给 ChatGPT 有什么区别？

AI Resume Agent 的重点不是简单让模型读取一份简历。

主要差异是：

1. 候选人知识被结构化管理；
2. 使用 RAG 精确检索相关证据；
3. 可以通过 Metadata 和 Intent 控制检索范围；
4. 有 JD Matching 和面试问题等专门 Tool；
5. 可以进行 Retrieval Evaluation；
6. 可以作为独立 API 和 Web 应用部署；
7. Knowledge 可以持续更新，而不是每次重新上传简历。

因此它更接近一个面向招聘场景的垂直 AI Agent，而不是一次性的文档问答。

# 高频技术面试问题与标准回答

## 1. 你这个 AI Resume Agent 最难的技术问题是什么？

我认为最难的不是把 RAG 跑起来，而是解决“系统没有报错，但检索结果其实是错的”这种问题。

一个典型例子是用户问：

“请介绍高颂岩的科研经历和实习经历。”

旧版本中，这类问题同时涉及“科研经历”和“实习经历”。Router 发现不是单一类别后，会退回全库 Top-K 检索。

结果 Top-5 里可能被 skills.md 和 research.md 占满，internships.md 完全没有被召回。最终 LLM 就会错误回答“没有实习经历资料”。

这个问题的难点在于程序本身没有报错，Chroma、Embedding、LLM 都正常运行，但最终答案还是错的。

我最后通过把链路拆开检查：

Query
→ Intent Detection
→ Metadata Filter
→ Retrieved Chunks
→ Context
→ LLM

定位到问题根源在 Retrieval，而不是 Prompt 或 LLM。

之后将检索架构升级为 Multi-Intent Retrieval，保证多类别问题能够分别检索对应 Knowledge，再合并去重。


## 2. RAG 为什么会检索错，你是怎么定位的？

RAG 检索错误通常不一定是模型能力问题，也可能来自：

- Chunk 缺少上下文
- Query Intent 判断不准确
- 全库 Top-K 被其他类别内容占满
- Metadata Filter 使用不合理
- Embedding 对某些表达的语义区分不够
- Knowledge 组织结构不合理

我的排查方式不是直接改 Prompt，而是先看 Retrieval Result。

具体流程是：

1. 打印 Query；
2. 查看 Intent Detection；
3. 查看是否启用了 Metadata Filter；
4. 查看实际召回了哪些 source 和 section；
5. 判断真正需要的 Chunk 是否进入 Context；
6. 最后才看 LLM 的回答。

例如“科研 + 实习”问题中，最终定位到的根因就是：

多意图问题
→ Router 返回 None
→ 退回 Global Vector Search
→ Top-K 被其他 Chunk 占满
→ internships.md 没进入 Context。

因此我把 Router 从“单类别或 None”升级成“保留所有 Intent”，再配合 Multi-Intent Retrieval 解决。


## 3. 为什么用了 RAG，而不是直接把整份简历放进 Prompt？

直接把整份简历放进 Prompt 对很小的静态 Demo 是可行的，但扩展性比较差。

我使用 RAG 主要有几个原因：

1. 可以只检索和当前问题相关的内容，减少无关 Context；
2. Knowledge 可以持续更新，不需要每次重写 Prompt；
3. 可以通过 Metadata Filter 区分教育、科研、实习、项目和技能；
4. 可以单独评估 Retrieval 是否正确；
5. 更容易扩展到更多资料和更多业务 Tool；
6. 更接近实际大模型应用中的知识检索架构。

例如用户问“AI Resume Agent 怎么实现”，系统没必要把全部教育经历、动物实验和材料表征内容都发送给 LLM。

RAG 可以只提供与 AI 项目相关的证据，减少噪声，也更容易控制事实边界。


## 4. 为什么选择 Chroma？

当前项目规模比较小，我选择 Chroma 主要考虑的是开发效率和功能匹配。

项目需要：

- 本地持久化向量数据
- Metadata Filtering
- Python 接入方便
- 支持快速实验
- 不需要额外维护复杂服务

Chroma 可以满足这些需求，而且与个人项目和本地开发场景比较匹配。

如果未来扩展到更大的知识规模、更高并发或多用户环境，我会进一步评估 Qdrant、Milvus、Elasticsearch 等方案。

所以选择 Chroma 并不是认为它适合所有场景，而是当前项目规模下的工程取舍。


## 5. 为什么不用 LangChain？

这个项目里我没有把 LangChain 作为核心依赖，主要是因为我希望自己实现核心链路，真正理解 RAG 和 Agent 内部是怎么工作的。

例如：

- Query Routing
- Embedding
- Chroma Retrieval
- Tool Schema
- Tool Dispatcher
- Agent Loop
- Tool Result 回传

这些核心逻辑都是自己实现的。

这样做的好处是遇到问题时，我能明确知道错误发生在哪一层，而不是只会调用框架接口。

当然 LangChain 本身很适合快速构建复杂 LLM 应用。如果后续业务规模更大、需要大量标准组件和复杂工作流，我也会考虑使用 LangChain 或类似框架提高开发效率。

当前项目更重视理解底层流程和可控性。


## 6. 为什么不用在线 Reranker？

我做过 Jina multilingual Cross-Encoder Reranker 的离线实验。

实验流程是：

Vector Retrieval
→ Candidate Recall
→ Cross-Encoder Rerank
→ Final Ranking。

离线测试中，Reranker 对一些 Bad Case 的 Top-1 排序确实有明显改善。

但是实际部署测试发现，CPU 环境下 Cross-Encoder 推理延迟比较明显。

因此我最后没有把 Reranker 放入线上主链路，而是保留为离线实验和效果评估方案。

这个决策主要考虑：

- 检索准确率提升
- 响应延迟
- CPU 资源
- 云部署成本
- 用户体验

所以不是“Reranker 没效果”，而是在当前资源条件下，收益不足以抵消在线延迟。


## 7. Fast Path 为什么能提高速度？

早期版本中，所有问题都会进入完整 Agent Loop。

例如一个普通问题：

“高颂岩做过哪些 AI 项目？”

其实只需要：

Query
→ RAG Retrieval
→ Context
→ DeepSeek
→ Answer。

但如果走 Full Agent，会变成：

Query
→ DeepSeek 判断 Tool
→ Tool Calling
→ Retrieval
→ Tool Result
→ DeepSeek 再生成答案。

这样普通问题会多一次 LLM 调用。

因此我加入 Fast Path：

普通候选人问答直接进入 Retrieval + LLM。

只有：

- JD 匹配
- 面试问题生成
- 复杂多步骤任务

才走 Full Agent。

这样优化的核心不是让 Chroma 更快，而是减少一次不必要的 LLM 请求，因此对整体响应时间提升更明显。


## 8. Multi-Intent Retrieval 是怎么实现的？

旧版本 Router 只能返回：

- 一个 source_type
- 或 None。

当问题同时包含多个类别时，例如：

“请介绍高颂岩的科研经历和实习经历。”

Router 会返回 None，然后系统退回全库向量检索。

这会导致不同类别的 Chunk 竞争 Top-K。

升级后，我把 Router 改为返回多个 Intent，例如：

["实习经历", "科研经历"]

之后的检索流程是：

1. Query 只计算一次 Embedding；
2. 每个 Intent 分别使用 Metadata Filter 查询 Chroma；
3. 每个类别召回少量高相关 Chunk；
4. 使用 Round-Robin 方式合并；
5. 对结果进行去重；
6. 限制最终 Chunk 数量，防止 Context 膨胀。

实际 Warm Retrieval 测试结果：

- 单意图约 0.0161 秒
- 双意图约 0.0229 秒
- 五意图约 0.0310 秒

所以多意图检索提高了类别覆盖率，但没有明显影响用户体感。


## 9. 怎么证明你的 RAG 优化有效？

我没有只通过主观观察判断效果，而是建立 Retrieval Evaluation。

主要使用：

- Hit@1
- Hit@3
- Hit@5
- Bad Case Analysis

优化流程是：

Evaluation
→ 找出错误 Query
→ 检查召回 Chunk
→ 分析错误原因
→ 修改 Chunking / Metadata / Routing / Retrieval
→ 再次评测。

例如我做过：

- Pure Vector Retrieval
- Intent-Aware Retrieval
- Reranker

之间的对比。

另外，Multi-Intent Retrieval 修改以后，我还增加了自动测试：

- Router 单元测试 15 项全部通过
- Retrieval 集成测试 3 项全部通过

这样可以保证新的优化既解决当前问题，也不会破坏已有功能。


## 10. Docker 部署时遇到过什么问题？

一个比较典型的问题是在 Railway 部署 FastAPI Agent 时：

/health 可以正常访问，
但 /chat 返回 502。

排查过程是：

1. 先确认 FastAPI 服务本身启动正常；
2. 查看 Railway Deploy Logs；
3. 发现 /chat 请求进入 Embedding Model 加载阶段后服务进程异常；
4. 本地使用 docker stats 检查容器内存占用；
5. 结合 Railway 约 1GB 的资源限制判断，当前 Sentence-Transformers + Agent 服务的内存需求超过平台限制。

这个问题让我认识到：

本地 Docker 能运行并不代表云端一定稳定，AI 应用部署还需要考虑：

- 模型内存占用
- CPU
- 冷启动
- 平台内存限制
- 容器资源
- 模型缓存

后续也尝试过线程、量化、模型缓存等方向，但部分优化反而增加内存，因此最终回退到稳定版本。


## 11. 如果有 10 万份简历，这个系统还能不能这么做？

不能完全照现在的架构直接放大。

当前系统是单候选人、较小 Knowledge Base，Chroma 本地持久化完全够用。

如果扩展到 10 万份简历，我会从几个方面调整：

1. 为每个候选人建立 candidate_id 等 Metadata；
2. 检索前先根据候选人、岗位等条件做 Metadata Filter；
3. 使用更适合大规模数据和并发的向量数据库；
4. 对 Knowledge 做异步批量 Embedding；
5. 将检索服务和 LLM 服务拆分；
6. 增加缓存；
7. 增加权限控制；
8. 加入分页、索引更新和数据版本管理；
9. 考虑 Hybrid Search；
10. 做更完整的在线 Evaluation 和 Monitoring。

如果并发量进一步增加，还需要考虑 API Gateway、负载均衡、任务队列以及模型服务扩容。

所以当前架构主要验证的是核心产品逻辑，而不是直接面向 10 万份简历的生产系统。


## 12. 你和计算机专业学生相比有什么优势？

我的优势不是计算机基础一定比计算机专业学生更强，而是我具备比较明显的跨领域背景和真实科研工程经验。

材料和生物工程经历让我长期接触：

- 实验设计
- 数据分析
- 文献检索
- 复杂问题拆解
- 多轮方案迭代
- 科研项目推进

这些能力和 AI 应用开发中的：

- Evaluation
- Bad Case Analysis
- Prompt / Retrieval 调试
- 产品问题拆解
- 业务场景理解

有比较强的迁移关系。

同时，我不是只停留在“学 AI 概念”，而是已经独立完成：

RAG
→ Agent
→ FastAPI
→ Docker
→ Testing
→ Deployment

这一整套应用开发链路。

因此我的差异化优势更偏向：

“跨领域理解能力 + 科研分析能力 + AI 应用落地能力”。


## 13. 为什么应该招一个跨专业转 AI 的你？

我认为跨专业本身既不是优势，也不是劣势，关键是有没有证明自己能够完成能力迁移。

我原有的科研经历让我具备：

- 问题拆解
- 数据分析
- 实验验证
- 项目推进
- 快速学习新领域

而转向 AI 之后，我已经通过真实项目证明自己不仅是在学习概念，而是能够独立完成：

Knowledge Base
→ RAG
→ Agent
→ Tool Calling
→ FastAPI
→ Docker
→ 公网部署
→ Evaluation
→ Debugging。

我最大的特点是能够把科研中形成的“提出问题—设计方案—验证结果—分析 Bad Case—继续迭代”的方法迁移到 AI 应用开发。

因此，如果岗位需要的是能够快速学习、理解业务并把大模型真正落地成产品的人，我认为我的跨学科背景会成为一种补充能力，而不是障碍。