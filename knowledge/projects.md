# 项目经历


## AI Resume Agent｜面向招聘场景的 RAG + Multi-Tool Agent 智能简历系统

- 项目角色：独立设计与开发。
- 项目定位：面向 HR、招聘方和面试官开发可交互 AI 简历 Agent，将传统静态简历升级为具备自然语言问答、岗位匹配、面试辅助及工具自主调用能力的智能招聘应用。
- 技术栈：Python、Streamlit、FastAPI、DeepSeek API、OpenAI SDK、Sentence-Transformers、Chroma、RAG、Embedding、Contextual Chunking、Cross-Encoder Reranker、Tool Calling、Multi-Tool Agent、Docker、Git/GitHub。

### 产品与业务能力

- 将候选人的教育背景、科研经历、实习经历、项目经历及技术能力构建为结构化个人知识库，使招聘方可以通过自然语言快速查询候选人信息。
- 支持候选人背景介绍、AI 项目查询、科研经历查询、技术能力说明、实习经历查询及职业方向说明等招聘高频问题。
- 支持招聘方直接输入岗位 JD，由 Agent 自动分析岗位要求，并从候选人的真实经历中检索相关能力证据，生成岗位匹配分析。
- 根据目标岗位和候选人的真实项目经历自动生成针对性面试问题、追问方向及对应能力考察点，提高面试信息获取效率。
- 使用 Streamlit Session State 实现多轮对话，使招聘方可以围绕上一轮答案继续追问项目细节、技术实现和岗位适配情况。

### Knowledge Base 与知识工程

- 将个人资料按职责拆分为教育、科研、实习、项目和技能等独立 Markdown Knowledge 文件，减少同一事实在多个文件重复出现导致的检索歧义。
- 根据 Markdown Section 自动进行知识切块，使不同经历保持独立语义单元，便于后续向量化及精确检索。
- 针对普通 Chunk 脱离原始文档后容易出现主体缺失的问题，实现 Contextual Chunking，自动为每个知识片段补充候选人姓名、资料类别及章节主题。
- 增加无效 Chunk 过滤逻辑，自动剔除仅包含 Markdown 标题、没有真实正文信息的空壳片段，降低无效知识对向量召回结果的干扰。

### Embedding 与 Chroma Vector Database

- 使用 Sentence-Transformers 对用户 Query 与知识片段生成文本 Embedding，通过向量空间语义相似度完成知识召回。
- 使用 Chroma 构建本地持久化向量数据库，统一存储知识文本、Embedding 及 Source、资料类型、Section Title 等 Metadata。
- 实现 Top-K Semantic Retrieval，为 RAG、岗位匹配 Tool 和面试问题 Tool 提供真实候选人资料。
- 基于 SHA256 Hash 实现 Knowledge 自动同步，当 Markdown 内容发生变化时自动重建 Chroma 向量索引。
- 将 Index Version 纳入索引 Hash，当 Chunking 或 Context 构造策略发生变化时，即使 Knowledge 原文未改变，也能够自动触发向量索引升级。

### Two-Stage Retrieval 与 Reranker

- 构建“向量召回 + Cross-Encoder 重排序”的 Two-Stage Retrieval Pipeline：
  
  Query  
  → Embedding  
  → Chroma Top-K Recall  
  → Multilingual Cross-Encoder Reranker  
  → Final Top-K Context

- 第一阶段使用 Embedding + Chroma 进行高召回语义检索，快速筛选相关候选知识片段。
- 第二阶段使用 Multilingual Cross-Encoder 对 Query 与候选 Chunk 进行联合相关性评分，对初步召回结果进行精细重排序。
- 通过 Recall + Rerank 架构改善单一 Embedding 模型在 Top-1 精确排序上的局限，提高高相关证据进入最终上下文的概率。

### RAG 与事实约束生成

- 构建完整 Retrieval-Augmented Generation 链路：
  
  User Query  
  → Query Embedding  
  → Vector Retrieval  
  → Reranker  
  → Context Build  
  → DeepSeek LLM  
  → Grounded Answer

- 将检索得到的候选人真实履历作为 LLM 上下文，降低个人经历类问答中的模型幻觉。
- 通过 System Prompt 约束 Agent 的角色、工具使用方式和事实边界，要求涉及候选人经历的信息必须基于 Knowledge 或 Tool Result，不允许虚构经历。
- 当知识库信息不足时，要求系统明确说明资料不足，而不是通过模型自由补全。

### Retrieval Evaluation 与 Bad Case Optimization

- 构建独立 RAG Retrieval Evaluation 测试集，对教育背景、科研经历、AI 项目、技术能力、求职方向等不同类型问题进行统一评测。
- 使用 Hit@1、Hit@3、Hit@5 作为主要检索指标，对 Chroma Baseline 与 Reranker Pipeline 进行量化比较。
- 针对 Top-1 未命中问题建立 Bad Case 列表，并逐条分析 Knowledge 内容重复、Chunk Context Loss、无效 Chunk、Embedding 语义偏差及知识组织方式等问题。
- 基于评测结果持续优化 Knowledge Structure、Contextual Chunking 和 Retrieval Pipeline，而不是依靠主观判断调整模型。
- 形成：
  
  Evaluation  
  → Bad Case Analysis  
  → Retrieval Optimization  
  → Re-Evaluation
  
  的完整 RAG 优化闭环。

### Tool Calling 与 Multi-Tool Agent

- 将不同招聘业务能力封装为独立 LLM Tool，通过 Function Schema 描述工具名称、用途和调用参数。
- 实现 `search_resume_knowledge`，用于检索候选人的教育、科研、实习、项目和技术能力信息。
- 实现 `analyze_job_fit`，根据岗位 JD 自动检索候选人与岗位要求相关的经历和能力证据。
- 实现 `generate_interview_questions`，根据目标岗位与候选人经历生成针对性的面试问题、追问方向及考察重点。
- 实现统一 Tool Dispatcher，根据 LLM 返回的 Tool Name 自动调用对应 Python Function，并解析模型生成的 JSON Arguments。
- 支持 Multi-Tool 协同，一个用户任务中 Agent 可以连续调用多个 Tool，并综合多个 Tool Result 后生成最终答案。

### Agent Loop

- 自主实现 Multi-Step Agent Loop：
  
  User Task  
  → LLM Decision  
  → Tool Calling  
  → Tool Execution  
  → Tool Result  
  → LLM Replanning  
  → Continue / Final Answer

- Agent 每轮根据当前上下文和 Tool Result 判断是否需要继续调用其他工具，在获得足够信息后自动结束工具调用并生成最终回答。
- 设置最大执行轮数，防止异常情况下出现无限 Tool Calling。
- 保留 Agent Tool Trace，记录工具名称及参数，便于分析 Agent 决策路径并进行问题排查。

### FastAPI 与工程化

- 将 Agent Engine 封装为 FastAPI REST API，实现 Streamlit 前端与 RAG / Agent 核心逻辑解耦。
- 最终形成：
  
  Streamlit Frontend  
  → FastAPI  
  → Agent Engine  
  → Tool Dispatcher  
  → Retrieval Pipeline  
  → DeepSeek API

- 对 UI、Agent Engine、Agent Tools、Vector Store、Reranker、Evaluation 等模块进行独立代码拆分，提高项目可维护性和扩展性。
- 增加异常处理机制，对 LLM API、Vector Retrieval 和 Tool Execution 异常进行捕获和反馈，避免单个模块异常导致服务整体中断。
- 增加 Logging、Tool Trace 和请求耗时记录，用于分析系统运行状态、Agent 行为及核心链路延迟。
- 使用 Secrets / Environment Variables 管理 API Key 等敏感配置，避免密钥直接写入代码仓库。
- 使用 requirements.txt 管理 Python 依赖，并通过 Git Branch 区分稳定版本与功能开发版本。

### Docker 与部署

- 编写 Dockerfile 对 FastAPI + Agent 后端进行容器化，将 Python Runtime、项目代码、依赖环境及服务启动命令统一封装为 Docker Image。
- 通过 Docker 提高本地开发、测试环境与线上部署环境的一致性。
- 完成 Agent V2 云端部署，使完整 AI Resume Agent 支持公网访问和招聘场景在线演示。
- 使用 Git/GitHub 完成项目版本管理和持续功能迭代。

### 项目成果

- 独立完成从个人 Knowledge Base 构建、Contextual Chunking、Embedding、Vector Database、Two-Stage Retrieval、RAG Evaluation、Reranker、Tool Calling、Multi-Tool Agent、Agent Loop、FastAPI、工程化、Docker 到在线部署的完整大模型应用开发流程。
- 项目不仅实现“能回答问题”的 AI 应用，还通过 Retrieval Evaluation、Bad Case Analysis 和检索优化对系统效果进行量化验证，形成可迭代、可调试、可部署的完整 AI Agent 工程。


## 科研文献智能分析 Agent｜面向科研场景的多文献 RAG + Research Agent

- 项目角色：独立设计与开发。
- 项目定位：面向科研人员开发文献智能分析 Agent，解决传统科研阅读中 PDF 信息分散、实验参数提取耗时、多论文横向比较困难及结论难以快速追溯等问题。
- 技术栈：Python、Streamlit、FastAPI、DeepSeek API、PyPDF、Sentence-Transformers、Chroma、RAG、Embedding、Reranker、Structured Extraction、Tool Calling、Agent、Docker。

### PDF 文献解析

- 支持上传科研论文 PDF，并自动提取标题、摘要、研究方法、实验条件、结果及结论等文本信息。
- 针对科研论文结构进行 Section-aware Parsing，尽量保持 Abstract、Methods、Results、Conclusion 等章节语义完整性。
- 对长篇 PDF 文本进行分段和 Chunking，为后续向量检索和多文献问答建立基础数据结构。

### 文献知识库

- 将不同论文按照 Document ID、论文题目、章节、页码、来源等信息构建结构化文献 Knowledge Base。
- 保存 Chunk 与原始论文之间的映射关系，使回答中的结论能够追溯至具体论文和文献片段。
- 支持多篇论文同时建立知识库，为横向文献对比和主题分析提供数据基础。

### Embedding、Vector Database 与 Reranker

- 使用 Sentence-Transformers 对论文 Chunk 和用户 Query 进行 Embedding。
- 使用 Chroma 存储文献向量、论文来源、章节信息和 Metadata，实现跨文献 Top-K 语义检索。
- 引入 Cross-Encoder Reranker 对初步召回的文献片段进行二次重排序，提高专业科研问题下高相关证据片段的排序质量。
- 构建：
  
  Query  
  → Vector Recall  
  → Reranker  
  → Evidence Context
  
  的多文献检索流程。

### Multi-Document RAG

- 支持基于多篇科研论文进行自然语言问答，而不是仅对单篇论文进行摘要。
- 根据问题从多个 PDF 中检索相关证据，并结合 DeepSeek LLM 生成基于文献内容的结构化回答。
- 在回答中保留论文来源和证据信息，降低科研场景中无法追溯答案出处的问题。
- 对资料不足或论文未报告的内容明确标记“当前文献未提供”，避免模型自行补充实验事实。

### 实验参数智能抽取

- 将实验条件抽取封装为独立 Research Tool，从论文 Methods / Experimental Section 中提取材料组成、比例、温度、时间、浓度、处理条件、测试方法等关键参数。
- 将自然语言实验描述转换为结构化字段，降低人工逐篇记录实验条件的时间成本。
- 支持针对同一研究问题聚合多篇论文参数，为后续实验方案设计和参数区间比较提供参考。

### 多论文对比分析

- 支持按照材料体系、制备方法、实验参数、测试方法、主要结果和研究结论等维度进行多篇论文横向比较。
- 自动整理不同论文之间的共同点、差异点及可能影响实验结果的关键变量。
- 将分散在多篇论文中的研究结论整理为结构化对比结果，提高科研综述和方案调研效率。

### Research Tool Calling

- 将 PDF 检索、论文摘要、实验参数抽取、多论文比较和证据查询等功能封装为独立 Tool。
- Agent 根据科研问题自动判断应该进行普通文献检索、参数抽取还是跨论文比较。
- 支持在一个科研任务中连续调用多个 Tool，例如：
  
  检索相关论文证据  
  → 提取实验参数  
  → 对比不同论文  
  → 综合形成研究结论

### Research Agent

- 构建面向科研任务的 Multi-Tool Agent，使系统不再局限于单轮 PDF 问答，而能够根据任务目标自主选择工具并完成多步骤信息处理。
- 通过 Agent Loop 将论文检索、参数提取、比较分析及最终总结组合成完整科研工作流。
- 通过 System Prompt 对文献事实边界进行约束，要求关键结论必须来源于已解析论文内容。

### FastAPI 与工程化

- 使用 FastAPI 将文献解析、RAG Retrieval 和 Research Agent 能力服务化，实现前后端解耦。
- 增加异常处理、Logging、运行耗时记录和配置管理，提高长文档分析过程中的稳定性和可调试性。
- 使用模块化代码结构拆分 PDF Parser、Vector Store、Retriever、Reranker、Research Tools 和 Agent Engine。

### Docker 与部署

- 使用 Docker 对 Research Agent 后端及运行依赖进行容器化。
- 完成科研文献智能分析 Agent 在线部署，支持通过 Web 页面上传 PDF、查询论文和执行多文献分析任务。

### 项目成果

- 独立完成从 PDF Parsing、文献 Chunking、Embedding、Vector Database、Multi-Document RAG、Reranker、Structured Extraction、Research Tool Calling、Agent Loop、FastAPI 到 Docker 部署的科研 AI 应用开发流程。
- 将通用 RAG + Agent 架构迁移至科研文献分析场景，实现从“问论文”进一步扩展到“抽参数、比论文、找证据、做科研分析”的完整垂直领域 AI Agent。


## “碧海琼硅”功能材料开发项目｜项目负责人

- 时间：2025.09—2026.06。
- 角色：项目负责人。
- 项目方向：围绕硅藻生物硅开展功能材料研发、性能优化、应用场景探索及成果转化。

- 技术路线：牵头制定项目研发路线，根据材料性能、实验结果和应用需求持续调整研究方案及产品开发方向。
- 项目统筹：负责实验任务拆解、成员协同、阶段数据汇总、项目进度跟踪及成果汇报，具备从技术研发到项目推进的完整实践经验。
- 产品开发：围绕硅藻生物硅开展功能材料及衍生产品开发，并结合实验结果持续优化材料性能。
- 成果转化：推动项目与外部单位开展应用及成果转化交流，与青岛啤酒、海南省 301 解放军医院等单位进行合作对接。
- 项目申报：统筹竞赛申报材料、成果展示及项目答辩。
- 项目成果：项目获得 2026 年海南省“挑战杯”大学生创业计划竞赛二等奖。
- 核心能力：形成技术路线设计、研发任务管理、跨团队协作、外部沟通及科研成果转化实践经验。


## 3D 智造快速制模项目｜项目成员

- 时间：2021.05—2023.06。
- 角色：项目成员。
- 项目方向：基于三维建模与 3D 打印开展塑料模具快速制模和样件开发。

- 产品设计：参与产品结构及模具结构设计，根据目标样件需求制定数字化建模方案。
- 工程制图：使用 AutoCAD 完成二维工程图设计。
- 三维建模：使用 Pro/E 完成零部件建模、模具结构模型及装配爆炸图绘制。
- 快速制造：将数字化模型应用于 3D 打印快速制模和实际样件制作。
- 工艺优化：根据样件成型结果持续调整模型结构及工艺参数，提高样件设计和成型效果。
- 项目成果：完成迷你梳子、收音机、玩具汽车等塑料模具样件设计与制作。
- 核心能力：积累产品结构设计、二维工程制图、三维建模、样件验证及工程问题迭代优化经验。