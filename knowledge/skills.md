# 技能能力


## AI 应用开发能力

- Python：具备 AI 应用开发实践，能够完成文件处理、数据结构操作、函数与模块设计、JSON 数据解析、异常处理及程序调试。

- 大模型 API：能够通过 OpenAI SDK 兼容接口调用 DeepSeek API，将大语言模型能力集成至 Web、后端服务及 Agent 工作流。

- Prompt Engineering：能够设计 System Prompt，对模型角色、任务边界、工具调用规则、事实依据及回答风格进行约束，降低模型幻觉和信息编造。

- RAG：能够独立搭建从 Knowledge Base、Chunking、Embedding、Vector Retrieval、Reranker、Context Build 到 LLM Generation 的完整 Retrieval-Augmented Generation 流程。

- Knowledge Engineering：能够根据业务场景拆分教育、科研、实习、项目和技能等不同知识模块，控制知识职责边界，降低重复信息造成的 Retrieval 歧义。

- Contextual Chunking：能够针对普通文本切块后的 Context Loss 问题，为 Chunk 自动补充主体、资料类型和章节信息，提高独立知识片段的语义完整性。

- Embedding：使用 Sentence-Transformers 对 Query 与知识片段进行文本向量化，实现基于语义相似度的知识检索。

- Vector Database：具备 Chroma 使用实践，能够完成向量持久化、Metadata 管理、Top-K Retrieval 及知识索引自动更新。

- Reranker：能够使用 Multilingual Cross-Encoder 对向量数据库初步召回的候选结果进行二次相关性评分和重排序，构建 Recall + Rerank 两阶段检索架构。

- Retrieval Evaluation：能够建立独立检索测试集，通过 Hit@1、Hit@3、Hit@5 对 Retrieval 效果进行量化评估。

- Bad Case Analysis：能够针对 Top-1 未命中问题分析 Knowledge 重复、Chunk Context Loss、无效 Chunk、Embedding 偏差及 Query-Knowledge 组织不一致等问题，并根据评测结果优化检索链路。

- Tool Calling：能够将 Python 业务函数封装为 LLM Tool，通过 Function Schema 定义 Tool Name、Description 和 Parameters，实现模型自主工具选择和参数生成。

- Multi-Tool Agent：能够为 Agent 注册多个 Tool，使模型根据用户任务连续调用知识检索、岗位匹配、面试问题生成等不同业务能力。

- Tool Dispatcher：能够根据 LLM 返回的 Tool Name 和 JSON Arguments 自动分发并执行对应 Python Function，再将 Tool Result 返回模型继续推理。

- Agent Loop：能够实现 LLM Decision → Tool Calling → Tool Execution → Tool Result → Replanning → Final Answer 的基础 Multi-Step Agent 执行循环。

- Conversation Memory：能够管理 User / Assistant 历史消息，将多轮对话上下文重新加入 Agent Messages，实现连续追问和上下文交互。

- FastAPI：能够将 RAG / Agent 核心能力封装为 REST API，实现 Streamlit 前端与 AI 后端逻辑解耦。

- Streamlit：能够开发聊天式 AI Web 应用，实现页面布局、Chat Input、快捷问题、历史消息展示、Tool Trace 及 Session State 多轮对话管理。

- Docker：具备 AI 应用容器化实践，能够编写 Dockerfile、构建 Docker Image 并运行容器化 FastAPI / Agent 服务。

- Git / GitHub：能够完成代码提交、远程同步、Branch 管理、功能迭代及稳定版本管理。

- 工程化：具备模块化代码设计、Secrets / Environment Variables 管理、异常处理、Logging、Tool Trace、Latency 记录及依赖管理实践。

- AI 应用部署：具备 AI Web 应用与 Agent 服务在线部署实践，能够完成依赖配置、API Key 管理、服务启动及线上版本迭代。


## 科研文献智能分析能力

- PDF 解析：能够使用 Python / PyPDF 对科研论文 PDF 进行文本提取和结构化处理。

- 文献 Chunking：能够根据论文章节和语义结构对长文档进行切块，为后续 Embedding 和 RAG 检索建立数据基础。

- Multi-Document RAG：能够基于多篇科研论文建立向量知识库，并围绕同一科研问题进行跨文献语义检索和综合分析。

- 实验参数抽取：能够将论文 Methods / Experimental Section 中的材料组成、比例、浓度、温度、时间、处理条件及测试方法抽取为结构化信息。

- 多论文比较：能够围绕材料体系、实验条件、测试方法、主要结果和研究结论对多篇论文进行横向比较。

- Evidence Tracking：能够保留回答与原始论文、章节和证据片段之间的映射关系，提高科研问答结果的可追溯性。

- Research Agent：能够将文献检索、参数提取、多论文比较及总结分析封装为 Tool，并通过 Agent Loop 完成多步骤科研信息处理。


## 材料制备与工艺能力

- 具备生物基复合材料、天然高分子/无机复合材料及多孔材料制备实践。
- 熟悉硅藻生物硅提取与纯化、多孔材料冻干成型及相关工艺优化。
- 具备配方设计、工艺参数优化、多组对照及重复性验证经验。
- 能够根据实验结果分析关键影响因素，并持续调整材料配比和工艺条件。


## 材料表征与性能测试

- 具备 SEM、FTIR、热分析、接触角、孔隙率等材料结构与理化性能表征经验。
- 具备吸液、溶胀、降解及拉伸、压缩、弯曲等性能测试经验。
- 熟悉 EDS、XRD、Raman、Micro-CT 等表征方法的基本原理及数据解读。
- 能够结合微观结构与宏观性能分析材料结构—性能关系。


## 生物与功能评价能力

- 具备微生物培养、平板涂布、菌落计数及抗菌性能评价实践。
- 具备全血凝固、血液相容性及止血材料相关功能评价经验。
- 具备动物实验相关实践，可完成腹腔注射、尾静脉注射、心脏采血等操作。
- 参与止血、肝损伤、股动脉出血及创面修复等动物模型实验。
- 具备材料体内功效评价、实验数据记录及结果分析经验。


## 数据分析与科研统计

- 熟练使用 Excel 进行数据整理、基础统计及结果归档。
- 熟练使用 Origin、GraphPad Prism 进行科研数据分析和图表制作。
- 熟悉 ImageJ 图像处理及科研图像定量分析。
- 具备 R 基础数据分析和生物信息学实践。
- 熟悉多组对照、平行重复、单因素方差分析（ANOVA）、Tukey 事后检验及 t 检验等基础统计方法。
- 重视数据一致性、实验可重复性、异常结果复核及原始数据追溯。


## 工程设计与数字化能力

- 熟悉 AutoCAD，可完成二维工程图设计。
- 熟悉 Pro/E、SolidWorks，可进行三维建模、零部件设计及装配结构设计。
- 具备 3D 打印、样件验证及基础模具设计实践。
- 具备 Linux 命令行及 Shell 基础使用能力。


## 科研与文档能力

- 具备英文科研文献检索和阅读能力。
- 具备 SCI 论文、科研报告、技术文档及实验记录整理经验。
- 能够完成科研图表制作、实验数据汇总、阶段成果汇报和项目答辩。
- 具备英文论文写作及科研结果表达基础。


## 项目管理与协作能力

- 具有项目负责人实践，可完成技术路线规划、任务拆解、成员协同、进度跟踪和阶段汇报。
- 具备跨团队沟通及外部合作对接经验。
- 具备科研成果转化、竞赛申报材料整理及项目答辩实践。
- 能够在科研、工程及 AI 开发任务中根据问题反馈持续迭代方案。


## 综合技术特点

- 具有材料成型及控制工程本科与生物技术与工程硕士的交叉技术背景。
- 同时具备材料研发、实验设计、数据分析、科研项目管理和 AI 应用开发实践。
- 能够将科研领域知识与 RAG、Agent、文档智能分析等 AI 技术结合，开发面向科研、知识管理和垂直领域场景的大模型应用。
- 目标方向为 AI 应用开发、大模型应用开发、RAG / Agent 应用开发及科研 AI 应用方向。