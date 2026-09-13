# AI Interactive Resume

一个基于 Python、Streamlit、DeepSeek API 和 RAG 的交互式 AI 简历项目。

面试官可以通过自然语言问答，了解我的教育背景、科研经历、项目经验和技术能力。

## 功能

- 自然语言问答
- Markdown 个人知识库
- 文本切块
- Sentence-Transformers Embedding
- 向量相似度检索
- Top-K 语义召回
- DeepSeek 大模型回答
- Streamlit 多轮对话
- 推荐问题按钮
- 回答资料来源展示

## 技术栈

- Python
- Streamlit
- DeepSeek API
- OpenAI SDK
- Sentence-Transformers
- Scikit-learn
- RAG
- Embedding
- Cosine Similarity
- Top-K Retrieval

## 项目流程

用户提问

↓

问题 Embedding 向量化

↓

与知识库片段向量计算余弦相似度

↓

召回 Top-K 相关知识片段

↓

将检索结果作为上下文交给 DeepSeek

↓

生成基于个人资料的回答

## 项目结构

```text
ai_resume
├── knowledge
│   ├── education.md
│   ├── research.md
│   ├── projects.md
│   └── skills.md
├── embedding_retriever.py
├── main.py
├── profile.md
├── requirements.txt
├── retriever.py
└── README.md