# Chat-with-RAG-system
## 功能实现

**LLM & Embedding Core**

* [x] 集成 Xinference 推理框架，实现大模型本地化部署。
* [x] 兼容 OpenAI API 协议。
* [x] ​**Embedding Models**​: 支持本地 Embedding 模型的调用与切换（e.g., `bge-large-zh`, `Qwen3-embedding`）。
* [ ] ​**Multimodal**​: 接入多模态大模型，支持“图+文”跨模态理解。

**Data Pipeline**

* [x] ​**Document Loader**​: 支持格式：Markdown, PDF, Word (`.docx`), Images (OCR)。PPT支持（待开发）。
* [x] ​**Text Splitting**​: 实现基于分隔符的递归切分及 NLTK/Spacy 语义切分策略。
* [ ] ​**Data Cleaning**​: 实现文档自动清理与格式规范化。
* [ ] ​**Automated Pipeline**​: 搭建 `Load` -> `Chunk` -> `Embed` -> `Store` 自动化处理流水线。
* [ ] **Vector Store**​: 集成 Chroma 向量库进行存储管理。
* [ ] ​**Indexing Strategy**​: 实现父子索引（Parent-Child Indexing）策略。

**Retrieval & Generation (RAG)**

* [ ] ​**Query Rewriting**​: 实现查询改写模块。
* [ ] ​**Hybrid Search**​: 混合检索策略（向量相似度 + BM25 关键词），支持加权融合。
* [ ] ​**Re-ranking**​: 引入重排序模型（e.g., `BGE-Reranker`）对召回文档进行精细打分。
* [ ] ​**Prompt Engineering**​: 构建动态提示词模版（System Prompt + Context + History + Query）。

**Evaluation**

* [ ] 基于 Ragas 框架实现检索与生成的自动化评估指标。

**UI**

* [ ] 基于 Streamlit 构建对话 Demo 及知识库管理界面。

**Fine-tuning**
未完待续...
