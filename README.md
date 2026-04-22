# Chat-with-RAG-system

## 项目描述

基于 LangChain 框架实现的本地知识库 RAG（Retrieval-Augmented Generation）智能问答系统。支持多格式文档处理、混合检索、重排序、多模态理解等功能，提供完整的 RAG 评估体系。

**核心特性**：
- 🌐 支持 Xinference 本地部署 / OpenAI API 两种大模型调用方式
- 📄 多格式文档加载：Markdown、PDF、Word、图片（OCR/多模态）、PPT、CSV
- 🔍 混合检索：HNSW 语义检索 + BM25 关键词 + RRF 融合 + 重排序
- 📊 完整的 RAG 评估体系：基于 Ragas 框架
- 🎨 Streamlit 可视化界面 + FastAPI RESTful 接口

---

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| **大模型** | OpenAI API / Xinference | 兼容 OpenAI 协议 |
| **Embedding** | bge-large-zh / Qwen3-embedding | 支持本地部署 |
| **Reranker** | BGE-Reranker | 文档重排序 |
| **多模态** | Qwen3-VL-8B-Instruct | 图文理解 |
| **向量库** | Chroma | 本地向量存储 |
| **OCR** | RapidOCR | 图片/扫描件识别 |
| **文本分割** | LangChain + 自定义 | 递归/语义分割 |
| **LLM框架** | LangChain | RAG 编排 |
| **后端框架** | FastAPI | RESTful API |
| **前端框架** | Streamlit | 可视化界面 |
| **评估框架** | Ragas | RAG 质量评估 |

---

## 功能实现

### LLM & Embedding Core

- [x] **大模型调用**：支持 API 调用（暂不实现 Xinference 本地部署）
- [ ] **Embedding Models**：支持本地 Embedding 模型调用与热切换（`bge-large-zh`、`Qwen3-embedding`）
- [ ] **Multimodal**：接入多模态大模型，支持"图+文"跨模态理解

### Data Pipeline

- [x] **Document Loader**：支持 Markdown、PDF、Word（`.docx`）、图片、PPT、CSV
- [x] **Text Splitting**：基于分隔符的递归切分及 NLTK 语义切分
- [x] **Data Cleaning**：文档自动清理与格式规范化
- [x] **Vector Store**：集成 Chroma 向量库
- [x] **Indexing Strategy**：父子索引（Parent-Child Indexing）策略

### Retrieval & Generation (RAG)

- [x] **Query Rewriting**：查询改写模块（含自动降级机制）
- [x] **Hybrid Search**：HNSW 语义检索 + BM25 关键词 + RRF 加权融合
- [x] **Re-ranking**：引入 BGE-Reranker 进行精细打分
- [ ] **Prompt Engineering**：动态提示词模板（System Prompt + Context + History + Query）

### Evaluation

- [x] **Ragas 评估**：检索与生成的自动化评估指标
- [x] **测试集生成**：基于 LLM 自动生成问答测试集

### WebUI & API

- [ ] **Streamlit**：对话 Demo 及知识库管理界面
- [ ] **FastAPI**：RESTful 接口（`/v1/chat/completions`, `/kb/upload`, `/kb/search`）

---

## 项目架构

```
Chat-with-RAG-system/
├── src/                          # 源代码目录
│   ├── __init__.py
│   │
│   ├── config/                   # 配置管理
│   │   ├── __init__.py
│   │   ├── settings.py           # 全局配置类
│   │   └── config.yaml          # YAML 配置文件
│   │
│   ├── llm/                     # 大模型调用
│   │   ├── __init__.py
│   │   ├── base.py           # LLM 基类
│   │   └── openai_llm.py     # OpenAI 实现
│   │
│   ├── embedding/            # Embedding 模型
│   │   ├── __init__.py
│   │   ├── base.py            # Embedding 基类
│   │   └── bge_embedding.py   # BGE 实现
│   │
│   ├── rag/                 # RAG Chain 核心功能实现
│   │   ├── __init__.py
│   │   ├── base.py          # Chain 基类
│   │   ├── prompts.py       # 统一管理 prompt
│   │   └── rag_chain.py     # RAG 总体实现
│   │
│   ├── data_process/             # 数据处理相关代码
│   │   ├── __init__.py
│   │   ├── loaders/              # 文档加载器
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Loader 抽象基类
│   │   │   ├── loader_factory.py  # 加载器工厂
│   │   │   ├── pdf_loader.py      # PDF 加载器（PyMuPDF + OCR）
│   │   │   ├── markdown_loader.py # Markdown 加载器
│   │   │   ├── docx_loader.py     # Word 加载器
│   │   │   ├── image_loader.py    # 图片加载器（OCR）
│   │   │   ├── ppt_loader.py      # PPT 加载器
│   │   │   └── csv_loader.py      # CSV 加载器
│   │   │
│   │   ├── splitters/            # 文本分割器
│   │   │   ├── __init__.py
│   │   │   ├── base.py                    # Splitter 基类
│   │   │   ├── recursive_splitter.py      # 递归字符分割
│   │   │   ├── semantic_splitter.py       # 语义分割
│   │   │   ├── parent_child_splitter.py   # 父子文档分割
│   │   │   └── markdown_splitter.py       # Markdown 结构分割
│   │   │
│   │   ├── cleaners/             # 数据清洗
│   │   │   ├── __init__.py
│   │   │   └── text_cleaner.py   # 文本清洗工具
│   │   │
│   │   └── data_pipeline.py    # 数据处理流水线，输入文档 → 加载对应的加载器 → 文本清洗与分割
│   │
│   ├── storage/                  # 存储层
│   │   ├── __init__.py
│   │   ├── vectorstore/          # 向量存储
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # VectorStore 基类
│   │   │   └── chroma_store.py   # Chroma 实现
│   │   │
│   │   └── knowledge_base/       # 知识库管理
│   │       ├── __init__.py
│   │       └── kb_manager.py        # 知识库管理器
│   │
│   ├── retrieval/                # 检索模块
│   │   ├── __init__.py
│   │   ├── base.py              # 检索器基类
│   │   ├── semantic_search.py   # 语义检索（HNSW）
│   │   ├── keyword_search.py    # 关键词检索（BM25）
│   │   ├── hybrid_search.py     # 混合检索 + RRF
│   │   ├── query_rewriter.py    # 查询改写
│   │   └── reranker.py          # 重排序（BGE-Reranker）
│   │
│   ├── evaluation/               # 评估模块
│   │   ├── __init__.py
│   │   ├── evaluator.py          # RAG 评估器主类
│   │   ├── metrics.py            # 评估指标定义
│   │   ├── ragas_eval.py         # Ragas 评估器
│   │   ├── testset_generator.py  # 测试集生成器
│   │   └── reporter.py           # 评估报告生成器
│   │
│   ├── api/                      # API 层
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI 入口
│   │   ├── routes/               # 路由
│   │   │   ├── __init__.py
│   │   │   ├── chat.py           # 对话接口
│   │   │   ├── knowledge_base.py # 知识库接口
│   │   │   └── evaluation.py     # 评估接口
│   │   ├── schemas/              # 数据模型
│   │   │   ├── __init__.py
│   │   │   ├── request.py        # 请求模型
│   │   │   └── response.py       # 响应模型
│   │   └── dependencies.py       # 依赖注入
│   │
│   └── webui/                    # WebUI
│       ├── __init__.py
│       ├── main.py               # Streamlit 入口
│       ├── pages/
│       │   ├── __init__.py
│       │   ├── chat.py           # 对话页面
│       │   ├── knowledge_base.py # 知识库管理页面
│       │   └── evaluation.py     # 评估页面
│       └── components/
│           ├── __init__.py
│           ├── chat_box.py        # 对话组件
│           ├── file_uploader.py   # 文件上传组件
│           └── settings.py        # 设置面板
│
├── data/                         # 数据目录
│   ├── knowledge_db/            # 知识库文件
│   ├── vector_db/               # 向量数据库存储
│   └── uploads/                 # 用户通过前端上传的文档暂存路径
│
├── tests/                        # 测试目录
│   ├── __init__.py
│   ├── unit/                    # 单元测试
│   │   ├── __init__.py
│   │   ├── test_loaders.py
│   │   ├── test_splitters.py
│   │   ├── test_retrieval.py
│   │   └── test_rag.py
│   └── integration/             # 集成测试
│       ├── __init__.py
│       └── test_pipeline.py
│
├── examples/                     # 示例文件
├── docs/                         # 文档
├── logs/                         # 日志目录
│
├── pyproject.toml               # Poetry 配置
├── .env.example                 # 环境变量示例
├── README.md
└── requirements.txt
```

---

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/your-repo/Chat-with-RAG-system.git
cd Chat-with-RAG-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
```

### 3. 启动服务

**方式一：仅启动 API**

```bash
cd src
uvicorn api.main:app --reload --port 8000
```

**方式二：启动完整服务（API + WebUI）**

```bash
cd src
streamlit run webui/main.py --server.port 8501
```

---

## 未来规划

- [ ] Xinference 本地部署支持
- [ ] 多用户知识库隔离
- [ ] 增量索引更新
- [ ] 知识库版本管理
- [ ] 更多 Embedding 模型支持
- [ ] Agent 能力扩展
- [ ] 模型微调
