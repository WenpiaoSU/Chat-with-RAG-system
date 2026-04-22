# -*- coding: utf-8 -*-
"""
存储模块

包含向量存储和知识库管理功能。

架构设计：
- vectorstore/: 向量存储层
    - base.py: 向量存储抽象基类
    - chroma_store.py: Chroma 实现
- knowledge_base/: 知识库管理层
    - kb_manager.py: 知识库管理器

使用示例：
```python
from src.storage import (
    ChromaVectorStore,
    KnowledgeBaseManager,
    KnowledgeBaseFactory,
)

# 方式一：直接使用向量存储
vectorstore = ChromaVectorStore(
    collection_name="my_kb",
    persist_directory="./data/vector_db",
)
vectorstore.add_texts(texts, embeddings)

# 方式二：使用知识库管理器
kb = KnowledgeBaseManager(
    collection_name="my_kb",
    embedder=embedding_model,
)
kb.add_documents_from_files(["doc1.pdf", "doc2.md"])

# 方式三：从配置创建
kb = KnowledgeBaseFactory.create(
    collection_name="my_kb",
    embedder=embedding_model,
)
```
"""

from src.storage.vectorstore.base import (
    BaseVectorStore,
    VectorStoreConfig,
    SearchResult,
    AddResult,
)
from src.storage.vectorstore.chroma_store import (
    ChromaVectorStore,
    ChromaConfig,
)
from src.storage.knowledge_base.kb_manager import (
    KnowledgeBaseManager,
    KnowledgeBaseFactory,
    DocumentRecord,
    KnowledgeBaseStats,
)

__all__ = [
    # VectorStore
    "BaseVectorStore",
    "VectorStoreConfig",
    "SearchResult",
    "AddResult",
    "ChromaVectorStore",
    "ChromaConfig",
    # KnowledgeBase
    "KnowledgeBaseManager",
    "KnowledgeBaseFactory",
    "DocumentRecord",
    "KnowledgeBaseStats",
]
