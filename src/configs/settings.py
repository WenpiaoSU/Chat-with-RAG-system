# -*- coding: utf-8 -*-
"""
配置管理模块

本模块从 config.yaml 加载所有配置参数，提供统一的配置访问接口。
配置支持环境变量插值（${ENV_VAR} 语法）。
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class OpenAIConfig(BaseModel):
    """OpenAI API 配置"""
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    streaming: bool = True


class LLMConfig(BaseModel):
    """大模型配置"""
    provider: str = "openai"
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)


class BGEEmbeddingConfig(BaseModel):
    """BGE Embedding 模型配置"""
    model_name: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cuda"
    batch_size: int = 32
    dimension: int = 1024
    normalize_embeddings: bool = True
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    passage_instruction: str = ""


class EmbeddingConfig(BaseModel):
    """Embedding 配置"""
    provider: str = "bge"
    bge: BGEEmbeddingConfig = Field(default_factory=BGEEmbeddingConfig)


class RerankerConfig(BaseModel):
    """Reranker 配置"""
    enabled: bool = True
    provider: str = "bge"
    model_name: str = "BAAI/bge-reranker-large"
    device: str = "cuda"
    top_k: int = 20
    max_length: int = 512


class ChromaConfig(BaseModel):
    """Chroma 向量库配置"""
    persist_directory: str = "./data/vector_db"
    collection_name: str = "default"
    distance_metric: str = "cosine"


class VectorStoreConfig(BaseModel):
    """向量数据库配置"""
    provider: str = "chroma"
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)


class ChunkingConfig(BaseModel):
    """文本分割配置"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: List[str] = Field(default_factory=lambda: [
        "\n\n", "\n", "。|！|？", r"\.\s|\!\s|\?\s", "；|;\s", "，|,\s"
    ])


class ParentChildConfig(BaseModel):
    """父子文档分割配置"""
    enabled: bool = False
    child_chunk_size: int = 200
    parent_chunk_size: int = 1500
    child_overlap: int = 20
    parent_overlap: int = 150


class OCRConfig(BaseModel):
    """OCR 配置"""
    enabled: bool = True
    use_cuda: bool = True
    image_threshold: float = 0.1


class CleaningConfig(BaseModel):
    """数据清洗配置"""
    remove_extra_whitespace: bool = True
    remove_special_chars: bool = False
    strip_html: bool = True


class DocumentConfig(BaseModel):
    """文档处理配置"""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    parent_child: ParentChildConfig = Field(default_factory=ParentChildConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)


class HybridSearchConfig(BaseModel):
    """混合检索配置"""
    enabled: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    rrf_k: int = 60


class QueryRewriteConfig(BaseModel):
    """查询改写配置"""
    enabled: bool = True
    timeout: float = 3.0
    enable_fallback: bool = True


class RetrievalConfig(BaseModel):
    """检索配置"""
    top_k: int = 5
    score_threshold: float = 0.5
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    query_rewrite: QueryRewriteConfig = Field(default_factory=QueryRewriteConfig)


class ConversationConfig(BaseModel):
    """对话配置"""
    max_messages: int = 20
    max_tokens: int = 4096
    include_history: bool = True


class TestSetConfig(BaseModel):
    """测试集生成配置"""
    default_num_pairs: int = 10
    max_context_length: int = 8000
    question_types: List[str] = Field(default_factory=lambda: [
        "fact", "understand", "analyze", "summarize"
    ])


class EvaluatorConfig(BaseModel):
    """评估器配置"""
    batch_size: int = 10
    timeout: int = 60
    save_results: bool = True
    output_dir: str = "./data/evaluation"


class ReportConfig(BaseModel):
    """报告配置"""
    default_format: str = "markdown"
    include_details: bool = True


class EvaluationWeights(BaseModel):
    """评估指标权重配置"""
    faithfulness: float = 0.3
    answer_relevancy: float = 0.3
    context_recall: float = 0.25
    context_precision: float = 0.15


class EvaluationConfig(BaseModel):
    """评估配置"""
    provider: str = "ragas"
    metrics: List[str] = Field(default_factory=lambda: [
        "faithfulness", "answer_relevancy", "context_recall"
    ])
    testset: TestSetConfig = Field(default_factory=TestSetConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    weights: EvaluationWeights = Field(default_factory=EvaluationWeights)


class CORSConfig(BaseModel):
    """CORS 配置"""
    enabled: bool = True
    allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = Field(default_factory=lambda: ["*"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])


class APIConfig(BaseModel):
    """API 配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1
    log_level: str = "info"
    cors: CORSConfig = Field(default_factory=CORSConfig)


class ServerConfig(BaseModel):
    """WebUI 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8501
    page_title: str = "RAG 智能问答系统"
    page_icon: str = "🤖"


class ThemeConfig(BaseModel):
    """WebUI 主题配置"""
    primary_color: str = "#1f77ff"
    background_color: str = "#ffffff"
    text_color: str = "#262730"


class WebUIConfig(BaseModel):
    """WebUI 配置"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/app.log"
    max_bytes: int = 10485760
    backup_count: int = 5


class Settings(BaseModel):
    """全局配置类，整合所有配置项"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    webui: WebUIConfig = Field(default_factory=WebUIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _resolve_env_vars(value: Any) -> Any:
    """递归解析配置值中的环境变量引用 ${ENV_VAR}"""
    if isinstance(value, str):
        import re
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for match in matches:
            env_value = os.environ.get(match, "")
            value = value.replace(f'${{{match}}}', env_value)
        return value
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例（带缓存）
    
    从 config.yaml 加载配置，支持环境变量插值。
    
    Returns:
        Settings: 全局配置实例
    """
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    
    resolved_config = _resolve_env_vars(raw_config)
    return Settings(**resolved_config)


# 全局配置实例
settings = get_settings()
