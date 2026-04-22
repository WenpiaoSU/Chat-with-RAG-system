# -*- coding: utf-8 -*-
"""
Streamlit WebUI 主入口

提供 RAG 系统的可视化界面，包括知识库管理、问答和评估功能。
"""

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="RAG 智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 导入页面模块
from chat import render_chat_page
from knowledge_base import render_knowledge_base_page
from evaluation import render_evaluation_page


def init_session_state() -> None:
    """初始化会话状态"""
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieval_params" not in st.session_state:
        st.session_state.retrieval_params = {
            "top_k": 5,
            "score_threshold": 0.5,
            "enable_rerank": True,
        }


def get_rag_chain():
    """获取或初始化 RAG Chain"""
    if st.session_state.rag_chain is None:
        try:
            from src.rag.rag_chain import RAGChain
            st.session_state.rag_chain = RAGChain.from_defaults()
        except Exception as e:
            st.error(f"初始化 RAG Chain 失败: {e}")
            return None
    return st.session_state.rag_chain


def get_knowledge_base():
    """获取或初始化知识库管理器"""
    if st.session_state.knowledge_base is None:
        try:
            from src.storage.knowledge_base import KnowledgeBaseManager
            from src.embedding import BGEEmbedding

            embedder = BGEEmbedding.from_settings()
            st.session_state.knowledge_base = KnowledgeBaseManager(
                collection_name="default",
                embedder=embedder,
            )
        except Exception as e:
            st.error(f"初始化知识库失败: {e}")
            return None
    return st.session_state.knowledge_base


def main() -> None:
    """主函数"""
    init_session_state()

    st.title("🤖 RAG 智能问答系统")
    st.markdown("---")

    # 侧边栏导航
    with st.sidebar:
        st.title("功能导航")
        page = st.radio(
            "选择功能",
            ["💬 智能问答", "📚 知识库管理", "📊 RAG 评估"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("v1.0.0")

    # 根据选择渲染对应页面
    if page == "💬 智能问答":
        render_chat_page(get_rag_chain, get_knowledge_base)
    elif page == "📚 知识库管理":
        render_knowledge_base_page(get_knowledge_base)
    elif page == "📊 RAG 评估":
        render_evaluation_page(get_rag_chain, get_knowledge_base)


if __name__ == "__main__":
    main()
