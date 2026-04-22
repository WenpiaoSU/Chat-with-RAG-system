# -*- coding: utf-8 -*-
"""
智能问答页面

提供基于 RAG 的对话式问答界面。
"""

import streamlit as st
import time
from typing import Any, Callable, List, Optional


def render_chat_page(
    get_rag_chain: Callable,
    get_knowledge_base: Callable,
) -> None:
    """渲染问答页面

    Args:
        get_rag_chain: 获取 RAG Chain 的回调函数
        get_knowledge_base: 获取知识库的回调函数
    """
    st.header("💬 智能问答")

    # 初始化消息历史
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # 侧边栏：检索参数设置
    with st.sidebar:
        st.subheader("🔍 检索参数")
        top_k = st.slider("Top-K", 1, 20, 5)
        score_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.5)
        st.session_state.retrieval_params = {
            "top_k": top_k,
            "score_threshold": score_threshold,
        }

    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("📄 引用来源"):
                            for i, source in enumerate(message["sources"], 1):
                                st.markdown(f"{i}. {source}")

    # 聊天输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # 助手回复
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                try:
                    rag_chain = get_rag_chain()
                    kb = get_knowledge_base()

                    if rag_chain is None:
                        st.error("RAG Chain 未初始化，请检查配置")
                        return

                    # 更新检索参数
                    rag_chain.top_k = st.session_state.retrieval_params["top_k"]
                    rag_chain.score_threshold = st.session_state.retrieval_params["score_threshold"]

                    # 执行 RAG 查询
                    response = rag_chain.invoke(prompt)

                    # 流式显示响应
                    full_response = ""
                    response_area = st.empty()

                    for chunk in _stream_response(response.answer):
                        full_response += chunk
                        response_area.markdown(full_response + "▌")

                    response_area.markdown(full_response)

                    # 显示引用来源
                    if response.sources:
                        with st.expander("📄 引用来源"):
                            for i, source in enumerate(response.sources, 1):
                                st.markdown(f"{i}. {source}")

                    # 保存到历史
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": response.sources,
                    })

                except Exception as e:
                    st.error(f"处理失败: {str(e)}")

    # 清空对话按钮
    if st.session_state.chat_messages:
        if st.button("🗑️ 清空对话"):
            st.session_state.chat_messages = []
            st.rerun()


def _stream_response(text: str, delay: float = 0.01) -> Any:
    """模拟流式输出

    Args:
        text: 响应文本
        delay: 字符间延迟

    Yields:
        str: 单个字符
    """
    for char in text:
        yield char
        time.sleep(delay)
