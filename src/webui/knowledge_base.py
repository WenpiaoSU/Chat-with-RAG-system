# -*- coding: utf-8 -*-
"""
知识库管理页面

提供文档上传、查看、删除等知识库管理功能。
"""

import streamlit as st
from typing import Any, Callable, List


def render_knowledge_base_page(get_knowledge_base: Callable) -> None:
    """渲染知识库管理页面

    Args:
        get_knowledge_base: 获取知识库的回调函数
    """
    st.header("📚 知识库管理")

    kb = get_knowledge_base()

    # 标签页：文档列表 | 上传文档
    tab1, tab2 = st.tabs(["📋 文档列表", "📤 上传文档"])

    with tab1:
        _render_document_list(kb)

    with tab2:
        _render_document_upload(kb)


def _render_document_list(kb: Any) -> None:
    """渲染文档列表

    Args:
        kb: 知识库管理器实例
    """
    if kb is None:
        st.warning("知识库未初始化")
        return

    # 获取统计信息
    try:
        stats = kb.get_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("总块数", stats.total_chunks)
        col2.metric("文档来源", len(stats.sources))
        col3.metric("集合名称", stats.collection_name)
    except Exception as e:
        st.error(f"获取统计信息失败: {e}")

    st.markdown("---")

    # 文档来源列表
    if kb and stats.sources:
        st.subheader("📁 文档来源")
        for source in stats.sources:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(source)
            with col2:
                if st.button("🗑️ 删除", key=f"del_{source}"):
                    try:
                        kb.delete_by_source(source)
                        st.success(f"已删除: {source}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")
    else:
        st.info("暂无文档，请上传文档到知识库")

    # 清空知识库
    st.markdown("---")
    st.subheader("⚠️ 危险操作")
    if st.button("清空所有文档", type="secondary"):
        if st.checkbox("确认清空知识库（此操作不可恢复）"):
            try:
                kb.reset(confirm=True)
                st.success("知识库已清空")
                st.rerun()
            except Exception as e:
                st.error(f"清空失败: {e}")


def _render_document_upload(kb: Any) -> None:
    """渲染文档上传组件

    Args:
        kb: 知识库管理器实例
    """
    st.info("支持格式: PDF, DOCX, MD, PNG, JPG, PPTX, CSV")

    # 文件上传
    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "docx", "md", "png", "jpg", "pptx", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"已选择 {len(uploaded_files)} 个文件")

        if st.button("🚀 开始处理"):
            if kb is None:
                st.error("知识库未初始化")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"正在处理: {uploaded_file.name}")

                try:
                    import tempfile
                    import os

                    # 保存临时文件
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=uploaded_file.name
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # 添加到知识库
                    kb.add_documents_from_files([tmp_path])

                    # 清理临时文件
                    os.unlink(tmp_path)

                except Exception as e:
                    st.error(f"处理 {uploaded_file.name} 失败: {e}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("处理完成!")
            st.success(f"成功处理 {len(uploaded_files)} 个文件")
            st.rerun()
