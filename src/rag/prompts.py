# -*- coding: utf-8 -*-
"""
RAG 提示词模板

提供 RAG 系统所需的各种提示词模板，包括检索、生成、总结等场景。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


@dataclass
class PromptTemplates:
    """提示词模板集合"""
    
    DEFAULT_SYSTEM_PROMPT: str = """你是一个专业的AI助手，擅长根据提供的上下文信息来回答用户的问题。

重要规则：
1. 仅根据提供的上下文信息回答问题，不要编造信息
2. 如果上下文中没有相关信息，诚实地告知用户
3. 回答应当清晰、准确、专业
4. 可以适当引用上下文中的原文来支撑回答
"""
    
    RAG_SYSTEM_PROMPT: str = """你是一个专业的问答助手，擅长从文档中提取信息并准确回答问题。

能力：
- 深入理解文档内容和语义
- 准确引用相关段落
- 在信息不确定时诚实表达

要求：
1. 只使用提供的上下文来回答问题
2. 如果上下文中没有答案，明确说明这一点
3. 回答要基于事实，不要添加上下文中不存在的信息
4. 用清晰的格式组织回答，如有必要可使用列表
"""
    
    CONDENSE_QUESTION_PROMPT: str = """给定以下对话历史和一个后续问题，将后续问题重写为一个独立的、完整的问题，使其能够直接从知识库中检索到相关信息。

对话历史：
{chat_history}

后续问题：{question}

独立问题："""

    QA_PROMPT_TEMPLATE: str = """上下文信息：
{context}

---
用户问题：{question}

请根据上述上下文信息来回答用户的问题。"""

    SUMMARIZE_PROMPT: str = """请对以下文本进行简要总结，提取关键信息和要点：

{text}

总结："""

    ENTITY_EXTRACTION_PROMPT: str = """从以下文本中提取所有重要的实体（人名、地名、组织名、专有名词等）和它们之间的关系：

{text}

提取结果（使用JSON格式）："""

    HYBRID_SEARCH_PROMPT: str = """分析以下查询，生成适合混合检索的关键词和语义表示：

查询：{query}

生成：
1. 关键词列表：
2. 语义扩展查询：
3. 同义词：
"""

    REFINE_PROMPT: str = """原始问题：{question}
当前回答：{existing_answer}
额外上下文：{context}

请根据额外上下文来改进和完善当前的回答。"""

    CRITIQUE_PROMPT: str = """你是一个评判专家，负责评估回答的质量。

问题：{question}
回答：{answer}

请评估以下方面并给出评分（1-10）和反馈：
1. 准确性 - 回答是否正确
2. 相关性 - 回答是否与问题相关
3. 完整性 - 回答是否完整
4. 清晰度 - 回答是否清晰易懂

评估结果："""


class PromptManager:
    """提示词管理器
    
    管理和构建 RAG 系统所需的各类提示词模板。
    
    Example:
        >>> manager = PromptManager()
        >>> qa_prompt = manager.build_qa_prompt(
        ...     context="上下文信息...",
        ...     question="问题是什么？"
        ... )
        >>> print(qa_prompt)
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        max_context_length: int = 8000,
    ) -> None:
        """初始化提示词管理器
        
        Args:
            system_prompt: 自定义系统提示词
            use_history: 是否使用对话历史
            max_context_length: 最大上下文长度（字符数）
        """
        self.system_prompt = system_prompt or PromptTemplates.DEFAULT_SYSTEM_PROMPT
        self.use_history = use_history
        self.max_context_length = max_context_length
        
        self._templates = PromptTemplates()
    
    def build_system_prompt(
        self,
        include_context_hint: bool = True,
        **kwargs: Any,
    ) -> List[SystemMessage]:
        """构建系统提示词
        
        Args:
            include_context_hint: 是否在提示词中包含上下文提示
            **kwargs: 其他参数
            
        Returns:
            List[SystemMessage]: 系统消息列表
        """
        prompt = self.system_prompt
        
        if include_context_hint:
            prompt += "\n\n你可以访问的知识库信息将在后续提供。"
        
        return [SystemMessage(content=prompt)]
    
    def build_qa_prompt(
        self,
        context: str,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List:
        """构建问答提示词
        
        Args:
            context: 上下文信息
            question: 用户问题
            history: 对话历史
            
        Returns:
            List: 消息列表
        """
        messages = self.build_system_prompt()
        
        if self.use_history and history:
            history_text = self._format_history(history)
            messages.append(HumanMessage(content=f"对话历史：\n{history_text}\n"))
        
        qa_template = PromptTemplate(
            template=self._templates.QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        
        messages.append(
            HumanMessage(content=qa_template.format(
                context=self._truncate_context(context),
                question=question
            ))
        )
        
        return messages
    
    def build_condense_prompt(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> ChatPromptTemplate:
        """构建问题改写提示词
        
        用于将对话历史中的问题改写为独立问题。
        
        Args:
            question: 当前问题
            chat_history: 对话历史
            
        Returns:
            ChatPromptTemplate: 改写提示词模板
        """
        history_text = self._format_history(chat_history) if chat_history else "无"
        
        template = PromptTemplate(
            template=self._templates.CONDENSE_QUESTION_PROMPT,
            input_variables=["chat_history", "question"],
        )
        
        return ChatPromptTemplate.from_messages([
            ("user", template.format(
                chat_history=history_text,
                question=question
            ))
        ])
    
    def build_refine_prompt(
        self,
        question: str,
        existing_answer: str,
        context: str,
    ) -> ChatPromptTemplate:
        """构建答案优化提示词
        
        Args:
            question: 原始问题
            existing_answer: 当前回答
            context: 额外上下文
            
        Returns:
            ChatPromptTemplate: 优化提示词模板
        """
        template = PromptTemplate(
            template=self._templates.REFINE_PROMPT,
            input_variables=["question", "existing_answer", "context"],
        )
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=template.format(
                question=question,
                existing_answer=existing_answer,
                context=context
            ))
        ])
    
    def build_critique_prompt(
        self,
        question: str,
        answer: str,
    ) -> ChatPromptTemplate:
        """构建评价提示词
        
        Args:
            question: 问题
            answer: 回答
            
        Returns:
            ChatPromptTemplate: 评价提示词模板
        """
        template = PromptTemplate(
            template=self._templates.CRITIQUE_PROMPT,
            input_variables=["question", "answer"],
        )
        
        return ChatPromptTemplate.from_messages([
            ("user", template.format(question=question, answer=answer))
        ])
    
    def build_summarize_prompt(
        self,
        text: str,
        max_length: Optional[int] = None,
    ) -> List[SystemMessage]:
        """构建总结提示词
        
        Args:
            text: 待总结文本
            max_length: 最大总结长度（字符数）
            
        Returns:
            List[SystemMessage]: 消息列表
        """
        if max_length:
            text = text[:max_length]
        
        messages = [
            SystemMessage(content="你是一个专业的文本总结助手。"),
            HumanMessage(content=self._templates.SUMMARIZE_PROMPT.format(text=text))
        ]
        
        return messages
    
    def build_hybrid_search_prompt(
        self,
        query: str,
    ) -> List[HumanMessage]:
        """构建混合搜索提示词
        
        Args:
            query: 查询文本
            
        Returns:
            List[HumanMessage]: 消息列表
        """
        return [
            HumanMessage(content=self._templates.HYBRID_SEARCH_PROMPT.format(query=query))
        ]
    
    def _format_history(
        self,
        history: List[Dict[str, str]],
    ) -> str:
        """格式化对话历史
        
        Args:
            history: 对话历史列表
            
        Returns:
            str: 格式化后的历史记录
        """
        if not history:
            return "无"
        
        formatted = []
        for i, entry in enumerate(history[-5:], 1):
            role = entry.get("role", "user")
            content = entry.get("content", "")
            formatted.append(f"第{i}轮 - {role}：{content}")
        
        return "\n".join(formatted)
    
    def _truncate_context(self, context: str) -> str:
        """截断上下文以符合最大长度限制
        
        Args:
            context: 原始上下文
            
        Returns:
            str: 截断后的上下文
        """
        if len(context) <= self.max_context_length:
            return context
        
        return context[:self.max_context_length] + "\n\n[上下文已被截断...]"


def get_default_prompt_manager() -> PromptManager:
    """获取默认的提示词管理器
    
    Returns:
        PromptManager: 默认提示词管理器实例
    """
    return PromptManager(
        system_prompt=PromptTemplates.RAG_SYSTEM_PROMPT,
        use_history=True,
        max_context_length=8000,
    )
