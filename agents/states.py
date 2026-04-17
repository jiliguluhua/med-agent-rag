from typing import Annotated, List, Optional, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # 原始查询
    query: str
    image_path: Optional[str]  # 影像文件路径
    # 检索到的医学文档列表 (来自 rag 目录)
    context_docs: List[str]
    # 影像计算指标 (来自 perception 目录)
    perception_data: str
    # 决策标志：是否需要进行影像计算
    needs_perception: bool
    # 最终生成的报告
    report: str
