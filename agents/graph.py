from langgraph.graph import StateGraph, END
from .states import AgentState
from .nodes import (
    retrieve_node, 
    perception_decision_node, 
    perception_action_node, 
    generate_report_node
)

def create_medical_graph():
    workflow = StateGraph(AgentState)

    # 1. 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("decide_perception", perception_decision_node)
    workflow.add_node("get_perception", perception_action_node)
    workflow.add_node("generate_report", generate_report_node)

    # 2. 建立逻辑连接
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "decide_perception")

    # 3. 条件路由：根据决策节点的 needs_perception 字段分流
    workflow.add_conditional_edges(
        "decide_perception",
        lambda x: "action" if x["needs_perception"] else "skip",
        {
            "action": "get_perception",
            "skip": "generate_report"
        }
    )

    # 4. 影像计算完成后，也要进入生成报告节点
    workflow.add_edge("get_perception", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()

# 初始化图实例
medical_app = create_medical_graph()
