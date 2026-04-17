from agents.states import AgentState
from core.initializer import logic_llm, report_llm
from rag.hybrid_searcher import hybrid_searcher
from perception.perception import MedicalPerception

# 1. 检索节点
def retrieve_node(state: AgentState):
    query = state["query"]
    docs = hybrid_searcher.search(query) 
    return {"context_docs": docs}

# 2. 决策节点 (YES/NO)
def decide_perception_node(state: AgentState):
    prompt = f"问题: {state['query']}\n是否涉及肿瘤体积、直径等需要影像量化计算的内容？只回答 YES 或 NO。"
    # 使用 0 温控的逻辑模型
    response = logic_llm.invoke(prompt)
    decision = "YES" in response.content.upper()
    return {"needs_perception": decision}

# 3. 影像感知节点
def perception_node(state: AgentState):
    result = MedicalPerception().run_inference(image_path=state.get("image_path"))
    return {"perception_data": result}

# 4. 报告生成节点
def generate_report_node(state: AgentState):
    # 汇总所有 State 信息，交给 LLM 生成报告
    prompt = f"""
    参考指南: {state['context_docs']}
    影像指标: {state.get('perception_data', '不涉及')}
    用户问题: {state['query']}
    请生成一份严谨的医疗建议报告。
    """
    res = report_llm.invoke(prompt)
    return {"report": res.content}
