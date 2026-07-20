from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 노드 간에 주고받는 데이터 구조
class GraphState(TypedDict):
    question: str
    context: str
    chat_history: list
    answer: str
    relevant: str  # "yes" | "no"
    sources: list


def build_graph(retriever, llm):

    # ── 노드 1: 관련성 판단 ──────────────────────────────────────────
    def check_relevance(state: GraphState) -> GraphState:
        check_prompt = ChatPromptTemplate.from_messages([
            ("system", """다음 질문이 MSP 운영(VM, Kubernetes, 네트워크, 보안, 모니터링, 장애처리 등 IT 운영)과
관련 있는지 판단하세요. 반드시 'yes' 또는 'no' 중 하나만 영어로 답하세요."""),
            ("human", "{question}"),
        ])
        chain = check_prompt | llm
        result = chain.invoke({"question": state["question"]})
        content = result.content.lower().strip()
        relevant = "yes" if "yes" in content or "예" in content or "관련" in content else "no"
        return {**state, "relevant": relevant}

    # ── 노드 2: 검색 + 답변 생성 ─────────────────────────────────────
    def retrieve_and_answer(state: GraphState) -> GraphState:
        docs = retriever.invoke(state["question"])
        context = "\n".join([doc.page_content for doc in docs])
        sources = list({doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")})

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 MSP 운영팀의 운영 도우미입니다.
아래 매뉴얼 내용을 바탕으로만 답변하세요. 매뉴얼에 없는 내용은 '매뉴얼에서 확인이 어렵습니다'라고 답하세요.
반드시 한국어로만 답변하세요.

[참고 매뉴얼]
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        chain = answer_prompt | llm
        response = chain.invoke({
            "context": context,
            "question": state["question"],
            "chat_history": state["chat_history"],
        })
        return {**state, "context": context, "answer": response.content, "sources": sources}

    # ── 노드 3: 관련 없음 반환 ───────────────────────────────────────
    def reject(state: GraphState) -> GraphState:
        return {**state, "answer": "MSP 운영과 관련 없는 질문입니다. 운영 매뉴얼 관련 질문을 입력해주세요.", "sources": []}

    # ── 분기 함수 ────────────────────────────────────────────────────
    def route(state: GraphState) -> str:
        return "retrieve_and_answer" if state["relevant"] == "yes" else "reject"

    # ── 그래프 조립 ──────────────────────────────────────────────────
    graph = StateGraph(GraphState)

    graph.add_node("check_relevance", check_relevance)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("reject", reject)

    graph.set_entry_point("check_relevance")
    graph.add_conditional_edges("check_relevance", route)
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("reject", END)

    return graph.compile()
