import json
import warnings
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from k8s_tools import get_deployments, get_pods, restart_deployment, scale_deployment, get_logs, find_deployment


class GraphState(TypedDict):
    question: str
    context: str
    chat_history: list
    answer: str
    relevant: str   # "yes" | "no"
    sources: list
    score_threshold: float
    intent: str     # "k8s" | "question"
    k8s_action: str
    k8s_target: str
    k8s_namespace: str


def build_graph(retriever, llm, vectorstore=None):

    # ── 노드 1: intent 분류 ───────────────────────────────────────────
    def classify_intent(state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """다음 질문이 Kubernetes 클러스터 제어 명령인지 판단하세요.
제어 명령 예시: "nginx 재시작해줘", "pod 상태 확인해줘", "deployment 중지해줘", "로그 보여줘", "스케일 줄여줘"
반드시 'k8s' 또는 'question' 중 하나만 영어로 답하세요."""),
            ("human", "{question}"),
        ])
        result = (prompt | llm).invoke({"question": state["question"]})
        intent = "k8s" if "k8s" in result.content.lower() else "question"
        return {**state, "intent": intent}

    # ── 노드 2: k8s 명령 파싱 ────────────────────────────────────────
    def parse_k8s_command(state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """다음 명령에서 대상과 액션을 추출해 JSON으로만 답하세요.
액션 종류: restart, stop, start, status, logs
네임스페이스가 명시되지 않으면 namespace는 빈 문자열로.

예시 출력: {{"action": "restart", "target": "nginx-demo", "namespace": ""}}"""),
            ("human", "{question}"),
        ])
        result = (prompt | llm).invoke({"question": state["question"]})
        try:
            raw = result.content.strip()
            # 코드블록 제거
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
        except Exception:
            parsed = {"action": "status", "target": "", "namespace": ""}

        return {
            **state,
            "k8s_action": parsed.get("action", "status"),
            "k8s_target": parsed.get("target", ""),
            "k8s_namespace": parsed.get("namespace", ""),
        }

    # ── 노드 3: k8s 명령 실행 ────────────────────────────────────────
    def execute_k8s(state: GraphState) -> GraphState:
        action = state["k8s_action"]
        target = state["k8s_target"]
        ns = state["k8s_namespace"]

        # 네임스페이스 자동 탐색
        if target and not ns:
            found = find_deployment(target)
            if found:
                target, ns = found
            else:
                ns = "default"

        try:
            if action == "restart":
                result = restart_deployment(target, ns)
            elif action == "stop":
                result = scale_deployment(target, 0, ns)
            elif action == "start":
                result = scale_deployment(target, 1, ns)
            elif action == "logs":
                result = get_logs(target, ns)
            else:  # status
                if target:
                    result = get_pods(ns if ns else None)
                else:
                    result = get_deployments()
        except Exception as e:
            result = f"실행 중 오류: {e}"

        return {**state, "answer": result, "sources": []}

    # ── 노드 4: MSP 관련성 판단 ──────────────────────────────────────
    def check_relevance(state: GraphState) -> GraphState:
        check_prompt = ChatPromptTemplate.from_messages([
            ("system", """다음 질문이 MSP 운영(VM, Kubernetes, 네트워크, 보안, 모니터링, 장애처리 등 IT 운영)과
관련 있는지 판단하세요. 반드시 'yes' 또는 'no' 중 하나만 영어로 답하세요."""),
            ("human", "{question}"),
        ])
        result = (check_prompt | llm).invoke({"question": state["question"]})
        content = result.content.lower().strip()
        relevant = "yes" if "yes" in content or "예" in content or "관련" in content else "no"
        return {**state, "relevant": relevant}

    # ── 노드 5: 검색 + 답변 생성 ─────────────────────────────────────
    def retrieve_and_answer(state: GraphState) -> GraphState:
        if vectorstore is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                docs_with_scores = vectorstore.similarity_search_with_relevance_scores(state["question"], k=4)
            threshold = state.get("score_threshold", 0.3)
            docs = [doc for doc, score in docs_with_scores if score >= threshold]
            sources = list({doc.metadata.get("source", "")
                            for doc, score in docs_with_scores
                            if score >= threshold and doc.metadata.get("source")})
        else:
            docs = retriever.invoke(state["question"])
            sources = list({doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")})

        if not docs:
            return {**state, "context": "", "answer": "매뉴얼에서 확인이 어렵습니다.", "sources": []}

        context = "\n".join([doc.page_content for doc in docs])
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 MSP 운영팀의 운영 도우미입니다.
아래 매뉴얼 내용을 바탕으로만 답변하세요. 매뉴얼에 없는 내용은 '매뉴얼에서 확인이 어렵습니다'라고 답하세요.
반드시 한국어로만 답변하세요.

[참고 매뉴얼]
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        response = (answer_prompt | llm).invoke({
            "context": context,
            "question": state["question"],
            "chat_history": state["chat_history"],
        })
        return {**state, "context": context, "answer": response.content, "sources": sources}

    # ── 노드 6: 관련 없음 반환 ───────────────────────────────────────
    def reject(state: GraphState) -> GraphState:
        return {**state, "answer": "MSP 운영과 관련 없는 질문입니다. 운영 매뉴얼 관련 질문을 입력해주세요.", "sources": []}

    # ── 분기 함수 ────────────────────────────────────────────────────
    def route_intent(state: GraphState) -> str:
        return "parse_k8s_command" if state["intent"] == "k8s" else "check_relevance"

    def route_relevance(state: GraphState) -> str:
        return "retrieve_and_answer" if state["relevant"] == "yes" else "reject"

    # ── 그래프 조립 ──────────────────────────────────────────────────
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("parse_k8s_command", parse_k8s_command)
    graph.add_node("execute_k8s", execute_k8s)
    graph.add_node("check_relevance", check_relevance)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("reject", reject)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent)
    graph.add_edge("parse_k8s_command", "execute_k8s")
    graph.add_edge("execute_k8s", END)
    graph.add_conditional_edges("check_relevance", route_relevance)
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("reject", END)

    return graph.compile()
