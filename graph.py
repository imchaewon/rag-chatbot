import json
import warnings
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from k8s_tools import get_resource, restart_deployment, scale_deployment, get_logs, get_containers, find_deployment, find_pod


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
    k8s_resource: str
    k8s_target: str
    k8s_namespace: str


def build_graph(retriever, llm, vectorstore=None):

    # ── 노드 1: intent 분류 ───────────────────────────────────────────
    def classify_intent(state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """다음 질문이 Kubernetes 클러스터 제어 명령인지 판단하세요.
제어 명령 예시: "nginx 재시작해줘", "pod 상태 확인해줘", "deployment 중지해줘", "로그 보여줘", "스케일 줄여줘"
이전 대화 맥락을 참고해 판단하세요.
출력 규칙: k8s 또는 question 중 하나만 출력하세요. 절대 다른 텍스트를 포함하지 마세요."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        result = (prompt | llm).invoke({"question": state["question"], "chat_history": state["chat_history"]})
        cleaned = result.content.lower().strip().strip(".,!? \n\t")
        intent = "k8s" if cleaned == "k8s" else "question"
        return {**state, "intent": intent}

    # ── 노드 2: k8s 명령 파싱 ────────────────────────────────────────
    def parse_k8s_command(state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """이전 대화 맥락을 참고해 현재 명령에서 액션·리소스·대상·네임스페이스를 추출해 JSON으로만 답하세요.

액션 종류:
- get: 리소스 목록/상태 조회
- restart: deployment 재시작
- stop: deployment 중지 (replicas=0)
- start: deployment 시작 (replicas=1)
- logs: 로그 조회
- containers: 특정 pod의 컨테이너 목록 조회

get 액션일 때 resource 필드에 kubectl 리소스명 그대로 입력:
pods, deployments, jobs, services, configmaps, secrets, ingresses,
statefulsets, daemonsets, replicasets, cronjobs, nodes, namespaces,
persistentvolumes, persistentvolumeclaims, serviceaccounts, events 등

네임스페이스 규칙:
- 특정 네임스페이스가 명시된 경우: 해당 값 사용
- "전체", "모든", "all" 네임스페이스를 의미하면: "all"
- 언급이 없으면: 빈 문자열("")

예시 출력:
{{"action": "get", "resource": "pods", "target": "", "namespace": ""}}
{{"action": "get", "resource": "configmaps", "target": "", "namespace": "kube-system"}}
{{"action": "restart", "resource": "deployment", "target": "nginx-demo", "namespace": ""}}
{{"action": "containers", "resource": "pod", "target": "nginx-demo-6b86554995-jdg85", "namespace": ""}}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        result = (prompt | llm).invoke({"question": state["question"], "chat_history": state["chat_history"]})
        try:
            raw = result.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
        except Exception:
            parsed = {"action": "get", "resource": "pods", "target": "", "namespace": ""}

        return {
            **state,
            "k8s_action": parsed.get("action", "get"),
            "k8s_resource": parsed.get("resource", "pods"),
            "k8s_target": parsed.get("target", ""),
            "k8s_namespace": parsed.get("namespace", ""),
        }

    # ── 노드 3: k8s 명령 실행 ────────────────────────────────────────
    def execute_k8s(state: GraphState) -> GraphState:
        action = state["k8s_action"]
        resource = state.get("k8s_resource", "pods")
        target = state["k8s_target"]
        ns = state["k8s_namespace"]

        # target이 있고 ns가 없으면 자동 탐색
        if target and not ns and action != "get":
            is_pod_name = len(target.split("-")) >= 4
            found = find_pod(target) if (action == "containers" or is_pod_name) else find_deployment(target)
            target, ns = found if found else (target, "default")

        try:
            if action == "get":
                result = get_resource(resource, ns if ns else None)
            elif action == "restart":
                result = restart_deployment(target, ns)
            elif action == "stop":
                result = scale_deployment(target, 0, ns)
            elif action == "start":
                result = scale_deployment(target, 1, ns)
            elif action == "logs":
                result = get_logs(target, ns)
            elif action == "containers":
                result = get_containers(target, ns or "default")
            else:
                result = get_resource("pods", ns if ns else None)
        except Exception as e:
            result = f"실행 중 오류: {e}"

        return {**state, "answer": result, "sources": []}

    # ── 노드 4: MSP 관련성 판단 ──────────────────────────────────────
    def check_relevance(state: GraphState) -> GraphState:
        check_prompt = ChatPromptTemplate.from_messages([
            ("system", """다음 질문이 MSP 운영(VM, Kubernetes, 네트워크, 보안, 모니터링, 장애처리 등 IT 운영)과
관련 있는지 판단하세요. 이전 대화 맥락을 참고해 판단하세요.
반드시 'yes' 또는 'no' 중 하나만 영어로 답하세요."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        result = (check_prompt | llm).invoke({"question": state["question"], "chat_history": state["chat_history"]})
        content = result.content.lower().strip()
        relevant = "yes" if "yes" in content else "no"
        return {**state, "relevant": relevant}

    # ── 노드 5: 검색 + 답변 생성 ─────────────────────────────────────
    def retrieve_and_answer(state: GraphState) -> GraphState:
        if vectorstore is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                docs_with_scores = vectorstore.similarity_search_with_relevance_scores(state["question"], k=4)
            threshold = state.get("score_threshold", 0.3)
            docs = [doc for doc, score in docs_with_scores if score >= threshold]
            display_min = max(threshold, 0.2)
            sources = list({doc.metadata.get("source", "")
                            for doc, score in docs_with_scores
                            if score >= display_min and doc.metadata.get("source")})
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

    # ── 노드 6: 비MSP 질문 응답 ──────────────────────────────────────
    def reject(state: GraphState) -> GraphState:
        reject_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 MSP 운영 매뉴얼 기반 챗봇입니다.
사용자가 인사나 간단한 대화를 하면 친근하게 응답하세요.
MSP 운영과 무관한 정보 질문이면 '저는 MSP 운영 매뉴얼 기반 챗봇이라 해당 내용은 답변이 어렵습니다'라고만 하세요.
어떤 경우에도 매뉴얼 외 사실 정보는 절대 제공하지 마세요.
반드시 한국어로 답하세요."""),
            ("human", "{question}"),
        ])
        result = (reject_prompt | llm).invoke({"question": state["question"]})
        return {**state, "answer": result.content, "sources": []}

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
