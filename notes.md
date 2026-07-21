# MSP 운영 도우미 챗봇 개발 학습 노트

## 환경 세팅

### 가상환경 (venv)
프로젝트별 독립된 Python 공간. 패키지 버전 충돌 방지.

```bash
python3 -m venv venv        # 가상환경 생성
source venv/bin/activate    # 활성화 (터미널에 (venv) 표시됨)
deactivate                  # 비활성화
```

### 패키지 관리
Spring Boot의 build.gradle과 동일한 역할.

```bash
pip install langchain langchain-groq python-dotenv   # 설치
pip freeze > requirements.txt                        # 목록 저장
pip install -r requirements.txt                      # 목록으로 일괄 설치 (배포 시)
```

### .env 파일
API 키 등 민감한 정보를 저장하는 파일. git에 올리면 안 됨.

```
GROQ_API_KEY=gsk_xxxxxxxx
```

---

## 1단계: LLM 연동

Groq API를 통해 Llama 모델 호출.

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # .env 파일 읽기

llm = ChatGroq(model="llama-3.3-70b-versatile")
response = llm.invoke("안녕!")
print(response.content)
```

---

## 2단계: LangChain 기초

### Prompt Template
LLM에 역할을 부여하고 질문 형식을 정의.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 MSP 운영팀의 운영 도우미입니다."),  # 역할 지정
    ("human", "{question}"),                              # 사용자 질문
])
```

- `system`: LLM의 역할, 말투, 행동 방식 지정
- `human`: 실제 사용자 입력 자리 (`{변수명}` 형태)

### Chain (`|` 연산자)
데이터가 순서대로 흘러가는 파이프. Linux 파이프(`|`)와 동일한 개념.

```python
chain = prompt | llm
response = chain.invoke({"question": "Pod가 CrashLoopBackOff일 때 어떻게 해?"})
```

실행 흐름:
```
invoke({"question": "..."})
  → prompt가 [system, human] 메시지로 변환
  → llm이 받아서 응답 생성
```

나중에 RAG 추가 시:
```python
chain = prompt | retriever | llm
#         ↑         ↑        ↑
#      역할지정   문서검색  응답생성
```

### Hallucination 문제
LLM은 모르는 정보를 그럴듯하게 지어냄 → RAG로 해결

```
현재: 질문 → LLM → 지어낸 답변
목표: 질문 → 매뉴얼 검색 → 검색결과 + 질문 → LLM → 실제 매뉴얼 기반 답변
```

---

## 3단계: RAG 구조

### 모델 두 가지

| 모델 | 역할 | 실행 위치 |
|------|------|----------|
| 임베딩 모델 | 텍스트를 숫자 벡터로 변환 (문서 검색용) | 로컬 (내 맥북) |
| LLM (Groq) | 검색된 문서 보고 답변 생성 | 외부 서버 |

둘은 완전히 독립적. LLM을 Groq → Gemini로 바꿔도 임베딩 모델은 그대로.

### 현재 사용 중인 임베딩 모델
`models/text-embedding-004` (Google Generative AI)
- 외부 API 호출 방식 → 로컬 모델 로드 없이 빠름
- 무료 티어: 분당 1500회
- 데이터가 Google 서버로 전송됨 (보안 민감한 환경엔 부적합)
- 인트라넷 배포 시 로컬 임베딩 모델로 교체 필요

### 임베딩이 하는 두 가지 역할

1. **문서 벡터화** (`ingest.py`) — 정책 문서를 벡터로 변환해서 저장
2. **질문 벡터화** (`main.py`) — 사용자 질문을 벡터로 변환해서 유사한 문서 검색

> ingest.py와 main.py에서 반드시 같은 임베딩 모델을 써야 함.
> 다른 모델 쓰면 벡터 형식이 달라 비교 불가. (cm vs inch 단위 비교 불가와 동일)
> 임베딩 모델을 바꾸면 chroma_db 삭제 후 ingest.py 재실행 필요.

### RAG 전체 흐름

**[준비 단계] `ingest.py` — 한 번만 실행**
```
사내 문서 → 임베딩 모델 → 벡터 변환 → 벡터DB(chroma_db/)에 저장
```

**[서비스 단계] `main.py` — 질문할 때마다 실행**
```
사용자 질문
    ↓
[임베딩 모델] 질문도 벡터로 변환
    ↓
[벡터DB] 유사한 문서 검색
    ↓
[Groq LLM] 문서 + 질문 받아서 답변 생성
    ↓
답변
```

> 임베딩 모델은 ingest.py(문서 변환)와 main.py(질문 변환) 양쪽에서 사용.
> 같은 모델로 변환해야 비교가 가능하기 때문.

### 왜 실행할 때마다 느리냐
- 임베딩 모델(470MB)을 하드디스크 → RAM으로 로드하는 시간
- FastAPI 서버로 만들면 서버 시작 시 한 번만 로드 → 이후 빠름 (5단계에서 해결)

### 벡터DB 종류

| 종류 | 특징 | 적합한 상황 |
|------|------|------------|
| Chroma | 로컬 파일 저장, 무료 | 개발/데모 |
| Pinecone | 클라우드, 유료 | 실서비스 |
| Weaviate | 자체 서버 설치, 무료 | 인트라넷 |
| Milvus | 자체 서버 설치, 무료 | 대용량 |

### 참고 문서 구성 (`docs/msp_manual.txt`)

| 섹션 | 내용 |
|---|---|
| 1. 팀 개요 | MSP 운영팀 역할, 관리 솔루션 소개 |
| 2. 모니터링 | CPU/메모리/디스크 임계값, 점검 주기 |
| 3. VM 운영 | 상태 확인, 재시작 절차, 디스크 증설 |
| 4. K8s 운영 | Pod 상태 확인, CrashLoopBackOff/NotReady 대응 |
| 5. Solar Pro | API 불가 대응, GPU OOM 대응, 업데이트 절차 |
| 6. 장애 대응 | P1~P4 등급 분류, 에스컬레이션 기준 |
| 7. 보안 | 접근 권한 원칙, 보안 사고 대응 |
| 8. 정기 점검 | 일간/주간/월간 체크리스트 |
| 9. 주요 연락처 | Upstage 기술지원, 팀장, 고객사 담당자 |

### 문서 업데이트 시
문서가 바뀌거나 임베딩 모델을 교체하면 반드시 아래 절차 필요.

```bash
rm -rf chroma_db   # 기존 벡터DB 삭제
python ingest.py   # 새 문서로 벡터DB 재생성
```

주기적 자동화는 cron job 등으로 처리 가능.

---

## 4단계: 멀티턴 대화 (대화 히스토리)

### 문제
기존 코드는 매 질문을 독립적으로 처리해 이전 대화를 기억하지 못함.

```
운영자: Pod CrashLoopBackOff 대응 절차 알려줘
챗봇: 로그 확인 → 이벤트 확인 → OOM이면 limit 상향...

운영자: 그럼 롤백은 어떻게 해?   ← "그럼"이 뭘 가리키는지 모름
챗봇: ???
```

### 해결 방법: MessagesPlaceholder

프롬프트 안에 대화 히스토리가 들어갈 자리를 추가하고, 매 대화마다 누적.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),
    MessagesPlaceholder(variable_name="chat_history"),  # 히스토리 자리
    ("human", "{question}"),
])

chat_history = []

# 대화 루프 안에서
response = chain.invoke({
    "context": context,
    "question": question,
    "chat_history": chat_history,   # 누적된 히스토리 전달
})

# 답변 후 히스토리에 추가
chat_history.append(HumanMessage(content=question))
chat_history.append(AIMessage(content=answer))
```

### 메시지 흐름 (3번째 질문 시)

```
[system] 당신은 MSP 운영팀의 운영 도우미입니다...
[human]  Pod CrashLoopBackOff 대응 절차 알려줘  ← 1번째 질문 (히스토리)
[ai]     로그 확인 후 OOM이면 limit 상향...      ← 1번째 답변 (히스토리)
[human]  그럼 롤백 명령어가 뭐야?               ← 2번째 질문 (히스토리)
[ai]     kubectl rollout undo deployment/...    ← 2번째 답변 (히스토리)
[human]  노드 NotReady는 어떻게 해?             ← 현재 질문
```

LLM이 전체 대화 흐름을 보고 답변하므로 "그럼", "거기서" 같은 참조 표현도 이해함.

### 추가된 명령어

| 명령어 | 동작 |
|--------|------|
| `/clear` | 대화 히스토리만 초기화 |
| `/model` | 모델 변경 + 히스토리 자동 초기화 |

### 주의점
히스토리가 길어질수록 LLM에 전달되는 토큰 수 증가 → 응답 느려지고 비용 증가.
실서비스에서는 최근 N개만 유지하거나 요약하는 전략 필요.

---

## 5단계: FastAPI 백엔드

### FastAPI란
HTTP 요청이 들어왔을 때 "이 URL이면 이 함수 실행해"라는 규칙을 정의하는 프레임워크.
Spring Boot 전체에 대응하는 개념.

| Python | Java |
|---|---|
| FastAPI | Spring Boot |
| Uvicorn | Tomcat |
| `@app.post("/chat")` | `@PostMapping("/chat")` |
| Pydantic 모델 | `RequestBody` DTO |

### FastAPI와 Uvicorn의 역할 분리

**Uvicorn** = 포트를 열고 HTTP 요청을 받아서 FastAPI에 넘겨주는 웹 서버. Tomcat과 동일한 역할.
**FastAPI** = 요청 라우팅/처리 규칙 정의. 웹 서버 기능 자체는 없음.

```
브라우저 → Uvicorn (포트 8000, 요청 수신) → FastAPI (라우팅/처리) → 응답
```

Spring Boot는 Tomcat이 내장되어 있어서 `main()` 하나로 둘 다 뜨지만,
FastAPI는 Uvicorn을 별도로 실행해야 함.

```bash
# 이렇게 실행해야 서버가 뜸 (python app.py로 실행하면 바로 종료됨)
uvicorn app:app --reload
#        ↑   ↑
#     파일명  FastAPI 객체명 (app = FastAPI(...))
```

### 왜 쓰는가

**현재 (CLI)**
```
터미널 → main.py → 챗봇 답변
```

**FastAPI 추가 후**
```
웹 브라우저 ──┐
모바일 앱   ──┤→ FastAPI 서버 → 챗봇 답변
다른 서비스 ──┘
```

1. **속도 문제 해결** — `main.py`는 실행할 때마다 임베딩 모델 로드. FastAPI는 서버 시작 시 한 번만 로드
2. **접근성** — 터미널 없이 웹/앱에서 사용 가능

### API 구조

| 엔드포인트 | 역할 |
|---|---|
| `POST /chat` | 질문 전송, 답변 수신 |
| `POST /clear` | 특정 세션 히스토리 초기화 |
| `GET /health` | 서버 상태 확인 |

### 요청/응답 예시

```json
// POST /chat 요청
{
  "question": "Solar Pro API 응답 불가 시 대응 절차 알려줘",
  "session_id": "user_001",
  "model": "groq"
}

// 응답
{
  "answer": "1) API 서버 Pod 상태 확인 2) GPU 메모리 사용량 확인...",
  "session_id": "user_001"
}
```

- `session_id`: 사용자별 대화 히스토리 분리. 여러 명이 동시에 써도 대화 안 섞임
- `model`: `groq` / `gemini` / `solar` 중 선택

### 핵심 코드 구조

```python
# 서버 시작 시 한 번만 실행 (lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, prompt
    embeddings = UpstageEmbeddings(...)   # 모델 로드 (1회)
    vectorstore = Chroma(...)
    retriever = vectorstore.as_retriever()
    yield  # 서버 실행

app = FastAPI(lifespan=lifespan)

# 요청마다 실행
@app.post("/chat")
def chat(req: ChatRequest):
    chat_history = sessions.get(req.session_id, [])
    docs = retriever.invoke(req.question)
    ...
    sessions[req.session_id] = chat_history  # 히스토리 저장
    return ChatResponse(answer=answer, ...)
```

### 실행 방법

```bash
source venv/bin/activate
uvicorn app:app --reload   # --reload: 코드 수정 시 자동 재시작
```

---

## 6단계: 웹 UI

### 구조
FastAPI가 `static/index.html`을 직접 서빙. 별도 프론트 서버 불필요.

```python
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")
```

브라우저에서 `http://localhost:8000` 접속하면 채팅 UI 표시.

### 세션 관리

브라우저마다 랜덤 `session_id`를 생성해 대화 히스토리를 분리.

```javascript
// index.html — 탭 열 때 딱 한 번 생성
const sessionId = "session_" + Math.random().toString(36).slice(2, 9);
```

```python
# app.py — 세션별 히스토리를 서버 메모리에 저장
sessions: dict[str, list] = {}

# 요청마다 해당 세션 히스토리 꺼내서 LLM에 전달
chat_history = sessions.get(req.session_id, [])
# ...LLM 호출 후...
sessions[req.session_id] = chat_history  # 누적 저장
```

### 채팅 1회의 실제 흐름

```
브라우저 → POST /chat { question, session_id, model }
               ↓
          서버: sessions에서 이전 대화 꺼내기
               ↓
          벡터DB 검색 (관련 문서)
               ↓
          LLM API 호출 (히스토리 + 문서 + 질문 통째로 전송)
               ↓
          브라우저 ← { answer, session_id }
```

메시지 1개 = API 요청 1회. ChatGPT도 동일한 구조.
LLM이 "기억"하는 게 아니라 매번 전체 대화 내역을 처음부터 읽는 방식.

### 주의점
~~현재 히스토리는 **서버 메모리**에 저장 → 서버 재시작 시 모든 대화 초기화.~~
→ 7단계에서 SQLite로 해결.

---

## 7단계: SQLite 히스토리 영구 저장

### Redis vs DB 선택 기준

| | Redis | SQLite/DB |
|---|---|---|
| 속도 | ~1ms | ~5ms |
| 영구 저장 | 기본 X | O |
| 설치 | 별도 서버 필요 | 파일 하나로 끝 |

LLM 응답이 1,000~3,000ms이라 히스토리 조회 5ms 차이는 체감 불가.
트래픽이 많지 않은 지금 단계에서는 SQLite로 충분.

### DB 스키마

```sql
CREATE TABLE chat_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL,   -- 'human' | 'ai'
    content    TEXT    NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### 파일 구조

```
database.py   -- DB 연결 및 CRUD 함수 모음
app.py        -- database.py 함수 호출
chat_history.db  -- 실제 데이터 저장 파일 (gitignore)
```

### database.py 핵심 함수

```python
def init_db():       # 서버 시작 시 테이블 생성
def get_history():   # 세션 히스토리 조회 → LangChain 메시지 객체 리스트 반환
def save_messages(): # 질문/답변 한 쌍 저장
def clear_history(): # 세션 히스토리 삭제
```

### app.py 변경 전/후

```python
# 변경 전 (메모리)
sessions: dict[str, list] = {}
chat_history = sessions.get(req.session_id, [])
sessions[req.session_id] = chat_history

# 변경 후 (SQLite)
chat_history = get_history(req.session_id)
save_messages(req.session_id, req.question, answer)
```

서버를 재시작해도 이전 대화 내역이 유지됨.

---

## LangGraph

### LangChain vs LangGraph

| | LangChain | LangGraph |
|---|---|---|
| 구조 | 단순 체인 (순서대로 실행) | 노드 + 엣지 (조건 분기, 루프 가능) |
| 적합한 경우 | 질문 → 검색 → 답변 같은 단순 흐름 | 판단, 반복, 여러 경로가 필요한 복잡한 흐름 |

### LangGraph 이전의 한계

지금은 질문이 뭐든 무조건 벡터DB 검색 → LLM 호출.
"안녕하세요", "점심 뭐 먹지" 같은 무관한 질문도 API 크레딧 소모.
관련 없는 질문에 대한 응답은 시스템 프롬프트("매뉴얼에 없으면 모른다고 해")에 LLM 양심을 맡기는 방식.

```
질문 → 벡터DB 검색 → LLM → 답변   (항상 이 경로, 모델에 따라 지시 무시 가능)
```

### LangGraph 도입 후 흐름

```
질문
 ↓
[관련성 판단 노드] ← LLM이 "MSP 운영과 관련 있나?" 판단
 ├── yes → 벡터DB 검색 → LLM → 답변
 └── no  → "MSP 운영과 관련 없는 질문입니다" 즉시 반환 (검색 skip)
```

판단 주체는 동일하게 LLM이지만, **판단을 구조적으로 분리**해서 흐름을 제어할 수 있게 됨.
관련 없으면 벡터DB 검색을 아예 안 함 → 비용/시간 절약.

### 구현 — GraphState (노드 간 공유 데이터)

```python
class GraphState(TypedDict):
    question: str
    context: str
    chat_history: list
    answer: str
    relevant: str  # "yes" | "no"
```

모든 노드는 이 딕셔너리를 받아서 수정 후 반환. `{**state, "relevant": "yes"}` 형태로 나머지는 그대로 두고 필요한 키만 업데이트.

### 구현 — 노드와 엣지

```python
graph = StateGraph(GraphState)

graph.add_node("check_relevance", check_relevance)   # 노드 등록
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("reject", reject)

graph.set_entry_point("check_relevance")             # 시작점
graph.add_conditional_edges("check_relevance", route) # 조건 분기
graph.add_edge("retrieve_and_answer", END)
graph.add_edge("reject", END)

compiled_graph = graph.compile()
```

### 실제 호출 — app.py `/chat-graph-stream` 엔드포인트

`graph.astream_events()`로 이벤트를 하나씩 받아 SSE로 스트리밍.

```python
async for event in graph.astream_events(initial_state, version="v2"):
    if event["event"] == "on_chat_model_stream":
        node = event.get("metadata", {}).get("langgraph_node", "")
        if node == "retrieve_and_answer":   # 답변 노드 토큰만 전송
            token = event["data"]["chunk"].content
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
    elif event["event"] == "on_chain_end":
        node = event.get("metadata", {}).get("langgraph_node", "")
        if node == "reject":                # 거절 메시지 한 번에 전송
            full_answer = event["data"]["output"]["answer"]
            yield f"data: {json.dumps({'type': 'token', 'content': full_answer})}\n\n"
```

`check_relevance` 노드도 LLM을 호출하므로 `langgraph_node` 필터로 `retrieve_and_answer` 토큰만 골라냄.

### LangGraph 모드의 한계

- 관련성 판단도 LLM이 하므로 소형 모델(3B)은 신뢰도 낮음 → 7B 이상 권장
- 사내 고유 용어(PPP 네트워크 등)는 일반 LLM이 IT 관련으로 인식 못할 수 있음

### LangGraph가 진짜 유용해지는 시점

| 시나리오 | 설명 |
|---|---|
| 실제 액션 수행 | "VM 재시작해줘" → kubectl 명령 직접 실행하는 에이전트 |
| 자동 재검색 루프 | 답변 품질이 낮으면 자동으로 다시 검색 후 재답변 |
| 멀티스텝 워크플로우 | 여러 매뉴얼을 단계적으로 검색해 종합 답변 생성 |

---

## 대화 히스토리 압축

### 문제

히스토리를 누적해서 LLM에 전달하면 대화가 길어질수록 토큰이 폭발적으로 증가.
→ 응답 느려짐, 비용 증가, Groq 같은 무료 티어는 Rate Limit 429 오류 발생.

### 해결 방법: Window + Summary

최근 N턴은 그대로 유지하고, 그 이전 대화는 LLM이 요약해서 압축.

```
DB (원본 전체 보존)
1턴: 질문A / 답변A
2턴: 질문B / 답변B    ← 오래된 대화
3턴: 질문C / 답변C
4턴: 질문D / 답변D
5턴: 질문E / 답변E    ← 최근 5턴 유지
6턴: 질문F / 답변F
             ↓
LLM에 전달되는 것 (6턴 질문 시)
[SystemMessage: "1~2턴 요약: A와 B에 대해 대화함"]
3턴~5턴 원본
현재 질문F
```

### 핵심 코드

```python
MAX_HISTORY_TURNS = 5  # 최근 5턴 초과 시 압축

def compress_history(history: list, llm) -> list:
    if len(history) <= MAX_HISTORY_TURNS * 2:
        return history  # 한도 이하면 그냥 반환

    old = history[:-(MAX_HISTORY_TURNS * 2)]     # 오래된 부분
    recent = history[-(MAX_HISTORY_TURNS * 2):]  # 최근 부분

    # 오래된 대화를 LLM으로 요약
    summary = (summary_prompt | llm).invoke({오래된 대화 텍스트}).content

    return [SystemMessage(content=f"[이전 대화 요약]\n{summary}")] + recent
```

### 중요한 점

**압축본은 DB에 저장하지 않는다.** DB에는 원본 전체가 보존되고, LLM에 넘기기 직전에만 압축.
→ `MAX_HISTORY_TURNS` 값을 나중에 바꿔도 원본 데이터 유지됨.

---

## SSE 스트리밍

### 문제

기존 구조는 일반 HTTP 요청/응답 방식 → 서버가 전부 처리 완료 후 한 번에 반환.
- 답변이 늦게 뜸 (첫 글자까지 LLM 전체 생성 시간 기다림)
- 압축 중인지 등 중간 상태를 알 수 없음

### SSE(Server-Sent Events)란

서버 → 클라이언트 단방향 실시간 전송 프로토콜.
채팅봇은 "질문 보내고 → 답변 받는" 구조라 단방향 SSE로 충분.
(WebSocket은 서버가 먼저 데이터를 push해야 하는 경우에 사용)

```
기존: 클라이언트 → 요청 → [서버 5초 처리] → 응답 (한 번에)

SSE:  클라이언트 → 요청 → 서버
                          ↓ "이전 대화를 압축하는 중..."  (즉시)
                          ↓ "K"                          (0.3초)
                          ↓ "8"
                          ↓ "s"
                          ↓ "는 ..."
```

총 처리 시간은 동일. **첫 글자가 빨리 뜨는 것**이 핵심 (체감 속도 향상).

### LLM이 스트리밍 가능한 이유

LLM은 "생각 후 출력"이 아니라 **다음 토큰 하나 예측 → 출력 → 반복** 방식.
앞에서부터 순서대로 확정되며 나오므로 첫 글자가 나중에 바뀔 일이 없음.
(추론 모델 o1, DeepSeek R1은 `<think>` 태그로 먼저 탐색 후 출력 → 스트리밍 체감 효과 낮음)

### 백엔드 구현 — FastAPI StreamingResponse

```python
@app.post("/chat-stream")
def chat_stream(req: ChatRequest):
    def event_generator():
        # 압축 필요 시 상태 먼저 전송
        if len(history) > MAX_HISTORY_TURNS * 2:
            yield f"data: {json.dumps({'type': 'status', 'content': '압축중...'})}\n\n"
            chat_history = compress_history(history, llm)

        # LLM 토큰 단위 스트리밍
        for chunk in (prompt | llm).stream({...}):
            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

SSE 메시지 형식: `data: <내용>\n\n` (개행 2개가 메시지 구분자)

### 프론트엔드 구현 — fetch + ReadableStream

EventSource는 GET만 지원하므로, POST가 필요한 경우 fetch로 스트림 직접 읽기.

```javascript
const res = await fetch("/chat-stream", { method: "POST", ... });
const reader = res.body.getReader();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // SSE 파싱 후 토큰을 bubble에 append
}
```

### 엔드포인트 정리

| 엔드포인트 | 방식 | 용도 |
|---|---|---|
| `POST /chat` | 일반 HTTP | (레거시) |
| `POST /chat-stream` | SSE 스트리밍 | 일반 모드 |
| `POST /chat-graph` | 일반 HTTP | (레거시) |
| `POST /chat-graph-stream` | SSE 스트리밍 | LangGraph 모드 |
| `POST /clear` | 일반 HTTP | 히스토리 초기화 |

---

## 마크다운 렌더링

### 문제

LLM은 답변을 `**굵게**`, ` ```코드블록``` `, `- 목록` 같은 마크다운 형식으로 출력하는 경우가 많은데,
기존 코드는 `\n`을 `<br>`로만 바꿔서 마크다운 문법이 그대로 텍스트로 보임.

### 해결 방법

marked.js 라이브러리를 CDN으로 불러와 `marked.parse()`로 렌더링.

```html
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
```

```javascript
marked.setOptions({ breaks: true, gfm: true });
// breaks: \n을 <br>로 변환
// gfm: GitHub Flavored Markdown (코드블록, 테이블 등)

// 스트리밍 중 토큰 받을 때마다 렌더링
bubble.innerHTML = marked.parse(fullText);
```

스트리밍 중에도 토큰이 추가될 때마다 `marked.parse()`를 호출해서 실시간으로 마크다운이 적용됨.

---

## 소스 문서 표시

### 목적

RAG는 벡터DB에서 관련 문서를 검색해서 답변을 생성하는데, 어떤 문서를 참고했는지 사용자에게 보여주면
답변의 신뢰성을 높이고 직접 확인할 수 있게 해줌.

### 구현 흐름

```
retriever.invoke(질문) → docs (Document 객체 리스트)
                           ↓
docs[i].metadata["source"] → 파일 경로 (예: "docs/msp_manual.txt")
                           ↓
os.path.basename() → 파일명만 추출 (예: "msp_manual.txt")
                           ↓
done 이벤트에 sources 포함해서 클라이언트 전송
```

### 백엔드 — sources 수집

```python
docs = retriever.invoke(req.question)
sources = list({os.path.basename(doc.metadata.get("source", ""))
                for doc in docs if doc.metadata.get("source")})

# done 이벤트에 포함
yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"
```

set으로 중복 제거 (같은 파일의 여러 청크가 검색될 수 있음).

### LangGraph 모드 — GraphState에 sources 추가

```python
class GraphState(TypedDict):
    ...
    sources: list  # retrieve_and_answer 노드에서 채움

# retrieve_and_answer 노드
sources = list({doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")})
return {**state, "answer": response.content, "sources": sources}

# reject 노드
return {**state, "answer": "MSP 운영과 관련 없는 질문입니다...", "sources": []}
```

`on_chain_end` 이벤트에서 `retrieve_and_answer` 노드의 output에서 sources 추출.

### 프론트엔드 — sources 표시

```javascript
} else if (data.type === "done") {
    bubble.innerHTML = marked.parse(fullText);
    if (data.sources && data.sources.length > 0) {
        const sourcesEl = document.createElement("div");
        sourcesEl.className = "sources";
        sourcesEl.innerHTML = `📄 참고: ${data.sources.map(s => `<span>${s}</span>`).join("")}`;
        div.appendChild(sourcesEl);
    }
}
```

답변 완료 후 버블 아래에 `📄 참고: msp_manual.txt` 형태로 표시.
관련 없는 질문(reject)은 sources가 빈 배열이라 표시되지 않음.

---

## 질문 로깅 및 통계

### 목적

어떤 질문이 자주 들어오는지 파악해 매뉴얼 보완 포인트를 식별.
별도 테이블 없이 기존 `chat_history` 테이블을 그대로 활용.

### 백엔드 — database.py

```python
def get_question_stats(limit: int = 20) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT content, COUNT(*) as count
            FROM chat_history
            WHERE role = 'human'
            GROUP BY content
            ORDER BY count DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [{"question": row[0], "count": row[1]} for row in rows]
```

`role = 'human'` 행만 골라 동일 질문 텍스트를 `GROUP BY`로 묶어 빈도 집계.

### 백엔드 — GET /stats 엔드포인트

```python
@app.get("/stats")
def stats():
    return {"questions": get_question_stats()}
```

### 프론트엔드

헤더의 📊 버튼 클릭 → `/stats` 호출 → 모달에 TOP 20 질문과 빈도 표시.
모달 바깥 클릭 또는 ✕ 버튼으로 닫기.

---

## 추천 검색어

### 동작 방식

페이지 로드 시 `/suggestions` 호출 → 입력창 위에 칩 형태로 표시.
클릭하면 해당 질문이 입력창에 채워지고 바로 전송.
첫 메시지 전송 후 칩 영역은 숨겨짐.

### 추천 순서

1. 통계 기반 자주 묻는 질문 최대 3개 (실사용 데이터 반영)
2. 사전 정의 질문으로 5개까지 채움

```python
PREDEFINED_SUGGESTIONS = [
    "MSP 서비스 제공 범위가 어떻게 되나요?",
    "PPP 네트워크 장애 시 확인 절차는?",
    "방화벽 정책 신청 방법을 알려주세요",
    "KTcloud 계정 접속 절차는?",
    "Kubernetes 플랫폼 지원 내용은?",
]

@app.get("/suggestions")
def suggestions():
    top = [s["question"] for s in get_question_stats(limit=3)]
    merged = list(dict.fromkeys(top + PREDEFINED_SUGGESTIONS))[:5]
    return {"suggestions": merged}
```

### 관련성 필터링

통계 기반 질문 중 거절되거나 매뉴얼에서 확인 불가 판정을 받은 질문은 제외.
LLM 호출 없이 DB에 저장된 AI 답변 텍스트를 SQL로 필터링.

```sql
SELECT h1.content, COUNT(*) as count
FROM chat_history h1
JOIN chat_history h2 ON h2.id = h1.id + 1 AND h2.role = 'ai'
WHERE h1.role = 'human'
  AND h2.content NOT LIKE '%MSP 운영과 관련 없는 질문%'
  AND h2.content NOT LIKE '%매뉴얼에서 확인이 어렵습니다%'
GROUP BY h1.content
ORDER BY count DESC
```

질문(h1)과 바로 다음 AI 답변(h2)을 JOIN해서, 부정적 답변이 달린 질문을 추천에서 제외.

---

## 멀티 세션 (ChatGPT 스타일 사이드바)

### 목적

하나의 챗봇에서 여러 주제의 대화를 독립적으로 관리.
ChatGPT처럼 왼쪽 사이드바에서 대화 목록을 보고 선택해 이어서 대화 가능.

### DB 추가 함수 — database.py

```python
def get_sessions() -> list:
    # 세션 목록 조회: 첫 질문을 제목으로, 최신 활동순 정렬
    # SQL 서브쿼리로 각 세션의 첫 human 메시지를 제목으로 추출

def get_full_history(session_id: str) -> list:
    # 특정 세션의 전체 대화 내역 반환 (role + content 딕셔너리 리스트)
```

### 백엔드 추가 엔드포인트 — app.py

```python
GET    /sessions               → 전체 세션 목록 (session_id, title, last_active)
GET    /sessions/{session_id}  → 특정 세션의 전체 히스토리
DELETE /sessions/{session_id}  → 세션 삭제
```

### 프론트엔드 구조 변경 — index.html

**레이아웃**
```
.app-container (900px)
├── .sidebar (250px, 다크 #1a1a2e)
│   ├── "+ 새 대화" 버튼
│   └── 세션 목록 (클릭 시 해당 세션으로 전환)
└── .chat-container
    └── 기존 채팅 UI
```

**핵심 JS 로직**

```javascript
let currentSessionId = "session_" + Math.random().toString(36).slice(2, 9);

// 세션 목록 불러와서 사이드바 렌더링
async function loadSessionList() { ... }

// 세션 선택 시 히스토리 불러와서 채팅창 복원
async function selectSession(sessionId) {
    currentSessionId = sessionId;
    const res = await fetch(`/sessions/${sessionId}`);
    const data = await res.json();
    data.history.forEach(msg => appendMessage(...));
}

// 새 대화 시작
function newChat() {
    currentSessionId = "session_" + Math.random().toString(36).slice(2, 9);
    // 채팅창 초기화 + 추천 검색어 다시 표시
}

// 메시지 전송 완료 후 사이드바 갱신
async function sendMessage() {
    await streamMessage(...);
    loadSessionList();  // ← 새 세션이 사이드바에 나타남
}
```

### 세션 제목 결정 방식

AI가 첫 번째 질문을 분석해 15글자 이내 제목 자동 생성 → `session_titles` 테이블에 저장.
AI 제목이 없는 기존 세션은 첫 질문 텍스트를 fallback으로 사용.

---

## 세션 삭제

사이드바 각 세션 항목에 ✕ 버튼 추가. 마우스 hover 시에만 표시.

### 백엔드

```python
DELETE /sessions/{session_id}
# database.py: delete_session() — chat_history + session_titles 동시 삭제
```

### 프론트엔드

```javascript
async function deleteSession(sessionId) {
    await fetch(`/sessions/${sessionId}`, { method: "DELETE" });
    if (sessionId === currentSessionId) newChat();  // 현재 세션이면 새 대화로 전환
    loadSessionList();
}
```

삭제 버튼 클릭 시 세션 선택 이벤트와 충돌하지 않도록 `e.stopPropagation()` 처리.

---

## AI 세션 제목 자동 생성

### 동작 방식

첫 메시지 전송 후, 답변 완료 시점에 같은 LLM으로 짧은 제목 생성.
두 번째 메시지부터는 생성하지 않음 (이미 저장된 제목 사용).

### DB — session_titles 테이블

```sql
CREATE TABLE session_titles (
    session_id TEXT PRIMARY KEY,
    title      TEXT NOT NULL
)
```

`get_sessions()`에서 `LEFT JOIN` + `COALESCE`로 AI 제목 우선, 없으면 첫 질문 텍스트 fallback.

### 제목 생성 프롬프트

```python
def generate_session_title(question: str, llm) -> str:
    # "사용자 질문을 채팅 목록에 표시할 짧은 제목으로 바꿔주세요. 15글자 이내, 제목만 출력"
    result = (title_prompt | llm).invoke({"question": question})
    return result.content.strip()[:20]
```

### 스트리밍 엔드포인트에서 호출

```python
is_first = len(history) == 0  # 첫 메시지 여부 미리 확인
# ... 스트리밍 완료 후 ...
save_messages(req.session_id, req.question, full_answer)
if is_first:
    title = generate_session_title(req.question, llm)
    save_session_title(req.session_id, title)
yield f"data: {json.dumps({'type': 'done', ...})}\n\n"
```

답변 스트리밍이 끝난 뒤 제목 생성 → 프론트에서 `loadSessionList()` 호출 시 AI 제목이 반영됨.

---

## 문서 업로드 / 삭제 / 목록

### 목적

지금까지는 문서를 추가하려면 `docs/` 폴더에 파일을 직접 넣고 `ingest.py`를 터미널에서 실행해야 했음.
웹 UI에서 바로 업로드/삭제할 수 있게 해서 서버 접근 없이 문서를 관리 가능하게 함.

### 지원 형식

`.txt`, `.pdf` (PyPDFLoader 사용)

### 백엔드 엔드포인트 — app.py

```python
GET    /documents              → 현재 벡터DB에 인덱싱된 문서 목록
POST   /documents              → 파일 업로드 → docs/ 저장 → 청크 분할 → vectorstore.add_documents()
DELETE /documents/{filename}   → 벡터DB에서 해당 파일 청크 전체 삭제 + 파일 삭제
```

### 업로드 흐름

```python
# 파일 저장
with open(os.path.join("docs", file.filename), "wb") as f:
    f.write(await file.read())

# 로더 선택
loader = PyPDFLoader(path) if ext == ".pdf" else TextLoader(path, encoding="utf-8")

# 청크 분할 후 기존 vectorstore에 추가 (서버 재시작 불필요)
chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50).split_documents(loader.load())
vectorstore.add_documents(chunks)
```

### 삭제 흐름

```python
# 메타데이터에서 source basename이 일치하는 청크 ID 수집
ids = [id for id, m in zip(ids, metadatas) if basename(m["source"]) == filename]
vectorstore._collection.delete(ids=ids)
os.remove(os.path.join("docs", filename))
```

### 프론트엔드 — 사이드바 하단 문서 관리 패널

- **📁 문서 관리** 토글 클릭 → 현재 인덱싱된 문서 목록 표시
- **+ txt / pdf 업로드** 버튼 → 파일 선택 → 업로드 중 표시 → 완료 메시지
- 각 문서 오른쪽 ✕ → 벡터DB + 파일 동시 삭제
- 서버 재시작 없이 실시간 반영

### 의존성 추가

```bash
pip install python-multipart  # FastAPI 파일 업로드에 필요
```

---

## API 오류 처리

### 문제

Groq 등 외부 API는 무료 티어에서 일일 토큰 한도(TPD)가 있어 초과 시 429 오류 발생.
기존 코드는 `event_generator()` 안에서 예외가 터지면 서버 에러 로그만 남고 클라이언트는 응답 없이 끊김.

### 해결 방법

`event_generator()` 전체를 try/except로 감싸고, 오류 발생 시 `type: error` SSE 이벤트 전송.

### 백엔드 — app.py

```python
def _api_error_message(e: Exception) -> str:
    msg = str(e)
    if "429" in msg or "rate_limit" in msg.lower():
        return "토큰 사용량 한도를 초과했습니다. 잠시 후 다시 시도하거나 다른 모델을 선택해주세요."
    if "401" in msg or "authentication" in msg.lower():
        return "API 인증에 실패했습니다. API 키를 확인해주세요."
    return "오류가 발생했습니다. 다시 시도해주세요."

# event_generator() 안
try:
    # ... 스트리밍 로직 ...
    yield done_event
    if is_first:
        try:
            save_session_title(...)   # 제목 생성 실패는 조용히 무시
        except Exception:
            pass
except Exception as e:
    logging.error("스트리밍 중 오류 [session=%s model=%s]: %s", ...)
    yield f"data: {json.dumps({'type': 'error', 'content': _api_error_message(e)})}\n\n"
```

- `done` 이벤트를 먼저 전송 후 제목 생성 → 제목 생성 실패가 사용자에게 노출되지 않음
- 오류 발생 시 `logging.error`로 서버 로그에도 기록 (`ERROR:root:스트리밍 중 오류 발생 [session=... model=...]: ...`)
- `/chat-stream`과 `/chat-graph-stream` 양쪽에 동일하게 적용

### 프론트엔드 — index.html

```javascript
if (data.type === "error") {
    bubble.innerHTML = `<span style="color:#ea4335">⚠️ ${data.content}</span>`;
}
```

채팅 버블에 빨간색으로 오류 메시지 표시.

### 출처 유사도 필터링

`similarity_search_with_relevance_scores()`로 점수를 받아 0.5 미만인 문서는 출처에서 제외.
"안녕" 같은 무관한 질문은 점수가 음수(-0.1 등)로 나와 출처가 표시되지 않음.
음수 점수에 대한 Chroma `UserWarning`은 `warnings.catch_warnings()`로 억제.

```python
SOURCE_SCORE_THRESHOLD = 0.5

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(req.question, k=4)

sources = list({os.path.basename(doc.metadata.get("source", ""))
                for doc, score in docs_with_scores
                if score >= SOURCE_SCORE_THRESHOLD and doc.metadata.get("source")})
```

---

## 스트리밍 중단 버튼

### 구현 방식

`AbortController`로 fetch 요청을 취소하면 클라이언트-서버 연결이 끊기고,
FastAPI의 스트리밍 제너레이터는 다음 `yield` 시점에 자동으로 멈춤. **백엔드 수정 불필요.**

```javascript
let abortController = null;

// 스트리밍 시작 시
abortController = new AbortController();
const res = await fetch(endpoint, {
    ...
    signal: abortController.signal,  // AbortController 연결
});

// 중단 버튼 클릭 시
function stopStreaming() {
    if (abortController) abortController.abort();
}

// AbortError는 오류가 아니라 의도적 취소 → 오류 메시지 미표시
} catch (e) {
    if (e.name !== "AbortError") {
        bubble.textContent = "서버에 연결할 수 없습니다.";
    }
}
```

### UX 흐름

```
스트리밍 시작 → 전송 버튼 비활성화 + 빨간 ■ 버튼 등장
■ 클릭       → fetch 즉시 취소, 서버 스트림 자동 중단
완료 또는 취소 → ■ 버튼 사라짐, 전송 버튼 복귀
```

---

## 👍/👎 피드백 기능

### 목적

사용자가 답변 품질을 직접 평가하게 해서, 어떤 답변이 부족했는지 파악하고 문서나 프롬프트 개선에 활용.

ChatGPT 같은 서비스는 피드백 데이터로 RLHF(인간 피드백 강화학습)를 수행하지만, 이 규모에서는 **👎 받은 질문/답변 목록을 사람이 직접 검토**해서 문서를 보강하는 방식이 실용적.

### DB — feedback 테이블

```sql
CREATE TABLE feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    answer     TEXT    NOT NULL,
    rating     INTEGER NOT NULL CHECK(rating IN (1, -1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

`rating`: 1 = 👍, -1 = 👎

### 백엔드 — app.py

```python
POST /feedback        → { session_id, question, answer, rating } → DB 저장
GET  /feedback/stats  → { total, positive, negative, recent[] }
```

`get_feedback_stats()`는 집계(total/positive/negative)와 최근 20건 목록을 함께 반환.

### 프론트엔드

스트리밍 완료(`done` 이벤트) 후, 오류가 아닌 정상 답변에만 버튼 추가.

```javascript
if (fullText) {
    const fbRow = document.createElement("div");
    fbRow.innerHTML = `
        <button class="btn-feedback" data-rating="1">👍</button>
        <button class="btn-feedback" data-rating="-1">👎</button>
    `;
    fbRow.querySelectorAll(".btn-feedback").forEach(btn => {
        btn.addEventListener("click", () =>
            submitFeedback(question, fullText, parseInt(btn.dataset.rating), fbRow));
    });
    div.appendChild(fbRow);
}
```

클릭 시 두 버튼 모두 비활성화 → 중복 제출 방지.
선택된 버튼만 색상 변경 (👍 초록 / 👎 빨강).

### 📊 통계 모달에서 활용

피드백 요약(👍 N / 👎 N) + **👎 받은 답변 목록**을 붉은 카드로 표시.
이 목록을 보고 관련 문서를 `docs/`에 추가하거나 프롬프트를 수정.

```
📊 모달 구성:
├── 피드백 요약: 👍 5 / 👎 2
├── 👎 받은 답변 카드 목록
│   ├── Q. 질문 텍스트
│   └── 답변 미리보기 (100자)
└── 자주 묻는 질문 TOP 20
```

---

## 생각 중 물결 애니메이션

### 목적

스트리밍 응답 대기 중 빈 버블이 노출되는 문제 개선. Gemini처럼 점 3개가 물결치는 애니메이션으로 "생각 중" 상태를 시각적으로 표현.

### 구현

첫 토큰이 도착하기 전까지 버블 안에 `.thinking` 요소를 표시하고, 토큰이 오는 순간 제거.

```javascript
// 초기 버블 생성 시
div.innerHTML = `<div class="bubble"><div class="thinking"><span></span><span></span><span></span></div></div>...`;

// 첫 토큰 수신 시
if (isFirstToken) { bubble.innerHTML = ""; isFirstToken = false; }
```

```css
@keyframes thinking-wave {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-6px); opacity: 1; }
}
.thinking span { animation: thinking-wave 1.4s ease-in-out infinite; }
.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }
```

---

## 답변 복사 / 모델 저장 / 대화 내보내기

### 답변 복사 버튼 (📋)

봇 버블에 마우스를 올리면 📋 버튼이 나타남. 클릭 시 마크다운 원문을 클립보드에 복사.
스트리밍 완료 후에는 👍/👎 버튼과 같은 행 오른쪽에 표시. 히스토리 로드 시에는 hover 시에만 보임.

```javascript
function copyToClipboard(btn, text) {
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = "✓";
    setTimeout(() => { btn.textContent = "📋"; }, 2000);
  });
}
```

### 모델 선택 localStorage 저장

페이지 새로고침해도 마지막 선택한 모델 유지.

```javascript
const savedModel = localStorage.getItem("selectedModel");
if (savedModel) modelSelect.value = savedModel;
modelSelect.addEventListener("change", () => {
  localStorage.setItem("selectedModel", modelSelect.value);
});
```

### 대화 내보내기 (⬇)

헤더 ⬇ 버튼 클릭 시 현재 세션 전체 대화를 `chat_날짜.txt`로 다운로드.
기존 `/sessions/{session_id}` API를 그대로 활용. 백엔드 변경 없음.

```javascript
async function exportChat() {
  const res = await fetch(`/sessions/${currentSessionId}`);
  const data = await res.json();
  const lines = ["=== MSP 운영 도우미 챗봇 대화 내보내기 ===", ...];
  const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
  // a 태그로 다운로드 트리거
}
```

### `/clear` 엔드포인트 제거

초기화 버튼과 `/clear` 엔드포인트를 제거. 세션 삭제(✕)와 새 대화로 동일한 기능을 대체할 수 있어 중복이었음.
