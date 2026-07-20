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
