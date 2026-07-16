# 고객서비스 챗봇 개발 학습 노트

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
    ("system", "당신은 친절한 고객서비스 챗봇입니다."),  # 역할 지정
    ("human", "{question}"),                            # 사용자 질문
])
```

- `system`: LLM의 역할, 말투, 행동 방식 지정
- `human`: 실제 사용자 입력 자리 (`{변수명}` 형태)

### Chain (`|` 연산자)
데이터가 순서대로 흘러가는 파이프. Linux 파이프(`|`)와 동일한 개념.

```python
chain = prompt | llm
response = chain.invoke({"question": "환불 정책이 어떻게 되나요?"})
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
목표: 질문 → 문서 검색 → 검색결과 + 질문 → LLM → 실제 문서 기반 답변
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

### 문서 업데이트 시
문서가 바뀌면 `ingest.py` 재실행 → 벡터DB 갱신 필요.
주기적 자동화는 cron job 등으로 처리 가능.

---

## 4단계: 멀티턴 대화 (대화 히스토리)

### 문제
기존 코드는 매 질문을 독립적으로 처리해 이전 대화를 기억하지 못함.

```
고객: 환불 정책 알려줘
챗봇: 7일 이내 환불 가능합니다.

고객: 그럼 배송비는?   ← "그럼"이 뭘 가리키는지 모름
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
[system] 당신은 친절한 고객서비스 챗봇입니다...
[human]  환불 정책 알려줘          ← 1번째 질문 (히스토리)
[ai]     7일 이내 환불 가능합니다.  ← 1번째 답변 (히스토리)
[human]  배송비는 누가 내?         ← 2번째 질문 (히스토리)
[ai]     고객 부담입니다.          ← 2번째 답변 (히스토리)
[human]  영업시간은?               ← 현재 질문
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
  "question": "환불 정책이 어떻게 되나요?",
  "session_id": "user_001",
  "model": "groq"
}

// 응답
{
  "answer": "구매 후 30일 이내에 환불이 가능합니다...",
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
