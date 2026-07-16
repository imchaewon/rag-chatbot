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

## 다음 단계
- 4단계: 대화 히스토리 (멀티턴 대화)
- 5단계: FastAPI 백엔드
- 6단계: 배포
