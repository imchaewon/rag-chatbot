import asyncio
import logging
import os
import warnings
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import init_db, get_history, save_messages, delete_last_pair, get_question_stats, get_sessions, get_full_history, delete_session, save_session_title, pin_session, unpin_session, save_feedback, get_feedback_stats, get_summary, save_summary, get_messages_after, create_user, get_user_by_username, set_session_owner, verify_session_owner
from auth import hash_password, verify_password, create_token, decode_token
from graph import build_graph

load_dotenv()

# 앱 시작 시 한 번만 로드
SOURCE_SCORE_THRESHOLD = 0.3  # bge-m3 기준 최고 점수가 0.5 수준이므로 0.3으로 설정


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, vectorstore, prompt
    embeddings = OllamaEmbeddings(model="bge-m3")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """IMPORTANT: You MUST respond ONLY in Korean (한국어). Never use Japanese, Chinese, or any other language. Korean only.

당신은 MSP 운영팀의 운영 도우미입니다.
VM, Kubernetes, Solar Pro 등 운영 관련 질문에 답변합니다.
아래 매뉴얼 내용을 바탕으로만 답변하세요. 매뉴얼에 없는 내용은 '매뉴얼에서 확인이 어렵습니다'라고 답하세요.

[참고 매뉴얼]
{context}

반드시 한국어로만 답변하세요. 절대 일본어, 중국어, 영어를 사용하지 마세요."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    init_db()
    yield

app = FastAPI(title="RAG 챗봇 API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

_bearer = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    try:
        payload = decode_token(credentials.credentials)
        return {"user_id": int(payload["sub"]), "username": payload["username"]}
    except Exception:
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")


class AuthRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/register")
def register(req: AuthRequest):
    if len(req.username) < 2:
        raise HTTPException(status_code=400, detail="사용자 이름은 2자 이상이어야 합니다.")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="비밀번호는 4자 이상이어야 합니다.")
    if get_user_by_username(req.username):
        raise HTTPException(status_code=409, detail="이미 사용 중인 사용자 이름입니다.")
    user_id = create_user(req.username, hash_password(req.password))
    token = create_token(user_id, req.username)
    return {"token": token, "username": req.username}


@app.post("/auth/login")
def login(req: AuthRequest):
    user = get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="사용자 이름 또는 비밀번호가 올바르지 않습니다.")
    token = create_token(user["id"], user["username"])
    return {"token": token, "username": user["username"]}


@app.get("/")
def index():
    return FileResponse("static/index.html")


COMPRESS_TRIGGER_TURNS = 5  # 이 턴 수 이상이면 압축 트리거 (5턴 = 10메시지)
COMPRESS_KEEP_TURNS = 2     # 압축 후 풀 텍스트로 유지할 턴 수 → 3번 질문마다 한 번 압축


def get_compressed_history(session_id: str, llm) -> tuple[list, bool]:
    """캐시된 요약을 활용해 LLM 재요약을 최소화.
    반환: (chat_history 리스트, 이번에 실제로 LLM 요약을 호출했는지 여부)
    """
    saved = get_summary(session_id)
    after_id = saved["summarized_up_to_id"] if saved else 0

    rows = get_messages_after(session_id, after_id)  # [(db_id, msg), ...]
    messages = [msg for _, msg in rows]

    trigger = COMPRESS_TRIGGER_TURNS * 2  # 10
    keep = COMPRESS_KEEP_TURNS * 2        # 4

    if len(messages) < trigger:
        # 창 안에 들어옴 → LLM 호출 없이 캐시 요약 + 최근 메시지 반환
        if saved:
            return [SystemMessage(content=f"[이전 대화 요약]\n{saved['summary']}")] + messages, False
        return messages, False

    # 창 밖으로 새 메시지가 밀려남 → 재요약 필요
    old_rows = rows[:-keep]
    recent_msgs = [msg for _, msg in rows[-keep:]]

    old_text = "\n".join([
        f"{'사용자' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for _, msg in old_rows
    ])
    if saved:
        text_to_summarize = f"[기존 요약]\n{saved['summary']}\n\n[추가 대화]\n{old_text}"
    else:
        text_to_summarize = old_text

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 대화 내용을 3~5문장으로 핵심만 요약하세요."),
        ("human", "{text}"),
    ])
    new_summary = (summary_prompt | llm).invoke({"text": text_to_summarize}).content
    new_up_to_id = old_rows[-1][0]
    save_summary(session_id, new_summary, new_up_to_id)

    return [SystemMessage(content=f"[이전 대화 요약]\n{new_summary}")] + recent_msgs, True


def get_llm(model: str):
    if model == "groq":
        return ChatGroq(model="llama-3.3-70b-versatile")
    elif model == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    elif model == "solar":
        return ChatUpstage(model="solar-pro")
    elif model == "llama":
        return ChatOllama(model="llama3.1:8b")
    elif model == "qwen3b":
        return ChatOllama(model="qwen2.5:3b")
    else:
        return ChatOllama(model="qwen2.5:7b")


def _api_error_message(e: Exception) -> str:
    msg = str(e)
    if "429" in msg or "rate_limit" in msg.lower():
        return "토큰 사용량 한도를 초과했습니다. 잠시 후 다시 시도하거나 다른 모델을 선택해주세요."
    if "401" in msg or "authentication" in msg.lower():
        return "API 인증에 실패했습니다. API 키를 확인해주세요."
    return "오류가 발생했습니다. 다시 시도해주세요."


def generate_session_title(question: str, llm) -> str:
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", """아래 질문을 채팅 목록에 표시할 제목으로 만드세요.
출력 규칙:
- 제목 텍스트만 출력, 다른 말 절대 금지
- 10글자 이내
예시: "VM 재시작 절차", "K8s Pod 오류", "인사말", "방화벽 정책 신청"
질문:"""),
        ("human", "{question}"),
    ])
    result = (title_prompt | llm).invoke({"question": question})
    title = result.content.strip().splitlines()[0]
    title = title.strip("\"'.,。·· ")
    return title[:15]


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    model: str = "ollama"  # ollama | groq | gemini | solar
    regenerate: bool = False
    preview: bool = False  # True면 DB 저장 안 함 (비교 모드용)
    score_threshold: float = SOURCE_SCORE_THRESHOLD


class ChatResponse(BaseModel):
    answer: str
    session_id: str



@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    llm = get_llm(req.model)
    chat_history, _ = get_compressed_history(req.session_id, llm)

    docs = retriever.invoke(req.question)
    context = "\n".join([doc.page_content for doc in docs])
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": req.question,
        "chat_history": chat_history,
    })

    answer = response.content
    save_messages(req.session_id, req.question, answer)

    return ChatResponse(answer=answer, session_id=req.session_id)


@app.post("/chat-graph", response_model=ChatResponse)
def chat_graph(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    llm = get_llm(req.model)
    chat_history, _ = get_compressed_history(req.session_id, llm)
    graph = build_graph(retriever, llm, vectorstore)

    result = graph.invoke({
        "question": req.question,
        "context": "",
        "chat_history": chat_history,
        "answer": "",
        "relevant": "",
    })

    answer = result["answer"]
    save_messages(req.session_id, req.question, answer)

    return ChatResponse(answer=answer, session_id=req.session_id)


@app.post("/chat-stream")
def chat_stream(req: ChatRequest, user=Depends(get_current_user)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    def event_generator():
        llm = get_llm(req.model)
        history = get_history(req.session_id)

        if req.regenerate and len(history) >= 2:
            delete_last_pair(req.session_id)
            history = history[:-2]

        is_first = not req.regenerate and len(history) == 0

        try:
            chat_history, did_compress = get_compressed_history(req.session_id, llm)
            if did_compress:
                yield f"data: {json.dumps({'type': 'status', 'content': '이전 대화를 압축하는 중...'}, ensure_ascii=False)}\n\n"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                docs_with_scores = vectorstore.similarity_search_with_relevance_scores(req.question, k=4)
            docs = [doc for doc, score in docs_with_scores if score >= req.score_threshold]
            sources = list({os.path.basename(doc.metadata.get("source", ""))
                            for doc, score in docs_with_scores
                            if score >= req.score_threshold and doc.metadata.get("source")})

            full_answer = ""
            if not docs:
                full_answer = "매뉴얼에서 확인이 어렵습니다."
                yield f"data: {json.dumps({'type': 'token', 'content': full_answer}, ensure_ascii=False)}\n\n"
            else:
                context = "\n".join([doc.page_content for doc in docs])
                for chunk in (prompt | llm).stream({
                    "context": context,
                    "question": req.question,
                    "chat_history": chat_history,
                }):
                    token = chunk.content
                    if token:
                        full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            if not req.preview and docs:
                save_messages(req.session_id, req.question, full_answer)
                set_session_owner(req.session_id, user["user_id"])
            yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"
            if is_first and not req.preview and docs:
                try:
                    save_session_title(req.session_id, generate_session_title(req.question, llm))
                except Exception:
                    pass

        except Exception as e:
            logging.error("스트리밍 중 오류 발생 [session=%s model=%s]: %s", req.session_id, req.model, e)
            yield f"data: {json.dumps({'type': 'error', 'content': _api_error_message(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat-graph-stream")
async def chat_graph_stream(req: ChatRequest, user=Depends(get_current_user)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    async def event_generator():
        llm = get_llm(req.model)
        history = await asyncio.to_thread(get_history, req.session_id)

        if req.regenerate and len(history) >= 2:
            await asyncio.to_thread(delete_last_pair, req.session_id)
            history = history[:-2]

        is_first = not req.regenerate and len(history) == 0

        try:
            chat_history, did_compress = await asyncio.to_thread(get_compressed_history, req.session_id, llm)
            if did_compress:
                yield f"data: {json.dumps({'type': 'status', 'content': '이전 대화를 압축하는 중...'}, ensure_ascii=False)}\n\n"

            graph = build_graph(retriever, llm, vectorstore)
            initial_state = {
                "question": req.question,
                "context": "",
                "chat_history": chat_history,
                "answer": "",
                "relevant": "",
                "sources": [],
                "score_threshold": req.score_threshold,
                "intent": "",
                "k8s_action": "",
                "k8s_target": "",
                "k8s_namespace": "",
            }

            full_answer = ""
            sources = []

            async for event in graph.astream_events(initial_state, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    node = event.get("metadata", {}).get("langgraph_node", "")
                    if node == "retrieve_and_answer":
                        token = event["data"]["chunk"].content
                        if token:
                            full_answer += token
                            yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end":
                    node = event.get("metadata", {}).get("langgraph_node", "")
                    if node == "retrieve_and_answer":
                        output = event["data"].get("output", {})
                        sources = [os.path.basename(s) for s in output.get("sources", []) if s]
                    elif node in ("reject", "execute_k8s"):
                        output = event["data"].get("output", {})
                        full_answer = output.get("answer", "")
                        yield f"data: {json.dumps({'type': 'token', 'content': full_answer}, ensure_ascii=False)}\n\n"

            if not req.preview:
                await asyncio.to_thread(save_messages, req.session_id, req.question, full_answer)
                await asyncio.to_thread(set_session_owner, req.session_id, user["user_id"])
            yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"
            if is_first and not req.preview:
                try:
                    title = await asyncio.to_thread(generate_session_title, req.question, llm)
                    await asyncio.to_thread(save_session_title, req.session_id, title)
                except Exception:
                    pass

        except Exception as e:
            logging.error("스트리밍 중 오류 발생 [session=%s model=%s]: %s", req.session_id, req.model, e)
            yield f"data: {json.dumps({'type': 'error', 'content': _api_error_message(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat-compare-stream")
async def chat_compare_stream(req: ChatRequest, user=Depends(get_current_user)):
    async def event_generator():
        history = await asyncio.to_thread(get_history, req.session_id)
        docs_with_scores = await asyncio.to_thread(
            vectorstore.similarity_search_with_relevance_scores, req.question, k=4
        )
        docs = [doc for doc, score in docs_with_scores if score >= req.score_threshold]
        sources = list({
            os.path.basename(doc.metadata.get("source", ""))
            for doc, score in docs_with_scores
            if score >= req.score_threshold and doc.metadata.get("source")
        })

        if not docs:
            no_answer = "매뉴얼에서 확인이 어렵습니다."
            yield f"data: {json.dumps({'type': 'token_a', 'content': no_answer}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'token_b', 'content': no_answer}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': []}, ensure_ascii=False)}\n\n"
            return

        context = "\n".join([doc.page_content for doc in docs])
        chain_input = {"context": context, "question": req.question, "chat_history": history}

        combined: asyncio.Queue = asyncio.Queue()

        async def relay(label: str):
            try:
                async for chunk in (prompt | get_llm(req.model)).astream(chain_input):
                    if chunk.content:
                        await combined.put((label, chunk.content))
            except Exception as e:
                await combined.put((label, {"error": _api_error_message(e)}))
            finally:
                await combined.put((label, None))

        task_a = asyncio.create_task(relay("a"))
        task_b = asyncio.create_task(relay("b"))

        done = 0
        while done < 2:
            label, token = await combined.get()
            if token is None:
                done += 1
            elif isinstance(token, dict):
                yield f"data: {json.dumps({'type': f'error_{label}', 'content': token['error']}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'type': f'token_{label}', 'content': token}, ensure_ascii=False)}\n\n"

        await asyncio.gather(task_a, task_b, return_exceptions=True)
        yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions")
def list_sessions(user=Depends(get_current_user)):
    return {"sessions": get_sessions(user["user_id"])}


@app.get("/sessions/{session_id}")
def session_history(session_id: str, user=Depends(get_current_user)):
    if not verify_session_owner(session_id, user["user_id"]):
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    return {"history": get_full_history(session_id)}


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str, user=Depends(get_current_user)):
    if not verify_session_owner(session_id, user["user_id"]):
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    delete_session(session_id)
    return {"message": f"세션 '{session_id}'이 삭제되었습니다."}


class SaveRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    model: str = "ollama"


@app.post("/chat/save")
def save_chat_result(req: SaveRequest, user=Depends(get_current_user)):
    is_first = len(get_history(req.session_id)) == 0
    save_messages(req.session_id, req.question, req.answer)
    set_session_owner(req.session_id, user["user_id"])
    if is_first:
        try:
            llm = get_llm(req.model)
            save_session_title(req.session_id, generate_session_title(req.question, llm))
        except Exception:
            pass
    return {"message": "저장 완료"}


class TitleUpdateRequest(BaseModel):
    title: str


@app.patch("/sessions/{session_id}/title")
def update_session_title(session_id: str, req: TitleUpdateRequest, user=Depends(get_current_user)):
    if not verify_session_owner(session_id, user["user_id"]):
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="제목은 비워둘 수 없습니다.")
    save_session_title(session_id, title[:30])
    return {"message": "제목이 업데이트되었습니다."}


@app.patch("/sessions/{session_id}/pin")
def toggle_pin(session_id: str, user=Depends(get_current_user)):
    if not verify_session_owner(session_id, user["user_id"]):
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    sessions = get_sessions(user["user_id"])
    current = next((s for s in sessions if s["session_id"] == session_id), None)
    if current and current["pinned"]:
        unpin_session(session_id)
        return {"pinned": False}
    pin_session(session_id)
    return {"pinned": True}


@app.get("/stats")
def stats(user=Depends(get_current_user)):
    return {"questions": get_question_stats()}


PREDEFINED_SUGGESTIONS = [
    "MSP 서비스 제공 범위가 어떻게 되나요?",
    "PPP 네트워크 장애 시 확인 절차는?",
    "방화벽 정책 신청 방법을 알려주세요",
    "KTcloud 계정 접속 절차는?",
    "Kubernetes 플랫폼 지원 내용은?",
]


@app.get("/suggestions")
def suggestions(user=Depends(get_current_user)):
    top = [s["question"] for s in get_question_stats(limit=3)]
    merged = list(dict.fromkeys(top + PREDEFINED_SUGGESTIONS))[:5]
    return {"suggestions": merged}


@app.get("/health")
def health():
    return {"status": "ok"}


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: int  # 1 = 좋아요, -1 = 싫어요


@app.post("/feedback")
def feedback(req: FeedbackRequest, user=Depends(get_current_user)):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating은 1 또는 -1이어야 합니다.")
    save_feedback(req.session_id, req.question, req.answer, req.rating)
    return {"message": "피드백이 저장되었습니다."}


@app.get("/feedback/stats")
def feedback_stats(user=Depends(get_current_user)):
    return get_feedback_stats()


DOCS_DIR = "docs"
ALLOWED_EXTENSIONS = {".txt", ".pdf"}
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


@app.get("/documents")
def list_documents(user=Depends(get_current_user)):
    results = vectorstore._collection.get(include=["metadatas"])
    sources = sorted({
        os.path.basename(m.get("source", ""))
        for m in results["metadatas"] if m.get("source")
    })
    return {"documents": sources}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...), user=Depends(get_current_user)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="txt 또는 pdf 파일만 업로드 가능합니다.")

    save_path = os.path.join(DOCS_DIR, file.filename)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    loader = PyPDFLoader(save_path) if ext == ".pdf" else TextLoader(save_path, encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(loader.load())
    vectorstore.add_documents(chunks)

    return {"message": f"'{file.filename}' 인덱싱 완료", "chunks": len(chunks)}


@app.delete("/documents/{filename}")
def delete_document(filename: str, user=Depends(get_current_user)):
    all_results = vectorstore._collection.get(include=["metadatas"])
    ids_to_delete = [
        all_results["ids"][i]
        for i, m in enumerate(all_results["metadatas"])
        if os.path.basename(m.get("source", "")) == filename
    ]
    if ids_to_delete:
        vectorstore._collection.delete(ids=ids_to_delete)

    file_path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"message": f"'{filename}' 삭제 완료", "chunks_removed": len(ids_to_delete)}