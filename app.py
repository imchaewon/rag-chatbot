import asyncio
import logging
import os
import warnings
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import json
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import init_db, get_history, save_messages, delete_last_pair, get_question_stats, get_sessions, get_full_history, delete_session, save_session_title, pin_session, unpin_session, save_feedback, get_feedback_stats
from graph import build_graph

load_dotenv()

# 앱 시작 시 한 번만 로드
SOURCE_SCORE_THRESHOLD = 0.5  # 이 점수 미만인 문서는 출처에 표시하지 않음


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, vectorstore, prompt
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 MSP 운영팀의 운영 도우미입니다.
VM, Kubernetes, Solar Pro 등 운영 관련 질문에 답변합니다.
아래 매뉴얼 내용을 바탕으로만 답변하세요. 매뉴얼에 없는 내용은 '매뉴얼에서 확인이 어렵습니다'라고 답하세요.
반드시 한국어로만 답변하세요.

[참고 매뉴얼]
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    init_db()
    yield

app = FastAPI(title="RAG 챗봇 API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


MAX_HISTORY_TURNS = 5  # 최근 5턴(메시지 10개) 초과 시 압축


def compress_history(history: list, llm) -> list:
    if len(history) <= MAX_HISTORY_TURNS * 2:
        return history

    old = history[:-(MAX_HISTORY_TURNS * 2)]
    recent = history[-(MAX_HISTORY_TURNS * 2):]

    old_text = "\n".join([
        f"{'사용자' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in old
    ])
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 대화를 3~5문장으로 핵심만 요약하세요."),
        ("human", old_text),
    ])
    summary = (summary_prompt | llm).invoke({}).content

    return [SystemMessage(content=f"[이전 대화 요약]\n{summary}")] + recent


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


class ChatResponse(BaseModel):
    answer: str
    session_id: str



@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    llm = get_llm(req.model)
    chat_history = compress_history(get_history(req.session_id), llm)

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
    chat_history = compress_history(get_history(req.session_id), llm)
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
def chat_stream(req: ChatRequest):
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
            if len(history) > MAX_HISTORY_TURNS * 2:
                yield f"data: {json.dumps({'type': 'status', 'content': '이전 대화를 압축하는 중...'}, ensure_ascii=False)}\n\n"
                chat_history = compress_history(history, llm)
            else:
                chat_history = history

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                docs_with_scores = vectorstore.similarity_search_with_relevance_scores(req.question, k=4)
            docs = [doc for doc, _ in docs_with_scores]
            context = "\n".join([doc.page_content for doc in docs])
            sources = list({os.path.basename(doc.metadata.get("source", ""))
                            for doc, score in docs_with_scores
                            if score >= SOURCE_SCORE_THRESHOLD and doc.metadata.get("source")})

            full_answer = ""
            for chunk in (prompt | llm).stream({
                "context": context,
                "question": req.question,
                "chat_history": chat_history,
            }):
                token = chunk.content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            if not req.preview:
                save_messages(req.session_id, req.question, full_answer)
            yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"
            if is_first and not req.preview:
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
async def chat_graph_stream(req: ChatRequest):
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
            if len(history) > MAX_HISTORY_TURNS * 2:
                yield f"data: {json.dumps({'type': 'status', 'content': '이전 대화를 압축하는 중...'}, ensure_ascii=False)}\n\n"
                chat_history = await asyncio.to_thread(compress_history, history, llm)
            else:
                chat_history = history

            graph = build_graph(retriever, llm, vectorstore)
            initial_state = {
                "question": req.question,
                "context": "",
                "chat_history": chat_history,
                "answer": "",
                "relevant": "",
                "sources": [],
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
                    elif node == "reject":
                        output = event["data"].get("output", {})
                        full_answer = output.get("answer", "")
                        yield f"data: {json.dumps({'type': 'token', 'content': full_answer}, ensure_ascii=False)}\n\n"

            if not req.preview:
                await asyncio.to_thread(save_messages, req.session_id, req.question, full_answer)
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



@app.get("/sessions")
def list_sessions():
    return {"sessions": get_sessions()}


@app.get("/sessions/{session_id}")
def session_history(session_id: str):
    return {"history": get_full_history(session_id)}


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    return {"message": f"세션 '{session_id}'이 삭제되었습니다."}


class SaveRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    model: str = "ollama"


@app.post("/chat/save")
def save_chat_result(req: SaveRequest):
    is_first = len(get_history(req.session_id)) == 0
    save_messages(req.session_id, req.question, req.answer)
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
def update_session_title(session_id: str, req: TitleUpdateRequest):
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="제목은 비워둘 수 없습니다.")
    save_session_title(session_id, title[:30])
    return {"message": "제목이 업데이트되었습니다."}


@app.patch("/sessions/{session_id}/pin")
def toggle_pin(session_id: str):
    sessions = get_sessions()
    current = next((s for s in sessions if s["session_id"] == session_id), None)
    if current and current["pinned"]:
        unpin_session(session_id)
        return {"pinned": False}
    pin_session(session_id)
    return {"pinned": True}


@app.get("/stats")
def stats():
    return {"questions": get_question_stats()}


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


@app.get("/health")
def health():
    return {"status": "ok"}


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: int  # 1 = 좋아요, -1 = 싫어요


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating은 1 또는 -1이어야 합니다.")
    save_feedback(req.session_id, req.question, req.answer, req.rating)
    return {"message": "피드백이 저장되었습니다."}


@app.get("/feedback/stats")
def feedback_stats():
    return get_feedback_stats()


DOCS_DIR = "docs"
ALLOWED_EXTENSIONS = {".txt", ".pdf"}
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


@app.get("/documents")
def list_documents():
    results = vectorstore._collection.get(include=["metadatas"])
    sources = sorted({
        os.path.basename(m.get("source", ""))
        for m in results["metadatas"] if m.get("source")
    })
    return {"documents": sources}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
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
def delete_document(filename: str):
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