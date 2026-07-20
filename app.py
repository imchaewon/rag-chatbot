import asyncio
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
from database import init_db, get_history, save_messages, clear_history
from graph import build_graph

load_dotenv()

# 앱 시작 시 한 번만 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, prompt
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


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    model: str = "ollama"  # ollama | groq | gemini | solar


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class ClearRequest(BaseModel):
    session_id: str = "default"


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
    graph = build_graph(retriever, llm)

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

        if len(history) > MAX_HISTORY_TURNS * 2:
            yield f"data: {json.dumps({'type': 'status', 'content': '이전 대화를 압축하는 중...'}, ensure_ascii=False)}\n\n"
            chat_history = compress_history(history, llm)
        else:
            chat_history = history

        docs = retriever.invoke(req.question)
        context = "\n".join([doc.page_content for doc in docs])
        sources = list({os.path.basename(doc.metadata.get("source", "")) for doc in docs if doc.metadata.get("source")})

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

        save_messages(req.session_id, req.question, full_answer)
        yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"

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

        if len(history) > MAX_HISTORY_TURNS * 2:
            yield f"data: {json.dumps({'type': 'status', 'content': '이전 대화를 압축하는 중...'}, ensure_ascii=False)}\n\n"
            chat_history = await asyncio.to_thread(compress_history, history, llm)
        else:
            chat_history = history

        graph = build_graph(retriever, llm)
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

        await asyncio.to_thread(save_messages, req.session_id, req.question, full_answer)
        yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/clear")
def clear(req: ClearRequest):
    clear_history(req.session_id)
    return {"message": f"세션 '{req.session_id}' 히스토리가 초기화되었습니다."}


@app.get("/health")
def health():
    return {"status": "ok"}