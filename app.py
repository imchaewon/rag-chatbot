from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from database import init_db, get_history, save_messages, clear_history

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


def get_llm(model: str):
    if model == "groq":
        return ChatGroq(model="llama-3.3-70b-versatile")
    elif model == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    elif model == "solar":
        return ChatUpstage(model="solar-pro")
    elif model == "llama":
        return ChatOllama(model="llama3.1:8b")
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

    chat_history = get_history(req.session_id)

    docs = retriever.invoke(req.question)
    context = "\n".join([doc.page_content for doc in docs])

    llm = get_llm(req.model)
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": req.question,
        "chat_history": chat_history,
    })

    answer = response.content
    save_messages(req.session_id, req.question, answer)

    return ChatResponse(answer=answer, session_id=req.session_id)


@app.post("/clear")
def clear(req: ClearRequest):
    clear_history(req.session_id)
    return {"message": f"세션 '{req.session_id}' 히스토리가 초기화되었습니다."}


@app.get("/health")
def health():
    return {"status": "ok"}
