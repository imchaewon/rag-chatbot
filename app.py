from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma

load_dotenv()

# 앱 시작 시 한 번만 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, prompt
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 친절한 고객서비스 챗봇입니다.
아래 문서 내용을 바탕으로만 답변하세요. 문서에 없는 내용은 '확인이 어렵습니다'라고 답하세요.

[참고 문서]
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    yield

app = FastAPI(title="RAG 챗봇 API", lifespan=lifespan)

# 세션별 대화 히스토리 저장 (메모리)
sessions: dict[str, list] = {}

def get_llm(model: str):
    if model == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    elif model == "solar":
        return ChatUpstage(model="solar-pro")
    else:
        return ChatGroq(model="llama-3.3-70b-versatile")


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    model: str = "groq"  # groq | gemini | solar


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class ClearRequest(BaseModel):
    session_id: str = "default"


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    chat_history = sessions.get(req.session_id, [])

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
    chat_history.append(HumanMessage(content=req.question))
    chat_history.append(AIMessage(content=answer))
    sessions[req.session_id] = chat_history

    return ChatResponse(answer=answer, session_id=req.session_id)


@app.post("/clear")
def clear(req: ClearRequest):
    sessions.pop(req.session_id, None)
    return {"message": f"세션 '{req.session_id}' 히스토리가 초기화되었습니다."}


@app.get("/health")
def health():
    return {"status": "ok"}
