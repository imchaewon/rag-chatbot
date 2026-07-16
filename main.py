from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma

load_dotenv()

# 1. 저장된 벡터DB 불러오기
embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# 2. 프롬프트 (대화 히스토리 포함)
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절한 고객서비스 챗봇입니다.
아래 문서 내용을 바탕으로만 답변하세요. 문서에 없는 내용은 '확인이 어렵습니다'라고 답하세요.

[참고 문서]
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def select_llm():
    print("\n사용할 AI를 선택하세요.")
    print("1. Groq (llama-3.3-70b)")
    print("2. Gemini (gemini-2.0-flash)")
    print("3. Solar (solar-pro)")
    choice = input("선택 (1 / 2 / 3): ").strip()
    if choice == "2":
        print("Gemini로 변경됩니다.")
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    elif choice == "3":
        print("Solar로 변경됩니다.")
        return ChatUpstage(model="solar-pro")
    else:
        print("Groq로 변경됩니다.")
        return ChatGroq(model="llama-3.3-70b-versatile")

# 3. 초기 LLM 선택
llm = select_llm()

# 4. 대화 루프
chat_history = []

print("-" * 40)
print("질문을 입력하세요. (종료: q | 모델변경: /model | 히스토리초기화: /clear)")

while True:
    question = input("고객: ").strip()

    if question.lower() == "q":
        print("챗봇을 종료합니다.")
        break

    if question.lower() == "/model":
        llm = select_llm()
        chat_history = []
        print("대화 히스토리가 초기화되었습니다.")
        continue

    if question.lower() == "/clear":
        chat_history = []
        print("대화 히스토리가 초기화되었습니다.")
        continue

    if not question:
        continue

    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history,
    })

    answer = response.content
    print(f"챗봇: {answer}")
    print()

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
