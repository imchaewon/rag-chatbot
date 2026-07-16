from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

response = llm.invoke("안녕! 넌 누구야?")
print(response.content)
