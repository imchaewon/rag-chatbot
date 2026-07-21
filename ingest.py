from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# 1. 문서 로딩
loader = TextLoader("docs/msp_manual.txt", encoding="utf-8")
documents = loader.load()

# 2. 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. 임베딩 & 벡터DB 영구 저장
embeddings = OllamaEmbeddings(model="bge-m3")
Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

print("벡터DB 저장 완료!")
