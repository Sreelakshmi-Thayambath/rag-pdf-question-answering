from fastapi import FastAPI
from pydantic import BaseModel


# LangChain Imports

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA


# FastAPI App

app = FastAPI(title="PDF RAG API")


# Load PDF

loader = PyPDFLoader("midterm_practice.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")


# Split into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Create FAISS Vector Store

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


# Retriever

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# Local LLM

llm = OllamaLLM(model="mistral")


# Build RAG Chain

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# Request Schema

class Query(BaseModel):
    question: str


# API Endpoint

@app.post("/ask")
def ask_question(query: Query):

    result = qa_chain.invoke({
        "query": query.question
    })

    return {
        "question": query.question,
        "answer": result["result"]
    }