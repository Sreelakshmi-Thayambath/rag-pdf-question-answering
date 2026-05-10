# PDF Question-Answering System using RAG

## Overview
This project is an end-to-end Retrieval-Augmented Generation (RAG) application for answering questions from PDF documents using local LLMs.

## Tech Stack
- Python
- FastAPI
- LangChain
- FAISS
- Ollama
- Streamlit
- HuggingFace Embeddings

## Features
- PDF ingestion
- Text chunking
- Vector embeddings
- Semantic search
- Local LLM inference
- Streamlit frontend

## Run Project

### Install dependencies
pip install -r requirements.txt

### Run FastAPI
uvicorn app:app --reload

### Run Streamlit
streamlit run frontend.py
