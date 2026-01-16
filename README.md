# 🎥 YouTube RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to ask
context-aware questions about any YouTube video using its transcript.
The system retrieves relevant transcript chunks using semantic search and
generates grounded, non-hallucinated answers with a large language model.

---

## 🚀 Project Overview

This project implements a production-style RAG pipeline:

1. Load YouTube video transcript
2. Split transcript into overlapping chunks
3. Generate semantic embeddings
4. Store embeddings in a vector database
5. Retrieve relevant context for a user query
6. Generate answers strictly from retrieved context

Embeddings and the vector store are built once per video, enabling multiple
questions per video efficiently.

---

## 🧠 Key Features

- Context-grounded Q&A using YouTube transcripts
- Hallucination reduction via strict prompt grounding
- Efficient architecture (no re-embedding per question)
- Supports multiple questions per video
- Clean CLI-based interaction

---

## 🛠️ Tech Stack

- Python
- LangChain
- Hugging Face
- FAISS (Vector Database)
- BAAI/bge-small-en-v1.5 (Embeddings)
- Gemma 2 (2B-IT) (LLM)
- YouTube Transcript Loader

---

## 📦 Setup Instructions

### 1) Clone the repository

git clone https://github.com/Xiao-1-1/YTBlud.git

cd YTBlud

---

### 2) (Optional) Create a virtual environment

python -m venv venv
source venv/bin/activate
Windows: venv\Scripts\activate

---

### 3) Install dependencies

pip install -r requirements.txt

---

## 🔐 Hugging Face Access Token Setup

This project uses the Hugging Face Inference API.

### Step 1: Generate a read-only access token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Read" access
4. Copy the generated token

---

### Step 2: Add token to .env file

Add it in the .env file:

HUGGINGFACEHUB_API_TOKEN=your_read_only_token_here

Do NOT commit the .env file to GitHub.

---

## ▶️ How to Run the Project

Start the application using:

python start.py

---

## 🧪 Usage Flow

1. Enter a YouTube video URL
2. Ask a question related to the video content
3. Ask multiple questions on the same video
4. Optionally switch to another video

---

## 📌 Usage Rules

- Ask only questions related to the video content
- Ask one question at a time
- If the transcript does not contain the answer, the system will say it does not know

---

## 🎯 Learning Outcomes

- RAG pipeline design and optimization
- Embedding normalization and similarity search
- Retriever architecture and chunking strategies
- Prompt grounding to reduce hallucinations
- LangChain runnable-based pipelines

---

## 📜 Author

Sahil Ranakoti
Built as a hands-on project to explore modern RAG systems and LLM application design.
