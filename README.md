# 🔍 HyPE-RAG: Retrieval-Augmented Generation with Hypothetical Prompts

This project implements a RAG (Retrieval-Augmented Generation) pipeline enhanced with HyPE (Hypothetical Prompt Embeddings). It enables accurate question answering over PDF documents by generating and indexing hypothetical questions that improve alignment between user queries and document content.

# 📌 Features
✅ PDF Document Parsing with automatic chunking

✅ Hypothetical Question Generation using a powerful LLM (Groq + Qwen)

✅ Embedding via Hugging Face for dense vector representation

✅ FAISS-based Vector Store for high-performance semantic retrieval

✅ RetrievalQA Pipeline for end-to-end question answering

✅ Multithreaded Embedding Pipeline for faster indexing

# 🧠 How It Works

1. PDF Encoding

- Loads a PDF file

- Cleans and splits the content into text chunks

- For each chunk:

    - Uses an LLM to generate multiple hypothetical questions

    - Embeds these questions using a Hugging Face model

- Stores the chunk-question pairs in a FAISS vector index

2. Question Answering

- Takes a user query

- Finds the top-K most relevant text chunks by comparing to embedded hypothetical questions

- Feeds these chunks and the query to a language model (Groq)

- Outputs a final answer


# 🛠️ Installation

1. Clone the repository
```bash
git clone https://github.com/tinyzqh/hype-rag.git
cd hype-rag
```

2. Create a virtual environment
```bash
conda create -n hpde-rag python=3.11
conda activate hpde-rag
```

3. Install dependencies
```bash
pip install -r requirements.txt
```