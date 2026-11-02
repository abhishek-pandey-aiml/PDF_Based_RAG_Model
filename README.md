# PDF-Based RAG Chatbot

## Overview

An LLM-powered PDF Question Answering application that allows users to upload PDF documents and ask questions based on their content. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate context-based answers.

## Features

* Upload PDF documents through a web interface.
* Extract text from PDF files using PyPDF.
* Split text into overlapping chunks.
* Generate embeddings using SentenceTransformers.
* Semantic search using FAISS.
* Context-based answer generation using Qwen 2.5.
* Interactive chatbot interface.

## Technologies Used

* Python
* FastAPI
* LangChain Ollama
* Qwen 2.5 (1.5B)
* SentenceTransformers
* FAISS
* PyPDF
* HTML, CSS & JavaScript

## RAG Workflow

PDF Upload → Text Extraction → Chunking → Embeddings → FAISS → Retrieval → Qwen LLM → Answer

## Installation

**1. Clone Repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Install and Run Ollama**

Download Ollama and pull the Qwen model:

```bash
ollama pull qwen2.5:1.5b
```

**4. Run Application**

```bash
python app.py
```

**5. Open in Browser**

`http://127.0.0.1:8001`

## Author

**Abhishek Pandey**

GitHub: https://github.com/abhishek-pandey-aiml
