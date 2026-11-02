import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_ollama import ChatOllama

# ---- LLM --------

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0)


# ---  EMBEDDING MODEL
embed_model = SentenceTransformer("all-MiniLM-L6-v2")



# --- PDF to Text --------

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

#  ---- Chunking -----

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ----- Vectore Database ---------
def create_index_from_pdf(pdf_path):
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)

    embeddings = embed_model.encode(chunks, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks

#  --- Retrieve only Relevent Chunks -------
def retrieve_chunks(question, index, chunks, top_k=3):
    query_vec = embed_model.encode([question], convert_to_numpy=True)
    distance, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]


# --- LLM Generated Answer ----------
def rag_answer(question, index, chunks):
    retrieved = retrieve_chunks(question, index, chunks)

    context = "\n\n".join(retrieved)

    prompt = f"""
Answer ONLY from the provided context.

If the answer is not present in the context, reply exactly:
Answer not found in the document.

Context:
{context}

Question:
{question}
"""
    answer= llm.invoke(prompt).content
    return answer
