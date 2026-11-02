from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import shutil
import os

from rag_engine import create_index_from_pdf, rag_answer

app = FastAPI()

templates = Jinja2Templates(directory="templates")

INDEX = None
CHUNKS = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}    )

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global INDEX, CHUNKS

    os.makedirs("data", exist_ok=True)
    path = f"data/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    INDEX, CHUNKS = create_index_from_pdf(path)

    return {"message": "PDF uploaded successfully"}


@app.post("/ask")
def ask(question: str = Form(...)):
    if INDEX is None:
        return {"answer": "Upload PDF first"}

    ans = rag_answer(question, INDEX, CHUNKS)
    return {"answer": ans}

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
    