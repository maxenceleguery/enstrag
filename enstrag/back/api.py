from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from dataclasses import asdict
from ..data import FileDocument
from ..rag import RagAgent

def build_server(agent: RagAgent):
    app: FastAPI = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "Enstrag API"}
    
    @app.get("/answer")
    async def answer_question(query: str, topk_context: int = 4):

        result, retrieved_context, sources, (pdf_path, pdf_name, context_to_highlight) = agent.answer_question(query, topk_context)
        return {
            "result": result,
            "retrieved_context": retrieved_context,
            "sources": sources,
            "pdf_path": pdf_path,
            "pdf_name": pdf_name,
            "context_to_highlight": context_to_highlight,
        }
    
    @app.get("/themes")
    async def get_themes():
        return {
            "themes": agent.get_themes(),
        }
    
    @app.post("/doc", response_model=FileDocument)
    async def add_document(filedoc: FileDocument):
        print(f"Receiving {filedoc}")
        agent.add_filedoc(filedoc)
        return filedoc
    
    @app.get("/documents", response_model=List[FileDocument])
    async def get_docs():
        docs = agent.get_docs()
        return docs
    
    @app.get("/topk", response_model=List[str])
    async def top_k_tokens(context: str, question: str, k: int, method: str):
        tokens_str = agent.top_k_tokens({"context": context, "question": question}, k, method)
        print("Sending : ", tokens_str)
        return tokens_str
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust if needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app