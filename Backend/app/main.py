from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import ingestRoute,queryRoute

server = FastAPI(
    title= "Codebase Q&A Assistant API",
    version="1.0.0"
)

server.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

server.include_router(ingestRoute.router,tags=["Ingestion"])
server.include_router(queryRoute.router,tags=["Chatbot"])

@server.get("/")
def root():
    return {"message":"Codebase Q&A Assistant backend running successfully."}










