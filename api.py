"""FastAPI service for workspace-isolated RAG."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env", override=False)

from rag_core import hybrid_retrieve, workspace_filter

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "").strip()
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "corrective_rag_v3").strip()
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() in {"1", "true", "yes"}
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() in {"1", "true", "yes"}
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

app = FastAPI(title="Enterprise RAG API", version="0.4.0", description="Workspace-isolated RAG with DeepSeek, BGE-M3, and Qdrant.")


class Principal(BaseModel):
    username: str
    workspace_id: str
    api_key: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    top_k: int = Field(default=RERANK_TOP_N, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    workspace_id: str
    trace_id: str
    elapsed_ms: float


class IngestResponse(BaseModel):
    file_name: str
    workspace_id: str
    chunks: int
    collection: str


def principal_map() -> dict[str, Principal]:
    mapping: dict[str, Principal] = {}
    demo_key = os.getenv("DEMO_API_KEY", "").strip()
    analyst_key = os.getenv("ANALYST_API_KEY", "").strip()
    if demo_key:
        mapping[demo_key] = Principal(username="demo", workspace_id="demo_workspace", api_key=demo_key)
    if analyst_key:
        mapping[analyst_key] = Principal(username="analyst", workspace_id="analyst_workspace", api_key=analyst_key)
    return mapping


def require_principal(x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None) -> Principal:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    principal = principal_map().get(x_api_key)
    if principal is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return principal


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, temperature=0, max_tokens=1000, timeout=90, max_retries=2)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL, chunk_size=32, timeout=90, max_retries=2, tiktoken_enabled=False, check_embedding_ctx_length=False)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=15)


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(collection_name=QDRANT_COLLECTION, vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE))


def source_label(document) -> str:
    name = document.metadata.get("file_name", "unknown")
    page = document.metadata.get("page")
    if page is None:
        return str(name)
    try:
        return f"{name} p.{int(page) + 1}"
    except (TypeError, ValueError):
        return str(name)


def load_uploaded_file(path: Path, file_name: str):
    extension = path.suffix.lower()
    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
    elif extension in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")
    documents = loader.load()
    for document in documents:
        document.metadata["file_name"] = file_name
        document.metadata["source"] = file_name
    return documents


def ingest_documents(documents, *, workspace_id: str, user_id: str, file_name: str) -> int:
    for document in documents:
        document.metadata["workspace_id"] = workspace_id
        document.metadata["user_id"] = user_id
        document.metadata["file_name"] = file_name
        document.metadata["source"] = file_name
    splits = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60, separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]).split_documents(documents)
    client = get_qdrant_client()
    ensure_collection(client)
    source_filter = Filter(must=[FieldCondition(key="metadata.workspace_id", match=MatchValue(value=workspace_id)), FieldCondition(key="metadata.file_name", match=MatchValue(value=file_name))])
    try:
        client.delete(collection_name=QDRANT_COLLECTION, points_selector=source_filter)
    except Exception:
        pass
    store = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=get_embeddings())
    store.add_documents(splits)
    return len(splits)


def retrieve_documents(question: str, workspace_id: str, top_k: int):
    client = get_qdrant_client()
    store = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=get_embeddings())
    retrieval_k = RETRIEVAL_TOP_K if (ENABLE_HYBRID_SEARCH or ENABLE_RERANKER) else top_k
    retriever = store.as_retriever(search_kwargs={"k": retrieval_k, "filter": workspace_filter(workspace_id)})
    vector_candidates = retriever.invoke(question)
    return hybrid_retrieve(client=client, collection_name=QDRANT_COLLECTION, question=question, workspace_id=workspace_id, vector_candidates=vector_candidates, vector_top_k=RETRIEVAL_TOP_K, bm25_top_k=RETRIEVAL_TOP_K, final_top_k=top_k, enable_hybrid=ENABLE_HYBRID_SEARCH, enable_reranker=ENABLE_RERANKER, reranker_model=RERANKER_MODEL, siliconflow_api_key=SILICONFLOW_API_KEY, siliconflow_base_url=SILICONFLOW_BASE_URL)


def answer_prompt() -> PromptTemplate:
    return PromptTemplate(template="""你是一个严谨的企业知识库助手。\n只能依据下面的上下文回答问题。\n如果上下文不足，明确说“根据当前知识库没有找到足够的信息”。\n回答中的关键结论必须标注来源，引用格式为 [文件名 p.页码]。\n不要编造上下文之外的信息。\n\n上下文：\n{context}\n\n问题：\n{question}\n\n回答：""", input_variables=["context", "question"])


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        get_qdrant_client().get_collections()
        qdrant_status = "ok"
    except Exception as exc:
        qdrant_status = f"error: {exc}"
    return {"status": "ok" if qdrant_status == "ok" else "degraded", "qdrant": qdrant_status, "chat_model": DEEPSEEK_MODEL, "embedding_model": EMBEDDING_MODEL, "hybrid": ENABLE_HYBRID_SEARCH, "reranker": ENABLE_RERANKER}


@app.post("/v1/documents", response_model=IngestResponse)
async def upload_document(file: Annotated[UploadFile, File()], principal: Annotated[Principal, Depends(require_principal)]) -> IngestResponse:
    file_name = Path(file.filename or "upload.txt").name
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only PDF, TXT and Markdown are supported")
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        temp_path = Path(tmp.name)
    try:
        documents = load_uploaded_file(temp_path, file_name)
    finally:
        temp_path.unlink(missing_ok=True)
    chunks = ingest_documents(documents, workspace_id=principal.workspace_id, user_id=principal.username, file_name=file_name)
    return IngestResponse(file_name=file_name, workspace_id=principal.workspace_id, chunks=chunks, collection=QDRANT_COLLECTION)


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, principal: Annotated[Principal, Depends(require_principal)]) -> ChatResponse:
    started = time.perf_counter()
    documents = retrieve_documents(request.question, principal.workspace_id, request.top_k)
    if documents:
        context = "\n\n".join(f"[来源: {source_label(document)}]\n{document.page_content}" for document in documents)
        answer = (answer_prompt() | get_llm() | StrOutputParser()).invoke({"context": context, "question": request.question})
    else:
        answer = "根据当前知识库没有找到足够的信息，无法可靠回答该问题。"
    sources = list(dict.fromkeys(source_label(document) for document in documents))
    return ChatResponse(answer=answer, sources=sources, workspace_id=principal.workspace_id, trace_id=request.session_id or uuid.uuid4().hex, elapsed_ms=round((time.perf_counter() - started) * 1000, 2))


@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest, principal: Annotated[Principal, Depends(require_principal)]):
    documents = retrieve_documents(request.question, principal.workspace_id, request.top_k)
    sources = list(dict.fromkeys(source_label(document) for document in documents))
    trace_id = request.session_id or uuid.uuid4().hex
    context = "\n\n".join(f"[来源: {source_label(document)}]\n{document.page_content}" for document in documents)

    async def event_stream():
        yield f"event: sources\ndata: {json.dumps({'sources': sources, 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
        if not documents:
            message = "根据当前知识库没有找到足够的信息，无法可靠回答该问题。"
            yield f"data: {json.dumps({'token': message}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        chain = answer_prompt() | get_llm() | StrOutputParser()
        async for chunk in chain.astream({"context": context, "question": request.question}):
            if chunk:
                yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
