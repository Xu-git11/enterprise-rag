"""
Enterprise RAG - a derivative of the upstream Corrective RAG example.

Upstream:
https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/corrective_rag
License: Apache-2.0
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, TypedDict
from urllib.parse import urlparse

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from auth import authenticate, load_users
from rag_core import hybrid_retrieve, workspace_filter

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None

load_dotenv(BASE_DIR / ".env", override=False)
nest_asyncio.apply()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "").strip()
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "corrective_rag_v3").strip()
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() in {"1", "true", "yes"}
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() in {"1", "true", "yes"}
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() in {"1", "true", "yes"}
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() in {"1", "true", "yes"}
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
USERS_FILE = Path(os.getenv("USERS_FILE", "users.json"))
if not USERS_FILE.is_absolute():
    USERS_FILE = BASE_DIR / USERS_FILE
if not USERS_FILE.exists():
    USERS_FILE = BASE_DIR / "users.example.json"

st.set_page_config(page_title="Enterprise RAG", page_icon="🔎", layout="wide")


def missing_configuration() -> list[str]:
    missing: list[str] = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not SILICONFLOW_API_KEY:
        missing.append("SILICONFLOW_API_KEY")
    return missing


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, temperature=0, max_tokens=1000, timeout=90, max_retries=2)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL, chunk_size=32, timeout=90, max_retries=2, tiktoken_enabled=False, check_embedding_ctx_length=False)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=10)


@st.cache_resource
def get_langfuse_client():
    if not LANGFUSE_ENABLED or Langfuse is None:
        return None
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return None
    return Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST, enabled=True)


def source_label(document: Document) -> str:
    file_name = document.metadata.get("file_name") or document.metadata.get("title") or Path(str(document.metadata.get("source", "unknown"))).name
    page = document.metadata.get("page")
    if page is None:
        return str(file_name)
    try:
        return f"{file_name} p.{int(page) + 1}"
    except (TypeError, ValueError):
        return str(file_name)


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(f"[来源: {source_label(document)}]\n{document.page_content}" for document in documents)


def initialize_session_state() -> None:
    defaults = {"doc_url": "", "ingested_source": None, "retriever": None, "authenticated_user": None, "session_id": uuid.uuid4().hex}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_login() -> dict[str, Any]:
    if st.session_state.authenticated_user:
        return st.session_state.authenticated_user
    users = load_users(USERS_FILE)
    st.info("演示账号：`demo / demo123` 或 `analyst / analyst123`。生产环境应替换为正式身份系统。")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")
    if submitted:
        user = authenticate(username.strip(), password, users)
        if user:
            st.session_state.authenticated_user = user
            st.session_state.ingested_source = None
            st.session_state.retriever = None
            st.rerun()
        st.error("用户名或密码错误。")
    st.stop()


def render_sidebar(user: dict[str, Any]) -> None:
    workspace_id = str(user["workspace_id"])
    with st.sidebar:
        st.subheader("当前用户")
        st.write(str(user.get("display_name", user["username"])))
        st.caption(f"workspace: {workspace_id}")
        if st.button("退出登录"):
            st.session_state.authenticated_user = None
            st.session_state.retriever = None
            st.session_state.ingested_source = None
            st.rerun()
        st.divider()
        st.subheader("运行配置")
        st.write(f"Chat：`{DEEPSEEK_MODEL}`")
        st.write(f"Embedding：`{EMBEDDING_MODEL}`")
        st.write(f"Qdrant：`{QDRANT_URL}`")
        st.write(f"Hybrid：`{'开启' if ENABLE_HYBRID_SEARCH else '关闭'}`")
        st.write(f"Reranker：`{'开启' if ENABLE_RERANKER else '关闭'}`")
        st.write(f"Langfuse：`{'开启' if LANGFUSE_ENABLED else '关闭'}`")
        missing = missing_configuration()
        if missing:
            st.error("缺少配置：" + ", ".join(missing))
            st.stop()
        try:
            get_qdrant_client().get_collections()
            st.success("Qdrant 连接正常")
        except Exception as exc:
            st.error(f"Qdrant 连接失败：{exc}")
            st.stop()


def load_documents(file_or_url: str, is_url: bool) -> list[Document]:
    try:
        if is_url:
            if urlparse(file_or_url).path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_or_url)
            else:
                loader = WebBaseLoader(file_or_url)
                loader.requests_per_second = 1
        else:
            extension = Path(file_or_url).suffix.lower()
            if extension == ".pdf":
                loader = PyPDFLoader(file_or_url)
            elif extension in {".txt", ".md"}:
                loader = TextLoader(file_or_url, encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file type: {extension}")
        return loader.load()
    except Exception as exc:
        st.error(f"文档加载失败：{exc}")
        return []


class GraphState(TypedDict):
    keys: Dict[str, Any]


def retrieve(state: GraphState) -> GraphState:
    question = state["keys"]["question"]
    retriever = st.session_state.get("retriever")
    user = st.session_state.get("authenticated_user") or {}
    workspace_id = str(user.get("workspace_id", ""))
    if retriever is None or not workspace_id:
        return {"keys": {"documents": [], "question": question}}
    vector_candidates = retriever.invoke(question)
    documents = hybrid_retrieve(client=get_qdrant_client(), collection_name=QDRANT_COLLECTION, question=question, workspace_id=workspace_id, vector_candidates=vector_candidates, vector_top_k=RETRIEVAL_TOP_K, bm25_top_k=RETRIEVAL_TOP_K, final_top_k=RERANK_TOP_N, enable_hybrid=ENABLE_HYBRID_SEARCH, enable_reranker=ENABLE_RERANKER, reranker_model=RERANKER_MODEL, siliconflow_api_key=SILICONFLOW_API_KEY, siliconflow_base_url=SILICONFLOW_BASE_URL)
    return {"keys": {"documents": documents, "question": question}}


def grade_documents(state: GraphState) -> GraphState:
    question = state["keys"]["question"]
    documents = state["keys"]["documents"]
    llm = get_llm()
    prompt = PromptTemplate(template="""你负责判断检索文档是否与用户问题相关。\n只返回 JSON，格式必须是 {{"score": "yes"}} 或 {{"score": "no"}}。\n\n文档：\n{context}\n\n问题：\n{question}\n\n规则：\n- 根据关键词和语义相关性判断\n- 只过滤明显无关的文档\n- 不要输出 JSON 以外的任何内容""", input_variables=["context", "question"])
    chain = prompt | llm | StrOutputParser()
    filtered_documents = []
    run_web_search = "No"
    for document in documents:
        try:
            response = chain.invoke({"question": question, "context": document.page_content})
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                response = match.group(0)
            score = json.loads(response)
            if score.get("score") == "yes":
                filtered_documents.append(document)
            else:
                run_web_search = "Yes"
        except Exception:
            filtered_documents.append(document)
    return {"keys": {"documents": filtered_documents, "question": question, "run_web_search": run_web_search}}


def transform_query(state: GraphState) -> GraphState:
    question = state["keys"]["question"]
    documents = state["keys"]["documents"]
    prompt = PromptTemplate(template="""请将下面的问题改写为更适合文档检索的问句。\n只返回改写后的问题，不要解释。\n\n原问题：\n{question}""", input_variables=["question"])
    better_question = (prompt | get_llm() | StrOutputParser()).invoke({"question": question})
    return {"keys": {"documents": documents, "question": better_question}}


def web_search(state: GraphState) -> GraphState:
    return state


def decide_to_generate(state: GraphState) -> str:
    return "transform_query" if state["keys"].get("run_web_search") == "Yes" and ENABLE_WEB_SEARCH else "generate"


def generate(state: GraphState) -> GraphState:
    question = state["keys"]["question"]
    documents = state["keys"]["documents"]
    if not documents:
        return {"keys": {"documents": [], "question": question, "generation": "根据当前知识库没有找到足够的信息，无法可靠回答该问题。", "sources": []}}
    prompt = PromptTemplate(template="""你是一个严谨的企业知识库助手。\n只能依据下面的上下文回答问题。\n如果上下文不足，明确说“根据当前知识库没有找到足够的信息”。\n回答中的关键结论必须标注来源，引用格式为 [文件名 p.页码]。\n不要编造上下文之外的信息。\n\n上下文：\n{context}\n\n问题：\n{question}\n\n回答：""", input_variables=["context", "question"])
    try:
        generation = (prompt | get_llm() | StrOutputParser()).invoke({"context": format_context(documents), "question": question})
    except Exception as exc:
        generation = f"生成回答时发生错误：{exc}"
    sources = list(dict.fromkeys(source_label(document) for document in documents))
    return {"keys": {"documents": documents, "question": question, "generation": generation, "sources": sources}}


def format_document(document: Document) -> str:
    return f"来源：{source_label(document)}\n内容：{document.page_content[:300]}..."


def format_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {key: ([format_document(document) for document in value] if key == "documents" and isinstance(value, list) else value) for key, value in state.items()}


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"})
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


st.title("🔎 Enterprise RAG")
st.caption("DeepSeek + SiliconFlow BGE-M3 + Qdrant + LangGraph")
initialize_session_state()
user = render_login()
render_sidebar(user)
workspace_id = str(user["workspace_id"])
user_id = str(user["username"])
embeddings = get_embeddings()
client = get_qdrant_client()
app = build_graph()

st.subheader("文档输入")
input_option = st.radio("输入方式", ["URL", "文件上传"], horizontal=True)
docs: list[Document] | None = None
source_key: str | None = None
source_name: str | None = None
if input_option == "URL":
    url = st.text_input("文档 URL", value=st.session_state.doc_url)
    if url:
        docs = load_documents(url, is_url=True)
        source_name = Path(urlparse(url).path).name or url
        source_key = f"url:{workspace_id}:{url}"
else:
    uploaded_file = st.file_uploader("上传 PDF / TXT / Markdown", type=["pdf", "txt", "md"])
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            temp_path = tmp_file.name
        try:
            docs = load_documents(temp_path, is_url=False)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        source_name = uploaded_file.name
        source_key = f"upload:{workspace_id}:{uploaded_file.name}:{hashlib.sha256(file_bytes).hexdigest()}"

if docs and source_key and source_name and st.session_state.ingested_source != source_key:
    for document in docs:
        document.metadata["workspace_id"] = workspace_id
        document.metadata["user_id"] = user_id
        document.metadata["file_name"] = source_name
        document.metadata["source"] = source_name
    splits = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60, separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]).split_documents(docs)
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(collection_name=QDRANT_COLLECTION, vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE))
    source_filter = Filter(must=[FieldCondition(key="metadata.workspace_id", match=MatchValue(value=workspace_id)), FieldCondition(key="metadata.file_name", match=MatchValue(value=source_name))])
    try:
        client.delete(collection_name=QDRANT_COLLECTION, points_selector=source_filter)
    except Exception:
        pass
    vectorstore = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=embeddings)
    with st.spinner("正在切分文档并生成向量..."):
        vectorstore.add_documents(splits)
    effective_top_k = RETRIEVAL_TOP_K if (ENABLE_HYBRID_SEARCH or ENABLE_RERANKER) else RERANK_TOP_N
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": effective_top_k, "filter": workspace_filter(workspace_id)})
    st.session_state.ingested_source = source_key
    st.success(f"已写入 {len(splits)} 个文档片段到工作区 {workspace_id}。")

st.subheader("知识库问答")
user_question = st.text_input("请输入问题")
if user_question:
    if st.session_state.get("retriever") is None:
        st.warning("请先上传文档或提供 URL。")
    else:
        inputs = {"keys": {"question": user_question}}
        final_generation = None
        final_sources = []
        langfuse_client = get_langfuse_client()
        trace = None
        if langfuse_client:
            trace = langfuse_client.trace(name="rag_query", user_id=user_id, session_id=st.session_state.session_id, input={"question": user_question}, metadata={"workspace_id": workspace_id, "model": DEEPSEEK_MODEL})
        with st.spinner("正在检索并生成回答..."):
            for output in app.stream(inputs):
                for node_name, value in output.items():
                    formatted = format_state(value["keys"])
                    with st.expander(f"步骤：{node_name}", expanded=False):
                        st.code(json.dumps(formatted, ensure_ascii=False, indent=2), language="json")
                    if trace:
                        span = trace.span(name=node_name, input={"question": user_question}, output=formatted)
                        span.end()
                    if "generation" in value["keys"]:
                        final_generation = value["keys"]["generation"]
                    if "sources" in value["keys"]:
                        final_sources = value["keys"]["sources"]
        if trace:
            trace.update(output=final_generation, metadata={"sources": final_sources, "workspace_id": workspace_id})
            langfuse_client.flush()
        st.subheader("回答")
        st.write(final_generation or "没有生成回答。")
        if final_sources:
            st.subheader("检索来源")
            for source in final_sources:
                st.markdown(f"- `{source}`")
