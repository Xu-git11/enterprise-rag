"""
Enterprise RAG - a derivative of the upstream Corrective RAG example.

Upstream:
https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/corrective_rag
License: Apache-2.0

This version uses DeepSeek for generation/grading, SiliconFlow BAAI/bge-m3 for
embeddings, and a local Qdrant service. Web search is disabled by default.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
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
from qdrant_client.models import Distance, VectorParams

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)
nest_asyncio.apply()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "").strip()
SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
).rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "corrective_rag").strip()
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() in {
    "1",
    "true",
    "yes",
}
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🔎",
    layout="wide",
)


def missing_configuration() -> list[str]:
    missing: list[str] = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not SILICONFLOW_API_KEY:
        missing.append("SILICONFLOW_API_KEY")
    return missing


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=1000,
        timeout=90,
        max_retries=2,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        chunk_size=32,
        timeout=90,
        max_retries=2,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=10)


def source_label(document: Document) -> str:
    file_name = (
        document.metadata.get("file_name")
        or document.metadata.get("title")
        or Path(str(document.metadata.get("source", "unknown"))).name
    )
    page = document.metadata.get("page")
    if page is not None:
        try:
            return f"{file_name} p.{int(page) + 1}"
        except (TypeError, ValueError):
            pass
    return str(file_name)


def format_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for document in documents:
        blocks.append(f"[来源: {source_label(document)}]\n{document.page_content}")
    return "\n\n".join(blocks)


def initialize_session_state() -> None:
    defaults: dict[str, Any] = {
        "doc_url": "",
        "ingested_source": None,
        "retriever": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("运行配置")
        st.write(f"Chat：`{DEEPSEEK_MODEL}`")
        st.write(f"Embedding：`{EMBEDDING_MODEL}`")
        st.write(f"Qdrant：`{QDRANT_URL}`")
        web_status = "开启" if ENABLE_WEB_SEARCH else "关闭"
        st.write(f"Web 搜索：`{web_status}`")

        missing = missing_configuration()
        if missing:
            st.error("缺少配置：" + ", ".join(missing))
            st.info("请在项目根目录的 .env 中填写 API Key，然后刷新页面。")
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
    if retriever is None:
        return {"keys": {"documents": [], "question": question}}
    documents = retriever.invoke(question)
    return {"keys": {"documents": documents, "question": question}}


def grade_documents(state: GraphState) -> GraphState:
    question = state["keys"]["question"]
    documents = state["keys"]["documents"]
    llm = get_llm()

    prompt = PromptTemplate(
        template="""你负责判断检索文档是否与用户问题相关。
只返回 JSON，格式必须是 {{"score": "yes"}} 或 {{"score": "no"}}。

文档：
{context}

问题：
{question}

规则：
- 根据关键词和语义相关性判断
- 只过滤明显无关的文档
- 不要输出 JSON 以外的任何内容""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm | StrOutputParser()

    filtered_documents: list[Document] = []
    run_web_search = "No"

    for document in documents:
        try:
            response = chain.invoke(
                {"question": question, "context": document.page_content}
            )
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                response = match.group(0)
            score = json.loads(response)
            if score.get("score") == "yes":
                filtered_documents.append(document)
            else:
                run_web_search = "Yes"
        except Exception:
            # Keep a document when grading fails to avoid dropping useful context.
            filtered_documents.append(document)

    return {
        "keys": {
            "documents": filtered_documents,
            "question": question,
            "run_web_search": run_web_search,
        }
    }


def transform_query(state: GraphState) -> GraphState:
    question = state["keys"]["question"]
    documents = state["keys"]["documents"]
    prompt = PromptTemplate(
        template="""请将下面的问题改写为更适合文档检索的问句。
只返回改写后的问题，不要解释。

原问题：
{question}""",
        input_variables=["question"],
    )
    chain = prompt | get_llm() | StrOutputParser()
    better_question = chain.invoke({"question": question})
    return {"keys": {"documents": documents, "question": better_question}}


def web_search(state: GraphState) -> GraphState:
    # First version is local-first. Tavily is disabled unless explicitly enabled.
    if not ENABLE_WEB_SEARCH or not TAVILY_API_KEY:
        return state
    raise NotImplementedError(
        "Tavily web fallback is not enabled in this first portfolio version."
    )


def decide_to_generate(state: GraphState) -> str:
    should_search = state["keys"].get("run_web_search") == "Yes"
    if should_search and ENABLE_WEB_SEARCH:
        return "transform_query"
    return "generate"


def generate(state: GraphState) -> GraphState:
    question = state["keys"]["question"]
    documents = state["keys"]["documents"]

    if not documents:
        return {
            "keys": {
                "documents": [],
                "question": question,
                "generation": "根据当前知识库没有找到足够的信息，无法可靠回答该问题。",
            }
        }

    prompt = PromptTemplate(
        template="""你是一个严谨的企业知识库助手。
只能依据下面的上下文回答问题。
如果上下文不足，明确说“根据当前知识库没有找到足够的信息”。
回答中的关键结论必须标注来源，引用格式为 [文件名 p.页码]。
不要编造上下文之外的信息。

上下文：
{context}

问题：
{question}

回答：""",
        input_variables=["context", "question"],
    )
    chain = prompt | get_llm() | StrOutputParser()

    try:
        generation = chain.invoke(
            {"context": format_context(documents), "question": question}
        )
    except Exception as exc:
        generation = f"生成回答时发生错误：{exc}"

    return {
        "keys": {
            "documents": documents,
            "question": question,
            "generation": generation,
        }
    }


def format_document(document: Document) -> str:
    return (
        f"来源：{source_label(document)}\n"
        f"内容：{document.page_content[:300]}..."
    )


def format_state(state: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for key, value in state.items():
        if key == "documents" and isinstance(value, list):
            formatted[key] = [format_document(document) for document in value]
        else:
            formatted[key] = value
    return formatted


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


st.title("🔎 Enterprise RAG")
st.caption("DeepSeek Chat + SiliconFlow BAAI/bge-m3 + 本地 Qdrant")

initialize_session_state()
render_sidebar()

embeddings = get_embeddings()
client = get_qdrant_client()
app = build_graph()

st.subheader("文档输入")
input_option = st.radio("输入方式", ["URL", "文件上传"], horizontal=True)

docs: list[Document] | None = None
source_key: str | None = None

if input_option == "URL":
    url = st.text_input("文档 URL", value=st.session_state.doc_url)
    if url:
        docs = load_documents(url, is_url=True)
        source_key = f"url:{url}"
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

        for document in docs:
            document.metadata["file_name"] = uploaded_file.name
            document.metadata["source"] = uploaded_file.name

        digest = hashlib.sha256(file_bytes).hexdigest()
        source_key = f"upload:{uploaded_file.name}:{digest}"

if docs and source_key and st.session_state.ingested_source != source_key:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    splits = splitter.split_documents(docs)

    try:
        client.delete_collection(QDRANT_COLLECTION)
    except Exception:
        pass

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
    )

    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings,
    )

    with st.spinner("正在切分文档并生成向量..."):
        vectorstore.add_documents(splits)

    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    st.session_state.ingested_source = source_key
    st.success(f"已写入 {len(splits)} 个文档片段到 Qdrant。")

st.subheader("知识库问答")
user_question = st.text_input("请输入问题")

if user_question:
    if st.session_state.get("retriever") is None:
        st.warning("请先上传文档或提供 URL。")
    else:
        inputs = {"keys": {"question": user_question}}
        final_generation: str | None = None

        with st.spinner("正在检索并生成回答..."):
            for output in app.stream(inputs):
                for node_name, value in output.items():
                    with st.expander(f"步骤：{node_name}", expanded=False):
                        st.code(
                            json.dumps(
                                format_state(value["keys"]),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            language="json",
                        )
                    if "generation" in value["keys"]:
                        final_generation = value["keys"]["generation"]

        st.subheader("回答")
        st.write(final_generation or "没有生成回答。")
