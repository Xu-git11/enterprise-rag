# Enterprise RAG

基于 Apache-2.0 开源项目 Corrective RAG 二次开发的企业知识库问答项目。

## 当前技术栈

- Streamlit：Web UI
- DeepSeek：Chat、文档相关性判断、查询改写
- SiliconFlow：`BAAI/bge-m3` Embedding
- Qdrant：本地向量数据库
- LangChain / LangGraph：RAG 和状态图编排
- Python 3.12.14

## 与上游版本的差异

上游项目：`Shubhamsaboo/awesome-llm-apps/rag_tutorials/corrective_rag`

本项目当前已完成：

- API Key 从页面输入改为 `.env`
- Chat 从 Anthropic 切换为 DeepSeek OpenAI-compatible API
- Embedding 从 OpenAI 切换为 SiliconFlow `BAAI/bge-m3`
- Qdrant 改为默认本地连接
- 向量维度改为 BGE-M3 的 1024 维
- 默认关闭 Tavily Web 搜索
- 增加 API 和 Qdrant 连通性检查脚本
- 中文切分和基础来源标签

## 本地配置

复制或直接编辑 `.env`：

```env
DEEPSEEK_API_KEY=你的DeepSeekKey
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

SILICONFLOW_API_KEY=你的SiliconFlowKey
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=corrective_rag

ENABLE_WEB_SEARCH=false
TAVILY_API_KEY=
```

`.env` 已写入 `.gitignore`，不要提交到 Git。

## 启动 Qdrant

```powershell
docker pull docker.m.daocloud.io/qdrant/qdrant:latest

docker run -d `
  --name qdrant-local `
  -p 6333:6333 `
  -p 6334:6334 `
  -v qdrant_storage:/qdrant/storage `
  docker.m.daocloud.io/qdrant/qdrant:latest
```

## 创建环境

```powershell
uv venv --python 3.12.14 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 检查 API 连通性

```powershell
.\.venv\Scripts\python.exe scripts\check_apis.py
```

成功时应该看到：

```text
DeepSeek: OK
SiliconFlow embedding: OK
Qdrant: OK
```

## 启动应用

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开：

```text
http://localhost:8501
```

## License

本项目保留上游 Apache-2.0 License。来源 commit 和原始路径记录在 `UPSTREAM.json`。
