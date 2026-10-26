# 课程资料问答与复习规划助手

一个面向**单门课程**的 RAG（检索增强生成）学习助手：把 PDF、PPTX、DOCX、TXT 等课程资料上传到后端，系统自动解析、分块并建立向量索引；学生可以用自然语言提问，并获得带文件名和页码的引用来源；还可以根据考试日期、章节和薄弱点生成逐日复习计划并导出 Markdown。

界面使用 Streamlit，后端使用 FastAPI，二者通过 REST API 通信。

## 功能

### 1. 资料管理

- 批量上传 PDF / PPTX / DOCX / TXT 课程资料；
- 上传时填写课程名称和可选章节，后端自动解析并建立索引；
- 文档列表展示文件名、课程、章节、页数、分块数、索引状态和上传时间；
- 支持刷新列表，以及按 `doc_id` 删除文档及其向量、解析产物；
- 重复文件由后端按哈希识别，接口返回「已跳过」。

### 2. 课程问答

- 输入课程名称、可选章节和问题，按课程/章节范围检索；
- 展示中文答案；
- 每条答案下方展示引用来源列表：**文件名 + 页码 + 原文摘录**，便于回到资料核验；
- 会话历史保存在 `st.session_state`，支持多轮查看和一键清空；
- 后端不可达时给出启动提示，不抛出堆栈。

### 3. 复习计划

- 选择考试日期；
- 从已上传资料自动提取章节并多选，也支持手动补充章节；
- 输入薄弱点（每行一个，或使用逗号/分号分隔）；
- 一键调用后端生成逐日计划，表格展示「天 / 日期 / 学习重点 / 任务」；
- 使用 `st.download_button` 将计划导出为 Markdown；
- 当 `generated_by=rule` 时明确提示规则模板降级，计划仍可正常使用。

## 技术栈

| 层次 | 技术 |
| --- | --- |
| 前端 | Streamlit 1.41、requests、Python 3.11 |
| 后端 | FastAPI、Uvicorn、Pydantic v2、pydantic-settings |
| RAG | LangChain、Chroma 向量库、Embedding（本地或 OpenAI 兼容 API） |
| 大模型 | DeepSeek Chat（默认）/ 通义千问（备用），OpenAI 兼容协议 |
| 文档解析 | pypdf、python-pptx、python-docx、TXT 解析 |
| 数据存储 | 本地文件 + Chroma + `data/` 下的 JSON/元数据 |
| 运行环境 | Windows PowerShell（也适用于 Linux/macOS） |

## 目录结构

```text
course-rag-assistant/
├── app/                        # FastAPI 后端（由后端模块提供）
│   ├── main.py                 # 应用入口、CORS、健康检查
│   ├── api/                    # /api/documents、/api/chat、/api/plan 路由
│   ├── agents/                 # 复习计划智能体与规则降级
│   ├── core/                   # 配置、日志、公共异常
│   ├── models/                 # Pydantic 数据契约
│   ├── rag/                    # 解析、检索、生成、引用
│   └── services/               # 解析、分块、Embedding 等服务
├── frontend/
│   └── streamlit_app.py        # Streamlit 三页签前端
├── docs/
│   ├── 需求分析说明书.md
│   └── 总体设计文档.md
├── tests/                      # 后端测试
├── data/                       # 运行数据：上传、解析、Chroma、计划等（Git 忽略）
├── .env.example                # 全部环境变量示例
├── .gitignore
├── requirements.txt
├── run_backend.ps1             # 一键启动后端
├── run_frontend.ps1            # 一键启动前端
└── README.md
```

## 快速开始

### 1. 环境要求

- Windows 10/11 64 位（开发/演示环境）；
- Python 3.11.x，建议使用 64 位版本；
- 后端端口 `8000`，前端端口 `8501`（均可在命令中修改）；
- 使用在线大模型时需要能访问 DeepSeek 或通义千问 API；检索本身可在本地完成。

### 2. 创建虚拟环境

在项目根目录打开 PowerShell：

```powershell
cd D:\work\course-rag-assistant

# 创建名为 venv 的虚拟环境
python -m venv venv

# 激活（PowerShell）
.\venv\Scripts\Activate.ps1

# 如提示执行策略限制，可只对当前进程放开
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 安装依赖
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CMD 用户激活命令为：

```bat
venv\Scripts\activate.bat
```

> 如果使用本地 Embedding（`EMBEDDING_MODE=local`），还需安装可选的 `sentence-transformers`；如果上传接口提示缺少表单解析依赖，请安装 `python-multipart`：
>
> ```powershell
> pip install sentence-transformers python-multipart
> ```
>
> 若本机无法下载模型，可改用 `EMBEDDING_MODE=api`，并填写对应的 Embedding API 配置。

### 3. 配置 `.env`

复制示例文件并编辑：

```powershell
Copy-Item .env.example .env
notepad .env
```

至少需要配置一个 LLM 供应商：

```dotenv
# 默认使用 DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

或者切换到通义千问：

```dotenv
LLM_PROVIDER=qwen
QWEN_API_KEY=你的 DashScope 密钥
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
```

常用配置说明（完整项见 `.env.example`）：

| 配置项 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥，默认供应商必填 |
| `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | DeepSeek 接口地址与模型名 |
| `QWEN_API_KEY` | 通义千问密钥，将 `LLM_PROVIDER` 改为 `qwen` 后使用 |
| `LLM_PROVIDER` | `deepseek` 或 `qwen` |
| `LLM_TIMEOUT` | 模型调用超时秒数，默认 60 |
| `LLM_TEMPERATURE` | 生成温度，默认 0.2 |
| `EMBEDDING_MODE` | `local`（本地模型）或 `api`（在线 Embedding） |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 本地模型名与向量维度，默认 384 |
| `EMBEDDING_API_KEY` / `EMBEDDING_API_BASE_URL` | API Embedding 模式下必填 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 文本分块大小与重叠字符数 |
| `TOP_K` | 检索返回的片段数 |
| `MAX_FILE_SIZE_MB` | 单文件大小上限，默认 20 MB |
| `DATA_DIR` / `CHROMA_DIR` | 运行数据与向量库目录 |
| `API_BASE_URL` | Streamlit 调用的后端 API 前缀，默认 `http://127.0.0.1:8000/api` |

安全提醒：`.env` 已被 `.gitignore` 忽略；模型密钥只由后端读取，Streamlit 前端不会也不能读取这些密钥。前端只会读取 `API_BASE_URL` 这一项。

### 4. 启动后端

终端 1（项目根目录）：

```powershell
# 已激活 venv 的情况下
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

也可以使用一键脚本：

```powershell
.\run_backend.ps1
```

后端地址：

- API：http://127.0.0.1:8000/api
- Swagger 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/api/health

### 5. 启动前端

终端 2（项目根目录，保持 venv 激活）：

```powershell
streamlit run frontend/streamlit_app.py --server.port 8501
```

也可以使用一键脚本：

```powershell
.\run_frontend.ps1
```

浏览器访问：http://127.0.0.1:8501

`run_frontend.ps1` 会自动读取 `.env` 中的 `API_BASE_URL`；如果没有配置，则使用 `http://127.0.0.1:8000/api`。

## 使用流程

1. **上传资料**：打开「资料管理」，填写课程名称（全程保持一致）和可选章节，选择文件后点击「上传并建立索引」。
2. **确认索引状态**：文档列表中状态为「可用」时代表已可检索；点击「刷新列表」可获取最新状态。
3. **课程问答**：切换到「课程问答」，填写相同课程名称；可填写章节缩小检索范围，然后提问。
4. **核对引用**：答案下方会列出文件名、页码和原文摘录。引用列表为空时，说明资料不足以支撑回答，建议补充相关章节。
5. **生成复习计划**：切换到「复习计划」，选择考试日期和章节，填写薄弱点，生成后查看每日安排。
6. **导出计划**：点击「导出 Markdown」保存复习计划；如果页面提示规则降级，可稍后在模型恢复后重新生成。

## API 约定

前端依赖以下接口（后端默认监听 `http://127.0.0.1:8000`）：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/documents/upload` | 上传文件（`files` 或 `file` 表单字段，另可带 `course`、`chapter`） |
| `GET` | `/api/documents` | 文档列表，返回 `items` 或直接列表 |
| `DELETE` | `/api/documents/{doc_id}` | 删除文档及索引 |
| `POST` | `/api/chat` | 请求 `{question, course, chapter?}`，返回 `{answer, citations}` |
| `POST` | `/api/plan` | 请求 `{exam_date, chapters, weak_points}`，返回 `{plan, generated_by}` |
| `GET` | `/api/health` | 前端侧边栏连接探测 |

问答的 `citations` 至少包含：

```json
[
  {
    "source_file": "第3章_动态规划.pdf",
    "page": 12,
    "excerpt": "dp[i][j] = max(...)"
  }
]
```

复习计划的响应示例：

```json
{
  "plan": [
    {
      "day": 1,
      "date": "2026-10-27",
      "tasks": ["阅读课件第 10-14 页，整理状态定义", "手推 0-1 背包"],
      "focus": "第三章 动态规划"
    }
  ],
  "generated_by": "llm"
}
```

## 降级与容错说明

本项目按「先保证可用，再追求智能」的原则设计，常见的降级路径如下：

| 场景 | 表现 | 处理方式 |
| --- | --- | --- |
| 后端未启动 / 网络不通 | 页面显示「无法连接后端服务」，并给出启动命令 | 启动 `run_backend.ps1` 后点击「重新检测连接」或刷新页面 |
| 上传文件过大 / 格式不支持 | 前端先做格式和 20 MB 限制校验；后端仍会二次校验 | 压缩、拆分或转换为 PDF/DOCX/PPTX/TXT |
| 重复上传同一文件 | 后端按内容哈希识别并返回 `skipped` | 复用已有文档，无需重新索引 |
| 大模型超时 / 额度耗尽 / 未配置 Key | 问答链路降级为检索片段摘录，仍展示答案和引用；计划降级为规则模板 | 检查 `.env` 中的 `DEEPSEEK_API_KEY`；模型恢复后重新提问或生成计划 |
| 无有效检索结果 | 返回「资料不足」类回答且 `citations=[]` | 补充对应章节资料，或放宽章节筛选 |
| 复习计划 `generated_by=rule` | 页面显示黄色降级提示，下载按钮仍可用 | 计划可执行；稍后可重新生成以获得 LLM 优化版本 |
| Streamlit 侧边栏探测失败 | 红色「后端未连接」以及 API 地址 | 核对 `API_BASE_URL`、端口和防火墙设置 |

前端所有后端调用均有异常兜底，不会因为后端未启动或接口返回 4xx/5xx 而显示 Python 堆栈。

## 常见问题

**激活虚拟环境时报执行策略错误**

只对当前 PowerShell 进程放开：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**端口被占用**

修改启动端口，或在 PowerShell 中查看占用进程：

```powershell
Get-NetTCPConnection -LocalPort 8000,8501
```

**上传成功但文档列表为空**

点击「刷新列表」；确认后端日志；后端向量库初始化失败时会降级为本地元数据列表。

**问答没有引用**

可能当前问题没有命中资料，或章节筛选过窄。尝试清空章节、换一种问法，或上传更相关的资料。

**计划生成失败**

确认考试日期晚于今天；若后端提示 `PLAN_VALIDATION_FAILED`，请调整日期、每日时长或章节后重试。未选择章节时后端会尝试自动推断，建议手动补充更明确的章节。

**首次问答/上传很慢**

本地 Embedding 模型首次加载需要下载并加载模型，属于正常现象。若机器内存或网络受限，改用 `EMBEDDING_MODE=api`。

## 安全与隐私

- `.env` 不纳入版本控制，提交前检查是否误加真实密钥；
- 模型 API Key 只在后端进程中使用，Streamlit 前端只调用后端 API；
- 前端错误提示不会展示完整的密钥或堆栈；
- 本项目的 Chroma 数据和上传资料默认保存在本机 `data/` 目录，请按需备份或清理。

## 相关文档

- [需求分析说明书](docs/需求分析说明书.md)
- [总体设计文档](docs/总体设计文档.md)
