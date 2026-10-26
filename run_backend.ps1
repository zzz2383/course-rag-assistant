# =============================================================================
# 一键启动 FastAPI 后端
# 用法：
#   .\run_backend.ps1
#   .\run_backend.ps1 -Port 8001
# =============================================================================
[CmdletBinding()]
param(
    [string]$BindHost = "",
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " 课程资料问答与复习规划助手 - 后端启动" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# 1. 定位虚拟环境（优先 venv，兼容 .venv）
# -----------------------------------------------------------------------------
$venvCandidates = @(
    (Join-Path $ProjectRoot "venv"),
    (Join-Path $ProjectRoot ".venv")
)
$venvPath = $null
foreach ($candidate in $venvCandidates) {
    if (Test-Path (Join-Path $candidate "Scripts\python.exe")) {
        $venvPath = $candidate
        break
    }
}

if (-not $venvPath) {
    Write-Host "[错误] 未找到虚拟环境 venv（或 .venv）。" -ForegroundColor Red
    Write-Host "请先在项目根目录执行：" -ForegroundColor Yellow
    Write-Host "  python -m venv venv" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

# -----------------------------------------------------------------------------
# 2. 激活虚拟环境；如果执行策略阻止激活，仍使用 venv 内的 Python 启动
# -----------------------------------------------------------------------------
if (Test-Path $activateScript) {
    try {
        . $activateScript
        Write-Host "[信息] 已激活虚拟环境：$venvPath" -ForegroundColor Green
    }
    catch {
        Write-Host "[警告] 激活脚本执行失败，改用 venv 内置 Python 直接启动。" -ForegroundColor Yellow
        Write-Host "       如仍失败，可在当前 PowerShell 执行：" -ForegroundColor Yellow
        Write-Host "       Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass" -ForegroundColor Yellow
    }
}

# -----------------------------------------------------------------------------
# 3. 基础检查
# -----------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $ProjectRoot "app\main.py"))) {
    Write-Host "[错误] 未找到 app\main.py，请确认后端代码位于项目根目录的 app\ 下。" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
    Write-Host "[警告] 未找到 .env 文件，将使用默认配置。" -ForegroundColor Yellow
    Write-Host "       建议先执行：Copy-Item .env.example .env，并填写 DEEPSEEK_API_KEY。" -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------
# 4. 确定监听地址与端口（命令行参数 > 环境变量 > 默认值）
# -----------------------------------------------------------------------------
if ($BindHost) {
    $hostAddress = $BindHost
}
elseif ($env:APP_HOST) {
    $hostAddress = $env:APP_HOST
}
else {
    $hostAddress = "127.0.0.1"
}

if ($Port -gt 0) {
    $listenPort = $Port
}
elseif ($env:APP_PORT) {
    $listenPort = [int]$env:APP_PORT
}
else {
    $listenPort = 8000
}

# -----------------------------------------------------------------------------
# 5. 启动 Uvicorn（优先使用 venv 中的可执行文件，缺失时回退到 python -m）
# -----------------------------------------------------------------------------
$uvicornExe = Join-Path $venvPath "Scripts\uvicorn.exe"
Write-Host ""
Write-Host "[信息] API 文档：http://${hostAddress}:$listenPort/docs" -ForegroundColor Green
Write-Host "[信息] 健康检查：http://${hostAddress}:$listenPort/api/health" -ForegroundColor Green
Write-Host "[信息] 按 Ctrl+C 停止后端。" -ForegroundColor DarkGray
Write-Host ""

if (Test-Path $uvicornExe) {
    & $uvicornExe app.main:app --host $hostAddress --port $listenPort --reload
}
else {
    & $pythonExe -m uvicorn app.main:app --host $hostAddress --port $listenPort --reload
}
