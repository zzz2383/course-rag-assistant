# =============================================================================
# 一键启动 Streamlit 前端
# 用法：
#   .\run_frontend.ps1
#   .\run_frontend.ps1 -Port 8502
# 说明：
#   脚本会激活 venv，并从 .env 中读取 API_BASE_URL（若环境变量未设置）。
# =============================================================================
[CmdletBinding()]
param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " 课程资料问答与复习规划助手 - 前端启动" -ForegroundColor Cyan
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
# 3. 确定 API 地址
#    优先级：当前进程环境变量 > .env 中的 API_BASE_URL > 默认值
# -----------------------------------------------------------------------------
if (-not $env:API_BASE_URL) {
    $envFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $envFile) {
        $apiLine = Get-Content -LiteralPath $envFile -Encoding UTF8 |
            Where-Object { $_ -match '^\s*API_BASE_URL\s*=' } |
            Select-Object -Last 1
        if ($apiLine) {
            $apiValue = ($apiLine -split "=", 2)[1].Trim()
            $apiValue = $apiValue.Trim('"').Trim("'")
            if ($apiValue) {
                $env:API_BASE_URL = $apiValue
            }
        }
    }
}

if (-not $env:API_BASE_URL) {
    $env:API_BASE_URL = "http://127.0.0.1:8000/api"
}

# -----------------------------------------------------------------------------
# 4. 基础检查
# -----------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $ProjectRoot "frontend\streamlit_app.py"))) {
    Write-Host "[错误] 未找到 frontend\streamlit_app.py。" -ForegroundColor Red
    exit 1
}

$streamlitExe = Join-Path $venvPath "Scripts\streamlit.exe"
if (-not (Test-Path $streamlitExe)) {
    Write-Host "[错误] 当前虚拟环境未安装 Streamlit。" -ForegroundColor Red
    Write-Host "请执行：pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "[信息] 后端 API：$env:API_BASE_URL" -ForegroundColor Green
Write-Host "[信息] 前端地址：http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "[信息] 按 Ctrl+C 停止前端。" -ForegroundColor DarkGray
Write-Host ""

# -----------------------------------------------------------------------------
# 5. 启动 Streamlit
# -----------------------------------------------------------------------------
& $streamlitExe run "frontend\streamlit_app.py" --server.address 127.0.0.1 --server.port $Port
