"""书卷风视觉主题。

集中存放全局 CSS 与若干「纸页化」HTML 片段渲染器。所有函数只返回字符串，
不依赖 Streamlit 运行时，便于单独测试。

设计约定：
- 米白纸色 + 墨色字 + 朱印点缀，标题黑体、正文宋体；
- 侧边栏是书的扉页，页签是书口标签，内容区是一张纸；
- 所有文案先经 ``html.escape``，外部数据不会被当成标签解析。
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 色板
# ---------------------------------------------------------------------------
PAPER = "#FAF7EF"
PAPER_EDGE = "#EFE9D8"
INK = "#2C2A24"
INK_SOFT = "#5C5648"
INK_FAINT = "#8A8371"
SEAL = "#A8442A"
JADE = "#4F6B62"
SPINE = "#2F3A34"
RULE = "#DCD4BC"

FONT_HEAD = '"Noto Serif SC","Source Han Serif SC","Songti SC","SimSun",serif'
FONT_BODY = '"Songti SC","SimSun","Noto Serif SC",serif'
FONT_UI = '-apple-system,"Segoe UI","Microsoft YaHei",sans-serif'

CN_NUM = "壹贰叁肆伍陆柒捌玖拾"


def esc(value: Any) -> str:
    """转义为可安全嵌入 HTML 的文本。"""
    if value is None:
        return ""
    return html.escape(str(value), quote=False)


# ---------------------------------------------------------------------------
# 全局样式
# ---------------------------------------------------------------------------
CSS = f"""
<style>
:root {{
  --paper: {PAPER};
  --paper-edge: {PAPER_EDGE};
  --ink: {INK};
  --ink-soft: {INK_SOFT};
  --ink-faint: {INK_FAINT};
  --seal: {SEAL};
  --jade: {JADE};
  --spine: {SPINE};
  --rule: {RULE};
}}

/* ========== 隐藏 Streamlit 的英文外壳 ========== */
[data-testid="stAppDeployButton"],
#MainMenu,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{ display: none !important; }}

/* ========== 整体底色与纸张纹理 ========== */
.stApp {{
  background:
    radial-gradient(circle at 18% 12%, rgba(255,255,255,.55), transparent 42%),
    repeating-linear-gradient(0deg, rgba(120,105,70,.018) 0 1px, transparent 1px 4px),
    #EFEADA;
  font-family: {FONT_BODY};
  color: var(--ink);
}}
[data-testid="stAppViewBlockContainer"],
.block-container {{
  max-width: 1180px;
  padding-top: 1.4rem;
  padding-bottom: 3rem;
}}

/* ========== 侧边栏：扉页 ========== */
section[data-testid="stSidebar"] {{
  background:
    linear-gradient(90deg, #2A342E 0%, var(--spine) 55%, #38443D 100%);
  border-right: 1px solid #223028;
  box-shadow: inset -14px 0 22px -18px rgba(0,0,0,.9);
}}
section[data-testid="stSidebar"] > div {{ padding-top: .6rem; }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
  padding-bottom: 3.5rem;
}}
section[data-testid="stSidebar"] * {{ color: #E7E2D2 !important; }}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ display: none; }}
section[data-testid="stSidebar"] hr {{
  border: none; border-top: 1px solid rgba(231,226,210,.16);
  margin: 1.05rem 0 .9rem;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] .stMarkdown {{ color: #DCD6C4 !important; font-size: 12.5px; }}

/* 侧边栏里的提示框压成描边小字 */
section[data-testid="stSidebar"] [data-testid="stAlert"] {{
  background: rgba(231,226,210,.07) !important;
  border: 1px solid rgba(231,226,210,.22) !important;
  border-radius: 3px; padding: .5rem .7rem;
}}
section[data-testid="stSidebar"] [data-testid="stAlert"] p {{
  font-size: 12.5px !important; letter-spacing: .04em;
}}
section[data-testid="stSidebar"] [data-testid="stAlert"] svg {{ display: none; }}

/* 侧边栏按钮：细描边 */
section[data-testid="stSidebar"] .stButton > button {{
  background: transparent; border: 1px solid rgba(231,226,210,.34);
  color: #E7E2D2; border-radius: 2px; padding: .34rem .5rem;
  font-family: {FONT_UI}; font-size: 12px; letter-spacing: .16em;
  transition: all .2s ease;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
  background: rgba(231,226,210,.12); border-color: rgba(231,226,210,.62);
  color: #FFF8E2;
}}

/* 书脊铭牌 */
.spine-plate {{
  text-align: center; padding: 6px 0 2px;
  border-top: 1px solid rgba(231,226,210,.22);
  border-bottom: 1px solid rgba(231,226,210,.22);
  margin: 2px 0 14px;
}}
.spine-plate .sp-name {{
  writing-mode: vertical-rl; text-orientation: upright;
  font-family: {FONT_HEAD}; font-size: 17px; letter-spacing: .26em;
  line-height: 1.5; color: #F2EBD8 !important;
  margin: 16px auto 12px; height: 208px;
}}
.spine-plate .sp-seal {{
  display: inline-block; line-height: 20px; padding: 0 5px;
  white-space: nowrap; vertical-align: middle;
  background: var(--seal); color: #FBEFE4 !important; border-radius: 2px;
  font-family: {FONT_HEAD}; font-size: 11px; letter-spacing: .1em;
  opacity: .92;
}}
.spine-plate .sp-sub {{
  font-family: {FONT_UI}; font-size: 10px; letter-spacing: .3em;
  color: rgba(231,226,210,.56) !important; margin: 10px 0 6px;
}}

/* 侧边栏状态行 */
.status-line {{
  display: flex; align-items: center; gap: 8px;
  font-family: {FONT_UI}; font-size: 12.5px; letter-spacing: .06em;
  margin: 4px 0 10px;
}}
.status-line .dot {{
  width: 7px; height: 7px; border-radius: 50%; flex: 0 0 auto;
  box-shadow: 0 0 0 3px rgba(255,255,255,.06);
  animation: pulse 2.6s ease-in-out infinite;
}}
.status-line .dot.ok {{ background: #7FA98F; }}
.status-line .dot.bad {{ background: var(--seal); }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: .45; }} }}

/* 侧边栏引文式说明 */
.side-note {{
  border-left: 2px solid rgba(231,226,210,.28);
  padding: 2px 0 2px 10px; margin: 6px 0 2px;
}}
.side-note .sn-title {{
  font-family: {FONT_HEAD}; font-size: 12px; letter-spacing: .22em;
  color: rgba(231,226,210,.72) !important; margin-bottom: 7px;
}}
.side-note ol {{
  margin: 0; padding-left: 15px; counter-reset: sn;
}}
.side-note li {{
  list-style: none; position: relative; counter-increment: sn;
  font-size: 12px; line-height: 1.85; color: #D6D0BE !important;
}}
.side-note li::before {{
  content: counter(sn); position: absolute; left: -15px; top: 0;
  font-family: {FONT_HEAD}; font-size: 10px; color: rgba(231,226,210,.5) !important;
}}

/* ========== 顶部刊头 ========== */
.book-masthead {{
  text-align: center; margin: 4px 0 16px; padding-bottom: 12px;
  border-bottom: 1px solid var(--rule); position: relative;
}}
.book-masthead .bm-title {{
  font-family: {FONT_HEAD}; font-size: 23px; letter-spacing: .16em;
  color: var(--ink); line-height: 1.5;
}}
.book-masthead .bm-sub {{
  font-family: {FONT_UI}; font-size: 11.5px; letter-spacing: .22em;
  color: var(--ink-faint); margin-top: 6px;
}}
.book-masthead .bm-rule {{
  width: 46px; height: 2px; background: var(--seal);
  margin: 10px auto 0; opacity: .8;
}}

/* ========== 页签：书口标签 + 纸页 ========== */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 3px; background: transparent; padding: 0 14px;
  border-bottom: none !important;
}}
[data-testid="stTabs"] button[data-baseweb="tab"] {{
  background: #E7E1CE;
  border: 1px solid #D8D0B6;
  border-bottom: 1px solid #D8D0B6;
  border-radius: 5px 5px 0 0;
  padding: 8px 20px 7px;
  font-family: {FONT_HEAD}; font-size: 14px; letter-spacing: .1em;
  color: #6E6853; height: auto; transition: background .18s ease, color .18s ease;
}}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
  background: #EFEAD9; color: var(--ink);
}}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
  background: var(--paper); color: var(--seal);
  border-bottom-color: var(--paper);
  box-shadow: inset 0 2px 0 var(--seal);
  margin-bottom: -2px; padding-bottom: 9px;
  position: relative; z-index: 3;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none !important; }}

/* 纸页本体 */
[data-testid="stTabs"] [role="tabpanel"] {{
  background:
    repeating-linear-gradient(0deg, rgba(120,105,70,.014) 0 1px, transparent 1px 5px),
    var(--paper);
  border: 1px solid #D8D0B6;
  border-radius: 0 6px 6px 6px;
  box-shadow:
    0 1px 0 #FFFDF6 inset,
    0 14px 34px -22px rgba(70,58,30,.5),
    0 2px 6px -4px rgba(70,58,30,.28);
  padding: 30px 40px 38px;
  margin-top: -1px;
  min-height: 58vh;
  position: relative;
}}
/* 纸页页眉：细线 + 小字 */
[data-testid="stTabs"] [role="tabpanel"]::before {{
  content: "课 程 讲 义 · COURSE READER";
  display: block; text-align: center;
  font-family: {FONT_UI}; font-size: 9.5px; letter-spacing: .34em;
  color: #B0A78E; padding-bottom: 12px; margin-bottom: 22px;
  border-bottom: 1px solid rgba(176,167,142,.34);
}}

/* ========== 标题层级 ========== */
.stMarkdown h3, [data-testid="stHeadingWithActionElements"] h3 {{
  font-family: {FONT_HEAD} !important; font-size: 17px !important;
  letter-spacing: .1em; color: var(--ink);
  padding-left: 11px; border-left: 3px solid var(--seal);
  margin-top: 4px !important;
}}
.stMarkdown h4 {{ font-family: {FONT_HEAD} !important; font-size: 14.5px !important; letter-spacing: .06em; }}
.stMarkdown h4::before {{
  content: "◆"; color: var(--seal); font-size: 9px;
  margin-right: 7px; vertical-align: 2px; opacity: .75;
}}
[data-testid="stCaptionContainer"] p,
.stCaption {{ font-family: {FONT_UI}; font-size: 11.5px !important; color: var(--ink-faint) !important; letter-spacing: .02em; }}

/* 正文段落 */
[data-testid="stMarkdownContainer"] p {{
  font-family: {FONT_BODY}; font-size: 14px; line-height: 2.0;
  color: #33302A; text-align: justify;
}}

/* ========== 容器：去掉礼盒，改成排版块 ========== */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border: none !important;
  border-top: 1px solid rgba(176,167,142,.45) !important;
  border-radius: 0 !important;
  padding: 4px 0 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}}

/* ========== 表单控件：拉平成「题在上、线在下」 ========== */
[data-testid="stTextInputRootElement"],
[data-testid="stTextAreaRootElement"] {{
  background: rgba(176,167,142,.06) !important;
  border: none !important;
  border-bottom: 1px solid #BFB49A !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  transition: border-color .18s ease, background .18s ease;
}}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within {{
  border-bottom-color: var(--seal) !important;
  background: rgba(168,68,42,.035) !important;
}}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {{
  background: transparent !important;
  border: none !important; border-radius: 0 !important;
  padding: .38rem .55rem !important;
  font-family: {FONT_BODY} !important; font-size: 14px !important;
  color: var(--ink) !important;
  box-shadow: none !important;
}}
/* 占位示例常有多行，默认高度会把最后一行压到拉伸手柄上 */
[data-testid="stTextArea"] textarea {{
  line-height: 1.7 !important;
  min-height: 86px !important;
  padding-bottom: .7rem !important;
  resize: none !important;
}}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {{
  color: #A79E88 !important; font-family: {FONT_BODY} !important;
}}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label > div p {{
  font-family: {FONT_UI} !important; font-size: 11.5px !important;
  letter-spacing: .16em; color: var(--ink-faint) !important;
}}
[data-baseweb="select"] > div {{
  background: transparent !important; border: none !important;
  border-bottom: 1px solid #BFB49A !important; border-radius: 0 !important;
  font-family: {FONT_BODY} !important; font-size: 14px !important;
  box-shadow: none !important; min-height: 34px;
}}
/* 日期输入框 */
[data-testid="stDateInput"] > div > div,
[data-testid="stDateInputField"] {{
  background: transparent !important; border: none !important;
  border-bottom: 1px solid #BFB49A !important; border-radius: 0 !important;
  box-shadow: none !important; font-family: {FONT_BODY} !important;
}}
/* 多选：选中项做成朱印小签（描边而非实心，避免整页出现大块红） */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {{
  background: rgba(168,68,42,.06) !important;
  border: 1px solid rgba(168,68,42,.42) !important;
  border-radius: 2px !important;
  font-family: {FONT_UI} !important; font-size: 11.5px !important;
  letter-spacing: .06em;
}}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {{
  color: var(--seal) !important; fill: var(--seal) !important;
  background: transparent !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
  background: var(--seal) !important;
}}
[data-testid="stRadio"] label p, [data-testid="stRadio"] label div {{
  font-family: {FONT_BODY} !important; font-size: 14px !important; line-height: 1.9;
}}
[data-testid="stCheckbox"] label p {{
  font-family: {FONT_UI} !important; font-size: 12.5px !important;
  letter-spacing: .06em; color: var(--ink-soft) !important;
}}
/* 题目之间的细分隔线 */
.q-sep {{
  border-top: 1px dotted rgba(176,167,142,.7);
  margin: 4px 0 2px;
}}

/* ========== 按钮：主操作为朱印，次要为描边 ========== */
/* 按钮内的 markdown 段落要继承按钮色，否则会被正文色覆盖 */
.stButton > button p, .stDownloadButton > button p, .stFormSubmitButton > button p,
.stButton > button div, .stDownloadButton > button div, .stFormSubmitButton > button div {{
  color: inherit !important; font-family: inherit !important;
  font-size: inherit !important; letter-spacing: inherit !important; margin: 0 !important;
}}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  background: transparent; border: 1px solid #C3B89C; color: var(--ink-soft);
  border-radius: 2px; padding: .42rem 1.25rem;
  font-family: {FONT_HEAD}; font-size: 13px; letter-spacing: .18em;
  box-shadow: none;
  transition: transform .14s ease, box-shadow .14s ease, background .14s ease, color .14s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{
  background: rgba(168,68,42,.07); color: var(--seal); border-color: var(--seal);
}}
.stButton > button[data-testid="stBaseButton-primary"],
.stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"],
.stDownloadButton > button[data-testid="stBaseButton-primary"] {{
  background: var(--seal); border: 1px solid #8E3823; color: #FBEFE4;
  box-shadow: 0 1px 0 rgba(255,255,255,.18) inset, 0 2px 5px -2px rgba(120,40,20,.5);
}}
.stButton > button[data-testid="stBaseButton-primary"]:hover,
.stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
.stDownloadButton > button[data-testid="stBaseButton-primary"]:hover {{
  background: #B84B2E; color: #FFF6EC;
  transform: translateY(-1px);
  box-shadow: 0 1px 0 rgba(255,255,255,.22) inset, 0 5px 12px -4px rgba(120,40,20,.55);
}}
.stButton > button:active {{ transform: translateY(0); }}
.stButton > button:disabled,
.stFormSubmitButton > button:disabled {{
  opacity: 1 !important; pointer-events: none;
  border-style: dashed !important; border-color: #D3C9B0 !important;
  color: #A79E88 !important;
}}
.stButton > button[data-testid="stBaseButton-primary"]:disabled,
.stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"]:disabled {{
  background: #C8BCA8 !important; border: 1px solid #B9AC95 !important;
  color: #FBF6EA !important; box-shadow: none !important;
}}

/* 引用脚注里的「在原文中查看」小按钮 */
.foot-actions .stButton > button {{
  background: transparent; border: none; border-bottom: 1px dotted #B9AE93;
  color: var(--jade) !important; border-radius: 0; box-shadow: none;
  font-family: {FONT_UI}; font-size: 11.5px; letter-spacing: .04em;
  padding: 0 0 1px; min-height: 0; line-height: 1.6;
}}
.foot-actions .stButton > button:hover {{
  background: transparent; color: var(--seal) !important; border-bottom-color: var(--seal);
  transform: none;
}}

/* ========== 文件上传区 ========== */
[data-testid="stFileUploaderDropzone"] {{
  background: rgba(176,167,142,.055) !important;
  border: 1px dashed #BFB49A !important; border-radius: 3px !important;
  padding: 16px !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] {{
  display: flex !important; flex-direction: column !important;
  align-items: flex-start !important; gap: 3px;
}}
/* 隐藏英文图标与原文案，用伪元素叠加中文 */
[data-testid="stFileUploaderDropzoneInstructions"] > span,
[data-testid="stFileUploaderDropzoneInstructions"] > div {{ display: none !important; }}
[data-testid="stFileUploaderDropzoneInstructions"]::before {{
  content: "将文件拖拽到此处上传";
  display: block; font-family: {FONT_HEAD}; font-size: 13.5px;
  color: var(--ink-soft); letter-spacing: .14em;
}}
[data-testid="stFileUploaderDropzoneInstructions"]::after {{
  content: "支持 PDF / PPTX / DOCX / TXT，单文件不超过 20MB";
  display: block; font-family: {FONT_UI}; font-size: 11px;
  color: var(--ink-faint);
}}
[data-testid="stFileUploaderDropzone"] button {{
  background: transparent !important; border: 1px solid var(--seal) !important;
  color: transparent !important; border-radius: 2px !important;
  padding: .3rem .9rem !important; box-shadow: none !important;
  font-size: 0 !important; min-height: 0;
}}
[data-testid="stFileUploaderDropzone"] button::after {{
  content: "选择文件"; font-family: {FONT_HEAD}; font-size: 12.5px;
  letter-spacing: .14em; color: var(--seal);
}}
[data-testid="stFileUploaderFileName"] {{ font-family: {FONT_BODY}; font-size: 13px; word-break: break-all; }}

/* ========== 表单容器：去掉卡片感，只留一条细线 ========== */
[data-testid="stForm"] {{
  border: none !important; border-radius: 0 !important;
  padding: 0 !important; background: transparent !important;
}}

/* ========== 表格 ========== */
[data-testid="stDataFrame"] {{
  border: 1px solid rgba(176,167,142,.4); border-radius: 2px;
}}
[data-testid="stDataFrame"] * {{ font-family: {FONT_BODY} !important; font-size: 12.5px !important; }}

/* ========== 提示条：改成批注条，按类型换色 ========== */
/* 底色画在 stAlertContainer 上，外层 stAlert 只留左侧竖线 */
[data-testid="stAlert"] {{
  --alert-accent: #4F6B62;
  --alert-tint: rgba(79,107,98,.06);
  border-radius: 2px !important;
  border: none !important;
  background: var(--alert-tint) !important;
  /* 左侧批注竖线用内阴影画，避免被圆角裁掉 */
  box-shadow: inset 3px 0 0 var(--alert-accent) !important;
  padding: 0 !important;
  overflow: hidden;
}}
[data-testid="stAlertContainer"] {{
  background: rgba(79,107,98,.06) !important;
  border: none !important; border-radius: 0 !important;
  padding: .55rem .85rem !important;
  box-shadow: none !important;
}}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {{
  font-size: 12.5px !important; line-height: 1.85;
  color: var(--ink-soft) !important;
}}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {{
  --alert-accent: #4F6B62; --alert-tint: rgba(79,107,98,.06);
}}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {{
  --alert-accent: #5F7F6B; --alert-tint: rgba(95,127,107,.075);
}}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {{
  --alert-accent: #A98A3E; --alert-tint: rgba(169,138,62,.075);
}}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{
  --alert-accent: #A8442A; --alert-tint: rgba(168,68,42,.06);
}}
[data-testid="stAlert"] [data-testid="stAlertContainer"] {{
  background: var(--alert-tint) !important;
}}
/* 提示图标缩小并去掉默认彩色，避免抢眼 */
[data-testid="stAlert"] [data-testid="stIconMaterial"] {{
  font-size: 14px !important; color: var(--ink-faint) !important;
  opacity: .65;
}}

/* ========== 脚注区 ========== */
.footnotes {{
  margin-top: 18px; padding-top: 12px;
  border-top: 1px solid rgba(176,167,142,.45);
}}
.footnotes .fn-head {{
  font-family: {FONT_HEAD}; font-size: 11px; letter-spacing: .3em;
  color: var(--ink-faint); margin-bottom: 9px;
}}
.footnotes .fn-item {{
  display: flex; gap: 9px; align-items: baseline;
  font-size: 12.5px; line-height: 1.95; color: var(--ink-soft);
  padding: 8px 0 7px; border-bottom: 1px dotted rgba(176,167,142,.4);
}}
.footnotes .fn-item:last-child {{ border-bottom: none; }}
.footnotes .fn-idx {{
  font-family: {FONT_HEAD}; font-size: 11px; color: var(--seal);
  flex: 0 0 auto; padding-top: 1px;
}}
.footnotes .fn-body {{ flex: 1 1 auto; min-width: 0; }}
.footnotes .fn-src {{ color: var(--ink); }}
.footnotes .fn-meta {{ color: var(--ink-faint); font-family: {FONT_UI}; font-size: 11px; letter-spacing: .04em; }}
.footnotes .fn-quote {{
  display: block; margin-top: 3px; padding-left: 9px;
  border-left: 2px solid rgba(168,68,42,.3);
  font-size: 12px; line-height: 1.9; color: #55503F;
}}

/* ========== 问答条目 ========== */
.qa {{ margin: 0 0 26px; }}
.qa .q {{ display: flex; gap: 10px; align-items: baseline; margin-bottom: 8px; }}
.qa .q-tag {{
  flex: 0 0 auto; width: 20px; height: 20px; line-height: 19px; text-align: center;
  border: 1px solid var(--seal); color: var(--seal);
  border-radius: 2px; font-family: {FONT_HEAD}; font-size: 11px;
}}
.qa .q-text {{
  font-family: {FONT_HEAD}; font-size: 15px; letter-spacing: .04em;
  color: var(--ink); line-height: 1.75;
}}
.qa .a {{ font-family: {FONT_BODY}; font-size: 14px; line-height: 2.05; color: #33302A; text-align: justify; }}
.qa .a p {{ margin: 0 0 9px; text-indent: 2em; }}
.qa .a p:last-child {{ margin-bottom: 0; }}
.qa .a p.a-h {{
  text-indent: 0; font-family: {FONT_HEAD}; font-size: 13.5px;
  letter-spacing: .08em; color: var(--ink); margin: 12px 0 6px;
}}
.qa .a p.a-h::before {{
  content: "◆"; color: var(--seal); font-size: 8px;
  margin-right: 7px; vertical-align: 2px; opacity: .7;
}}
.qa .a strong {{ font-weight: 600; color: #1F1D18; }}
.qa .a code {{
  font-family: Consolas, "Courier New", monospace; font-size: 12.5px;
  background: rgba(176,167,142,.16); padding: 1px 4px; border-radius: 2px;
}}
.qa .a .a-list {{ margin: 4px 0 10px; padding-left: 1.2em; }}
.qa .a .a-li {{
  position: relative; font-size: 13.8px; line-height: 1.95;
  color: #33302A; text-align: justify;
}}
.qa .a .a-li::before {{
  content: "·"; position: absolute; left: -.9em; color: var(--seal); font-weight: 700;
}}
.qa .a sup.fnref {{
  font-family: {FONT_UI}; font-size: 9.5px; color: var(--seal);
  margin-left: 1px; vertical-align: super; letter-spacing: 0;
}}
.qa .qa-meta {{
  margin-top: 9px; font-family: {FONT_UI}; font-size: 10.5px;
  letter-spacing: .07em; color: var(--ink-faint);
}}

/* ========== 计划：边注式 ========== */
.plan-head {{
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 16px; padding-bottom: 10px; margin-bottom: 4px;
  border-bottom: 1px solid rgba(176,167,142,.5);
}}
.plan-head .ph-title {{ font-family: {FONT_HEAD}; font-size: 15px; letter-spacing: .12em; }}
.plan-head .ph-meta {{ font-family: {FONT_UI}; font-size: 11px; color: var(--ink-faint); letter-spacing: .06em; }}
.plan-day {{
  display: flex; gap: 18px; padding: 13px 0;
  border-bottom: 1px dotted rgba(176,167,142,.55);
}}
.plan-day:last-child {{ border-bottom: none; }}
.plan-day .pd-date {{
  flex: 0 0 92px; text-align: right;
}}
.plan-day .pd-day {{
  font-family: {FONT_HEAD}; font-size: 19px; color: var(--jade);
  letter-spacing: .04em; line-height: 1.25;
}}
.plan-day .pd-week {{ font-family: {FONT_UI}; font-size: 10px; color: var(--ink-faint); letter-spacing: .12em; }}
.plan-day .pd-body {{
  flex: 1 1 auto; min-width: 0; padding-left: 18px;
  border-left: 1px solid rgba(176,167,142,.42);
}}
.plan-day .pd-task {{
  font-family: {FONT_BODY}; font-size: 13.5px; line-height: 1.95;
  color: #33302A; display: flex; gap: 8px;
}}
.plan-day .pd-task::before {{
  content: "○"; flex: 0 0 auto; color: #B5A886; font-size: 9px; padding-top: 5px;
}}
.plan-day .pd-focus {{
  margin-top: 6px; font-family: {FONT_UI}; font-size: 11px;
  color: var(--ink-faint); letter-spacing: .04em;
}}
.plan-day .pd-focus::before {{
  content: "重点"; display: inline-block; margin-right: 7px;
  border: 1px solid rgba(176,167,142,.8); border-radius: 2px;
  padding: 0 4px; font-size: 9.5px; letter-spacing: .1em; color: #9A9078;
}}

/* ========== 判分：朱批 ========== */
.score-strip {{
  display: flex; align-items: center; gap: 26px;
  padding: 4px 0 14px; margin-bottom: 6px;
  border-bottom: 1px solid rgba(176,167,142,.5);
}}
.score-strip .ss-main {{ font-family: {FONT_HEAD}; font-size: 26px; letter-spacing: .02em; }}
.score-strip .ss-main em {{ font-style: normal; font-size: 14px; color: var(--ink-faint); margin-left: 3px; }}
.score-strip .ss-item {{ font-family: {FONT_UI}; font-size: 11.5px; color: var(--ink-faint); letter-spacing: .08em; }}
.score-strip .ss-item b {{ font-family: {FONT_HEAD}; font-size: 15px; color: var(--ink); font-weight: normal; margin-left: 5px; }}
.score-strip .ss-stamp {{
  margin-left: auto; font-family: {FONT_HEAD}; font-size: 12px;
  color: var(--seal); border: 1px solid var(--seal); border-radius: 2px;
  padding: 3px 9px; letter-spacing: .16em; transform: rotate(-3deg); opacity: .88;
}}

.q-head {{ display: flex; gap: 9px; align-items: baseline; margin-bottom: 7px; }}
.q-head .qh-no {{
  font-family: {FONT_HEAD}; font-size: 13px; color: var(--seal);
  flex: 0 0 auto; letter-spacing: .04em;
}}
.q-head .qh-type {{
  font-family: {FONT_UI}; font-size: 10px; color: var(--ink-faint);
  border: 1px solid rgba(176,167,142,.7); border-radius: 2px;
  padding: 1px 5px; letter-spacing: .1em; flex: 0 0 auto;
}}
.q-head .qh-stem {{ font-family: {FONT_BODY}; font-size: 14px; line-height: 1.95; color: #33302A; }}
.qh-hint {{
  margin: 3px 0 9px 32px; font-family: {FONT_UI}; font-size: 11.5px;
  color: #9A9078; letter-spacing: .03em; line-height: 1.7;
}}
.qh-hint span {{
  border: 1px solid rgba(176,167,142,.75); border-radius: 2px;
  padding: 0 4px; margin-right: 7px; font-size: 9.5px;
  letter-spacing: .1em; color: #A2977F;
}}

.mark {{
  margin-top: 10px; padding: 9px 12px 10px;
  border-left: 2px solid #8FA79B; background: rgba(79,107,98,.05);
  font-size: 12.5px; line-height: 1.95;
}}
.mark.bad {{ border-left-color: var(--seal); background: rgba(168,68,42,.05); }}
.mark .mk-verdict {{
  font-family: {FONT_HEAD}; font-size: 12.5px; letter-spacing: .14em;
  color: var(--jade); margin-bottom: 5px;
}}
.mark.bad .mk-verdict {{ color: var(--seal); }}
.mark .mk-key {{ color: var(--ink); }}
.mark .mk-key b {{
  font-family: {FONT_HEAD}; font-weight: normal; color: var(--seal);
  padding: 1px 6px; background: rgba(168,68,42,.09); border-radius: 2px;
}}
.mark .mk-exp {{ color: var(--ink-soft); margin-top: 4px; }}
.mark .mk-src {{ margin-top: 6px; font-family: {FONT_UI}; font-size: 10.5px; color: var(--ink-faint); letter-spacing: .04em; }}

/* 空态 */
.empty-note {{
  text-align: center; padding: 26px 0;
  font-family: {FONT_HEAD}; font-size: 13px; letter-spacing: .16em;
  color: var(--ink-faint);
}}

/* ========== 移动端（≤ 640px） ========== */
/* 手机上没有足够的横向空间留给双层留白：纸页内边距减半、页签改成可横滑的书口、
   表单列改为整行堆叠，字号整体下调一档但保留 serif 气质。 */
@media (max-width: 640px) {{
  [data-testid="stAppViewBlockContainer"],
  .block-container {{
    /* 手机上 Streamlit 顶栏（汉堡菜单）是浮层，留白不够会把报头标题压到它下面 */
    padding: 3.4rem .7rem 2rem !important;
    max-width: 100% !important;
  }}

  /* 报头：字号收紧，副标题不再占两行 */
  .book-masthead {{ margin: 0 0 10px; padding-bottom: 8px; }}
  .book-masthead .bm-title {{ font-size: 17px; letter-spacing: .08em; line-height: 1.45; }}
  .book-masthead .bm-sub {{ font-size: 10px; letter-spacing: .1em; margin-top: 4px; }}
  .book-masthead .bm-rule {{ width: 34px; margin-top: 7px; }}

  /* 页签：单行横滑，标签收窄；滚动条隐藏避免出现一条灰线 */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {{
    overflow-x: auto; overflow-y: hidden; flex-wrap: nowrap;
    padding: 0 2px; gap: 1px; scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }}
  [data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}
  [data-testid="stTabs"] button[data-baseweb="tab"] {{
    flex: 0 0 auto; padding: 7px 6px 6px;
    font-size: 12px; letter-spacing: 0;
    border-radius: 4px 4px 0 0;
  }}
  [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
    padding-bottom: 8px;
  }}

  /* 纸页：内边距减到一半，页眉小字缩短字距 */
  [data-testid="stTabs"] [role="tabpanel"] {{
    padding: 16px 14px 22px;
    border-radius: 0 4px 4px 4px;
    min-height: 46vh;
  }}
  [data-testid="stTabs"] [role="tabpanel"]::before {{
    font-size: 8.5px; letter-spacing: .18em;
    padding-bottom: 8px; margin-bottom: 14px;
  }}

  /* 表单列在窄屏堆叠，否则「课程 / 章节」两个下拉会被压成一条缝 */
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: .4rem !important; }}
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
    flex: 1 1 100% !important; width: 100% !important;
    min-width: 100% !important;
  }}

  /* 排版细节 */
  .stMarkdown h3, [data-testid="stHeadingWithActionElements"] h3 {{
    font-size: 15px !important; margin: .8rem 0 .5rem;
  }}
  .stMarkdown h4 {{ font-size: 13.5px !important; }}
  .q-head {{ gap: 7px; }}
  .q-head .qh-stem {{ font-size: 13.5px; line-height: 1.9; }}
  .qh-hint {{ margin-left: 0; }}

  /* 边注式计划：日期列收窄，任务正文不再二次缩进 */
  .plan-day {{ gap: 10px; padding: 11px 0; }}
  .plan-day .pd-date {{ flex: 0 0 54px; }}
  .plan-day .pd-day {{ font-size: 16px; }}
  .plan-day .pd-body {{ padding-left: 10px; }}
  .plan-day .pd-task {{ font-size: 13px; line-height: 1.85; }}

  /* 得分条：允许换行，印记不再被挤到屏幕外 */
  .score-strip {{ gap: 12px 18px; flex-wrap: wrap; padding-bottom: 10px; }}
  .score-strip .ss-main {{ font-size: 22px; }}
  .score-strip .ss-stamp {{ margin-left: 0; }}

  .mark {{ padding: 8px 10px 9px; font-size: 12px; }}
  .footnotes .fn-item {{ font-size: 12px; }}
  .qa .a p, .qa .a .a-li {{ font-size: 13.5px; line-height: 1.9; }}
  .qa .a p.a-h {{ font-size: 13px; }}

  /* 侧边栏在小屏是一张抽屉，书脊竖排太长会顶到屏幕外 */
  .spine-plate .sp-name {{ font-size: 15px; height: 150px; margin: 10px auto 8px; }}
  .side-note li {{ font-size: 12px; }}
}}
</style>
"""


# ---------------------------------------------------------------------------
# HTML 片段
# ---------------------------------------------------------------------------
def masthead(title: str, subtitle: str) -> str:
    """顶部刊头：书名 + 副题 + 朱色短线。"""
    return (
        '<div class="book-masthead">'
        f'<div class="bm-title">{esc(title)}</div>'
        f'<div class="bm-sub">{esc(subtitle)}</div>'
        '<div class="bm-rule"></div>'
        "</div>"
    )


def spine_plate(name: str, subtitle: str, seal: str) -> str:
    """侧边栏书脊铭牌：竖排书名 + 朱印。"""
    return (
        '<div class="spine-plate">'
        f'<div class="sp-name">{esc(name)}</div>'
        f'<div class="sp-seal">{esc(seal)}</div>'
        f'<div class="sp-sub">{esc(subtitle)}</div>'
        "</div>"
    )


def status_line(ok: bool, text: str) -> str:
    """侧边栏状态行：圆点 + 文字。"""
    return (
        f'<div class="status-line"><span class="dot {"ok" if ok else "bad"}"></span>'
        f"<span>{esc(text)}</span></div>"
    )


def side_note(title: str, steps: Iterable[str]) -> str:
    """侧边栏引文式步骤说明。"""
    items = "".join(f"<li>{esc(step)}</li>" for step in steps if str(step).strip())
    return (
        '<div class="side-note">'
        f'<div class="sn-title">{esc(title)}</div>'
        f"<ol>{items}</ol>"
        "</div>"
    )


def book_title(value: str, number: int) -> str:
    """页内小节标题：中文序号 + 朱点，替代 st.subheader。"""
    mark = CN_NUM[number - 1] if 1 <= number <= len(CN_NUM) else str(number)
    return (
        f'<div style="display:flex;align-items:baseline;gap:10px;margin:0 0 4px;">'
        f'<span style="font-family:{FONT_HEAD};font-size:13px;color:{SEAL};letter-spacing:.1em;">{mark}</span>'
        f'<span style="font-family:{FONT_HEAD};font-size:17px;letter-spacing:.12em;color:{INK};">{esc(value)}</span>'
        "</div>"
    )


def footnotes(citations: list[dict[str, Any]]) -> str:
    """脚注区：序号 + 来源 + 页码 + 原文摘录。"""
    if not citations:
        return ""
    rows: list[str] = []
    for item in citations:
        index = esc(item.get("index") or "")
        source = esc(item.get("source") or "未知来源")
        meta_bits = [b for b in (item.get("page"), item.get("locator")) if b]
        meta = " ｜ ".join(esc(b) for b in meta_bits)
        excerpt = str(item.get("excerpt") or "").strip()
        quote = f'<span class="fn-quote">{esc(excerpt)}</span>' if excerpt else ""
        rows.append(
            '<div class="fn-item">'
            f'<span class="fn-idx">[{index}]</span>'
            '<span class="fn-body">'
            f'<span class="fn-src">{source}</span> '
            f'<span class="fn-meta">{meta}</span>'
            f"{quote}"
            "</span></div>"
        )
    return (
        '<div class="footnotes"><div class="fn-head">参考来源</div>'
        + "".join(rows)
        + "</div>"
    )


def qa_block(
    question: str,
    answer: str,
    meta: str = "",
    *,
    answer_html: str | None = None,
    number: int = 1,
) -> str:
    """一条问答：问号方印 + 题干 + 正文（可按句拆段）。"""
    body = answer_html if answer_html is not None else f"<p>{esc(answer)}</p>"
    meta_html = f'<div class="qa-meta">{esc(meta)}</div>' if meta else ""
    return (
        '<div class="qa">'
        '<div class="q">'
        f'<span class="q-tag">{number}</span>'
        f'<span class="q-text">{esc(question)}</span>'
        "</div>"
        f'<div class="a">{body}</div>'
        f"{meta_html}"
        "</div>"
    )


_CITE_RE = re.compile(
    r"[\[【]\s*来源\s*[:：]\s*(?P<src>[^\]】]+?)\s*[\]】]"
)
_BOLD_RE = re.compile(r"\*\*(?P<text>[^*]+?)\*\*")
_CODE_RE = re.compile(r"`(?P<code>[^`]+)`")
_BULLET_RE = re.compile(r"^[-•*·]\s+")
# 模型输出里常见「第4页」「第 4 页」，从来源串里剥出来单独标注
_PAGE_RE = re.compile(r"^(?P<name>.*?)\s*第\s*(?P<page>\d+)\s*页$")


def _inline(text: str) -> str:
    """行内 markdown → HTML（调用前文本已转义）。"""
    text = _BOLD_RE.sub(r"<strong>\g<text></strong>", text)
    text = _CODE_RE.sub(r"<code>\g<code></code>", text)
    return text


def citation_keys(text: str) -> list[str]:
    """按出现顺序提取正文里的来源键（``文件名|页码``），已去重。"""
    keys: list[str] = []
    for match in _CITE_RE.finditer(str(text or "")):
        raw = match.group("src").strip()
        name, page = raw, ""
        page_match = _PAGE_RE.match(raw)
        if page_match:
            name, page = page_match.group("name"), page_match.group("page")
        key = f"{name}|{page}"
        if key not in keys:
            keys.append(key)
    return keys


def answer_paragraphs(
    answer: str,
    *,
    citation_map: dict[str, Any] | None = None,
    max_paragraphs: int = 40,
) -> str:
    """把答案渲染成排版好的 HTML。

    - ``[来源:文件名 第N页]`` 换成上标角标，编号与脚注一致；
    - 连续重复的角标只保留一个，避免满屏数字；
    - ``**加粗**``、行首 ``- `` 列表、``###`` 小标题按轻量 markdown 处理。
    """
    text = str(answer or "").strip()
    if not text:
        return "<p>（无回答内容）</p>"

    mapping = citation_map or {}
    counter = {"n": 0}

    def repl(match: re.Match[str]) -> str:
        raw = match.group("src").strip()
        page = ""
        name = raw
        page_match = _PAGE_RE.match(raw)
        if page_match:
            name, page = page_match.group("name"), page_match.group("page")
        key = f"{name}|{page}"
        index = mapping.get(key)
        if index is None:
            counter["n"] += 1
            index = counter["n"]
        return f'<sup class="fnref">[{esc(index)}]</sup>'

    escaped = esc(text)
    escaped = _CITE_RE.sub(repl, escaped)
    # 连续重复角标折叠
    escaped = re.sub(r'(<sup class="fnref">\[\d+\]</sup>)(?:\s*\1)+', r"\1", escaped)

    blocks = [block.strip() for block in re.split(r"\n\s*\n", escaped) if block.strip()]
    if not blocks:
        blocks = [escaped]

    parts: list[str] = []
    for block in blocks[:max_paragraphs]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) == 1 and lines[0].startswith("#"):
            title = lines[0].lstrip("#").strip()
            parts.append(f'<p class="a-h">{_inline(title)}</p>')
            continue
        bullets = [line for line in lines if _BULLET_RE.match(line)]
        if bullets and len(bullets) >= max(2, len(lines) - 1):
            items = "".join(
                '<div class="a-li">' + _inline(_BULLET_RE.sub("", line)) + "</div>"
                for line in bullets
            )
            parts.append('<div class="a-list">' + items + "</div>")
            continue
        parts.append("<p>" + "<br>".join(_inline(line) for line in lines) + "</p>")
    return "".join(parts)


def plan_days(days: list[dict[str, str]]) -> str:
    """边注式计划：左侧日期大字，右侧任务列表。"""
    if not days:
        return '<div class="empty-note">暂无每日安排</div>'
    rows: list[str] = []
    for item in days:
        tasks = item.get("tasks") or []
        task_html = "".join(
            f'<div class="pd-task"><span>{esc(task)}</span></div>' for task in tasks
        ) or '<div class="pd-task"><span>（当日无任务）</span></div>'
        focus = item.get("focus") or ""
        focus_html = f'<div class="pd-focus">{esc(focus)}</div>' if focus else ""
        rows.append(
            '<div class="plan-day">'
            '<div class="pd-date">'
            f'<div class="pd-day">{esc(item.get("label") or "")}</div>'
            f'<div class="pd-week">{esc(item.get("weekday") or "")}</div>'
            "</div>"
            f'<div class="pd-body">{task_html}{focus_html}</div>'
            "</div>"
        )
    return "".join(rows)


def score_strip(score: int, total: int, accuracy: float, answered: int, stamp: str) -> str:
    """判分得分条。"""
    return (
        '<div class="score-strip">'
        f'<div class="ss-main">{score}<em>/ {total}</em></div>'
        f'<div class="ss-item">正确率<b>{accuracy:.0f}%</b></div>'
        f'<div class="ss-item">已作答<b>{answered}/{total}</b></div>'
        f'<div class="ss-stamp">{esc(stamp)}</div>'
        "</div>"
    )


def question_head(number: int, kind_label: str, stem: str, hint: str = "") -> str:
    """题干：序号 + 题型小签 + 题干正文。"""
    hint_html = (
        f'<div class="qh-hint"><span>提示</span>{esc(hint)}</div>' if hint else ""
    )
    return (
        '<div class="q-head">'
        f'<span class="qh-no">{number}.</span>'
        f'<span class="qh-type">{esc(kind_label)}</span>'
        f'<span class="qh-stem">{esc(stem)}</span>'
        "</div>"
        f"{hint_html}"
    )


def mark_block(
    *,
    correct: bool | None,
    correct_answer: str,
    explanation: str = "",
    source: str = "",
    user_answer: str = "",
) -> str:
    """朱批式判分块；``correct=None`` 表示未作答。"""
    if correct is None:
        verdict = "未作答"
        head = f'<span class="mk-verdict">○ 未作答</span>'
    elif correct:
        verdict = "正"
        head = '<span class="mk-verdict">✓ 回答正确</span>'
    else:
        verdict = "误"
        head = (
            '<span class="mk-verdict">✗ 回答不正确</span>'
            f'<span style="font-family:{FONT_UI};font-size:11px;color:{INK_FAINT};">'
            f'　你的答案：{esc(user_answer or "（空）")}</span>'
        )
    exp_html = (
        f'<div class="mk-exp">解析：{esc(explanation)}</div>' if explanation else ""
    )
    src_html = f'<div class="mk-src">{esc(source)}</div>' if source else ""
    cls = "mark" if (correct is True or correct is None) else "mark bad"
    return (
        f'<div class="{cls}">{head}'
        f'<div class="mk-key">正确答案：<b>{esc(correct_answer or "—")}</b></div>'
        f"{exp_html}{src_html}</div>"
    )
