# Kronos A 股预测面板 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `E:\workspace\python\kronos-forecast\` 用 uv + Gradio 构建深色科技风 A 股预测面板，用 Kronos-base 模型预测淳中科技、剑桥科技等任意 A 股日线。

**Architecture:** 逻辑层（predictor.py）与 UI 层（app.py）分离。predictor.py 负责腾讯接口拉数、Kronos 预测、涨跌停限幅、matplotlib 画图；app.py 负责 Gradio 界面与状态管理。Kronos `model/` 目录从已验证仓库复制进项目，非 pip 包。

**Tech Stack:** Python 3.12、uv、gradio、torch (CPU)、pandas、matplotlib、requests、huggingface_hub（hf-mirror 镜像）、Kronos-base + Kronos-Tokenizer-base。

## Global Constraints

- Python 3.12（uv 管理），Windows，纯 CPU，无 GPU
- 数据源固定腾讯接口 `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get`，后复权，价格恒为正
- 模型：`NeoQuasar/Kronos-base` + `NeoQuasar/Kronos-Tokenizer-base`，走 `HF_ENDPOINT=https://hf-mirror.com`
- 涨跌停限幅 ±10%
- 回看 400 交易日，默认预测 60 交易日（UI 可调 10-120）
- `model/` 目录内容必须与 Kronos 仓库原样一致，禁止改动（除非本计划明确要求）
- 预测约 1 分钟/股票，UI 需有进度反馈
- 所有代码注释用中文，无 em dash（—）
- 项目不建 git（用户未要求），提交步骤跳过；用文件变更记录替代

---

### Task 1: 项目脚手架与依赖

**Files:**
- Create: `E:\workspace\python\kronos-forecast\pyproject.toml`
- Create: `E:\workspace\python\kronos-forecast\.python-version`
- Create: `E:\workspace\python\kronos-forecast\README.md`
- Create: `E:\workspace\python\kronos-forecast\docs\REQUIREMENTS.md`（已存在，保持）
- Create: `E:\workspace\python\kronos-forecast\docs\PROGRESS.md`（已存在，保持）

**Interfaces:**
- Consumes: 无
- Produces: uv 项目骨架，`uv run python -c "import gradio, torch"` 可执行

- [ ] **Step 1: 创建目录与 pyproject.toml**

`E:\workspace\python\kronos-forecast\pyproject.toml`:
```toml
[project]
name = "kronos-forecast"
version = "0.1.0"
description = "Kronos A股预测面板，基于Kronos基础模型的金融K线预测"
requires-python = ">=3.12"
dependencies = [
    "gradio>=5.0",
    "torch>=2.2",
    "pandas>=2.1",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "requests>=2.31",
    "huggingface_hub>=0.23",
    "einops>=0.7",
    "safetensors>=0.4",
    "tqdm>=4.66",
]
```

`.python-version` 内容：`3.12`

- [ ] **Step 2: 初始化 uv 项目并安装依赖**

```powershell
cd E:\workspace\python\kronos-forecast
uv venv --python 3.12
uv pip install -e .
```

预期：依赖安装成功。若 torch 下载慢，设置 `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。

- [ ] **Step 3: 复制 Kronos model 目录**

从 `C:\Users\lcj\AppData\Local\Temp\opencode\Kronos\model\` 复制 `kronos.py`、`module.py`、`__init__.py` 到 `E:\workspace\python\kronos-forecast\model\`。

- [ ] **Step 4: 验证导入**

```powershell
uv run python -c "import gradio, torch, pandas, matplotlib; print('deps ok')"
uv run python -c "import sys; sys.path.append('.'); from model import Kronos, KronosTokenizer, KronosPredictor; print('kronos ok')"
```

预期：两行都打印 ok。

- [ ] **Step 5: 更新 PROGRESS.md**

在 PROGRESS.md 的里程碑表标记「项目搭建 ✅ 完成」。

---

### Task 2: 逻辑层 predictor.py

**Files:**
- Create: `E:\workspace\python\kronos-forecast\predictor.py`
- Create: `E:\workspace\python\kronos-forecast\stocks.py`

**Interfaces:**
- Consumes: `model/` 目录（Task 1）
- Produces:
  - `fetch_daily(code: str) -> pd.DataFrame`（列 date/open/high/low/close/volume/amount，date 为 datetime）
  - `load_model() -> KronosPredictor`（惰性单例，全局缓存）
  - `run_prediction(code: str, pred_len: int = 60) -> dict`（含 df_hist、df_pred、last_close、pred_end、pct、truncated、timings）
  - `plot_result(code, df_hist, df_pred) -> str`（返回 PNG 路径）
  - `stocks.py` 导出 `DEFAULT_STOCKS = {"淳中科技": "603516", "剑桥科技": "603083"}`

- [ ] **Step 1: 创建 stocks.py**

`E:\workspace\python\kronos-forecast\stocks.py`:
```python
# 内置股票池，加股票只改这一个字典
DEFAULT_STOCKS = {
    "淳中科技": "603516",
    "剑桥科技": "603083",
}
```

- [ ] **Step 2: 创建 predictor.py（数据拉取部分）**

```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
import time
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

LOOKBACK = 400
MAX_CONTEXT = 512
LIMIT_RATE = 0.10
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_NAME = "NeoQuasar/Kronos-base"


def market_prefix(code: str) -> str:
    """根据代码判断沪深前缀"""
    return "sh" if code.startswith(("6", "9")) else "sz"


def fetch_daily(code: str, retries: int = 3, timeout: int = 15) -> pd.DataFrame:
    """腾讯后复权日线接口，返回 date/open/high/low/close/volume/amount"""
    sym = f"{market_prefix(code)}{code}"
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{sym},day,,,2000,qfq"}
    headers = {"User-Agent": "Mozilla/5.0"}
    rows = []
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            node = r.json()["data"][sym]
            rows = node.get("qfqday") or node.get("day") or []
            if rows:
                break
        except Exception as e:
            print(f"重试 {attempt + 1}/{retries}: {e}")
            time.sleep(1.5)
    if not rows:
        raise RuntimeError(f"无法获取 {code} 的行情数据")
    data = []
    for k in rows:
        data.append({
            "date": k[0], "open": float(k[1]), "close": float(k[2]),
            "high": float(k[3]), "low": float(k[4]), "volume": float(k[5]),
        })
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["amount"] = df["close"] * df["volume"]
    return df
```

- [ ] **Step 3: 创建 predictor.py（预测部分）**

```python
_loaded = {"predictor": None}


def load_model():
    """惰性单例加载模型，避免重复加载"""
    if _loaded["predictor"] is None:
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        model = Kronos.from_pretrained(MODEL_NAME)
        _loaded["predictor"] = KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT)
    return _loaded["predictor"]


def apply_price_limit(pred_df, last_close, limit_rate=LIMIT_RATE):
    cols = ["open", "high", "low", "close"]
    pred = pred_df.copy().reset_index(drop=True)
    floor = last_close * (1 - limit_rate)
    ceil = last_close * (1 + limit_rate)
    pred[cols] = pred[cols].clip(lower=floor, upper=ceil)
    return pred


def run_prediction(code: str, pred_len: int = 60) -> dict:
    """完整预测流程，返回 dict 结果"""
    t0 = time.time()
    df = fetch_daily(code)
    t_fetch = time.time() - t0
    if len(df) < LOOKBACK + 5:
        raise RuntimeError(f"{code} 数据不足: {len(df)} 行，需要至少 {LOOKBACK + 5}")

    predictor = load_model()
    t_load = time.time() - t0

    x_df = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_ts = df.iloc[-LOOKBACK:]["date"].reset_index(drop=True)
    y_ts = pd.bdate_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=pred_len)

    pred = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(y_ts),
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=True,
    )
    t_pred = time.time() - t0

    last_close = float(df["close"].iloc[-1])
    raw_end = float(pred["close"].iloc[-1])
    truncated = abs(raw_end / last_close - 1) > LIMIT_RATE + 1e-6
    pred = apply_price_limit(pred, last_close)
    pred["date"] = pd.Series(y_ts).values
    pred_end = float(pred["close"].iloc[-1])
    pct = (pred_end / last_close - 1) * 100

    return {
        "code": code,
        "df_hist": df,
        "df_pred": pred,
        "last_close": last_close,
        "pred_end": pred_end,
        "pct": pct,
        "truncated": truncated,
        "pred_high": float(pred["high"].max()),
        "pred_low": float(pred["low"].min()),
        "timings": {"fetch": t_fetch, "load": t_load, "predict": t_pred},
    }
```

- [ ] **Step 4: 创建 predictor.py（画图部分）**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_result(code: str, df_hist, df_pred, out_dir: str = "assets") -> str:
    """绘制历史+预测价格对比图，返回 PNG 路径"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pred_{code}.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   facecolor="#0a0e17")
    fig.patch.set_facecolor("#0a0e17")

    colors = {"hist": "#00e5ff", "pred": "#ff6ec7", "grid": "#1e2733"}
    for ax in (ax1, ax2):
        ax.set_facecolor("#0e1420")
        ax.grid(True, color=colors["grid"], alpha=0.6, linewidth=0.5)
        ax.tick_params(colors="#8fa3bf")

    ax1.plot(df_hist["date"], df_hist["close"], color=colors["hist"],
             linewidth=1.3, label="历史收盘")
    ax1.plot(df_pred["date"], df_pred["close"], color=colors["pred"],
             linewidth=1.6, linestyle="--", label="预测收盘")
    ax1.axvline(x=df_hist["date"].iloc[-1], color="#f5c542",
                linestyle=":", alpha=0.8)
    ax1.set_ylabel("收盘价", color="#8fa3bf")
    ax1.legend(facecolor="#0e1420", edgecolor="#1e2733", labelcolor="#c9d6e8")

    ax2.plot(df_hist["date"], df_hist["volume"], color=colors["hist"],
             linewidth=0.8, label="历史成交量")
    ax2.plot(df_pred["date"], df_pred["volume"], color=colors["pred"],
             linewidth=0.8, linestyle="--", label="预测成交量")
    ax2.set_ylabel("成交量", color="#8fa3bf")
    ax2.legend(facecolor="#0e1420", edgecolor="#1e2733", labelcolor="#c9d6e8")

    plt.tight_layout()
    plt.savefig(path, dpi=140, facecolor="#0a0e17")
    plt.close(fig)
    return path
```

- [ ] **Step 5: 自检脚本**

在项目根目录临时运行：
```powershell
uv run python -c "import sys; sys.path.append('.'); from predictor import fetch_daily, run_prediction, plot_result; df=fetch_daily('603516'); print(len(df), df['close'].min()>0, df.columns.tolist())"
```

预期：输出如 `701 True ['date','open','high','low','close','volume','amount']`。负价断言为 True。

> 完整预测自检留到 Task 4（base 模型首次加载慢）。

- [ ] **Step 6: 更新 PROGRESS.md**

标记「逻辑层 ✅ 完成」，记录腾讯接口验证结果。

---

### Task 3: UI 层 app.py（科技感 Gradio 界面）

**Files:**
- Create: `E:\workspace\python\kronos-forecast\app.py`
- Create: `E:\workspace\python\kronos-forecast\assets\style.css`

**Interfaces:**
- Consumes: `predictor.py`（fetch_daily/load_model/run_prediction/plot_result）、`stocks.py`
- Produces: `uv run app.py` 启动可访问的 Web 界面

- [ ] **Step 1: 创建 assets/style.css（深色科技风）**

```css
/* 深色科技风全局样式 */
.gradio-container { background: #0a0e17 !important; }
#app-header { text-align: center; padding: 1.2rem 0; }
#app-title {
  font-size: 2.2rem; font-weight: 800;
  background: linear-gradient(90deg, #00e5ff, #8b5cf6, #ff6ec7);
  -webkit-background-clip: text; background-clip: text;
  color: transparent; letter-spacing: 2px;
}
#app-subtitle { color: #8fa3bf; font-size: 1rem; }
.stock-btn button { height: 56px !important; font-size: 1.1rem !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg, #122036, #0e1420) !important;
  border: 1px solid #00e5ff55 !important; color: #c9d6e8 !important;
  transition: all .25s ease; }
.stock-btn button:hover { border-color: #00e5ff !important;
  box-shadow: 0 0 18px #00e5ff44; color: #00e5ff !important; }
.card { border-radius: 14px !important; border: 1px solid #1e2733 !important;
  background: #0e1420 !important; }
```

- [ ] **Step 2: 创建 app.py 骨架（主题 + 布局）**

```python
import gradio as gr

from predictor import run_prediction, plot_result
from stocks import DEFAULT_STOCKS

theme = gr.themes.Base(
    primary_hue="cyan", secondary_hue="violet", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter")],
).set(
    body_background_fill="#0a0e17",
    block_background_fill="#0e1420",
    block_border_color="#1e2733",
    body_text_color="#c9d6e8",
    button_primary_background_fill="linear-gradient(135deg, #00e5ff, #8b5cf6)",
    button_primary_text_color="#0a0e17",
)


def predict_and_show(code: str, pred_len: int):
    """预测按钮回调：跑预测 + 画图 + 生成摘要文本"""
    try:
        result = run_prediction(code, pred_len)
        chart = plot_result(code, result["df_hist"], result["df_pred"])
        flag = "⚠️ 预测已超出涨跌停，被截断" if result["truncated"] else "正常"
        summary = (
            f"### 预测摘要\n\n"
            f"- 最新收盘: **{result['last_close']:.2f}**\n"
            f"- 预测终点: **{result['pred_end']:.2f}** ({result['pct']:+.2f}%)\n"
            f"- 预测区间: {result['pred_low']:.2f} ~ {result['pred_high']:.2f}\n"
            f"- 限幅状态: {flag}\n"
            f"- 耗时: 拉数{result['timings']['fetch']:.1f}s / 预测{result['timings']['predict']:.1f}s"
        )
        return chart, result["df_pred"], summary, f"✅ {code} 预测完成"
    except Exception as e:
        return None, None, f"❌ 预测失败: {e}", f"错误: {e}"


with gr.Blocks(theme=theme, css_path="assets/style.css", title="Kronos A股预测面板") as demo:
    gr.HTML(
        "<div id='app-header'>"
        "<div id='app-title'>KRONOS · A股智能预测</div>"
        "<div id='app-subtitle'>金融K线基础模型 · 实时行情 · 未来60日走势预测</div>"
        "</div>"
    )
    with gr.Row():
        code_input = gr.Textbox(label="股票代码", placeholder="如 600519", scale=2)
        pred_len_slider = gr.Slider(10, 120, value=60, step=5, label="预测天数", scale=1)
    with gr.Row():
        predict_btn = gr.Button("🚀 开始预测", variant="primary", scale=2)
        quick_btns = [gr.Button(name, elem_classes="stock-btn", scale=1)
                      for name in DEFAULT_STOCKS.keys()]
    status = gr.Markdown("等待预测...")
    with gr.Row():
        chart_out = gr.Image(label="预测走势", height=560, show_download_button=True)
    with gr.Row():
        table_out = gr.Dataframe(label="预测数据", interactive=False, scale=1)
        summary_out = gr.Markdown(label="摘要", scale=1)

    def on_quick(name):
        return DEFAULT_STOCKS[name]

    for btn in quick_btns:
        btn.click(on_quick, inputs=btn, outputs=code_input)

    predict_btn.click(predict_and_show, inputs=[code_input, pred_len_slider],
                      outputs=[chart_out, table_out, summary_out, status])


if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 3: 启动验证**

```powershell
cd E:\workspace\python\kronos-forecast
uv run app.py
```

预期：控制台显示 Gradio 本地地址，浏览器打开看到深色科技风页面。仅验证页面渲染，不点预测（base 模型首次加载需下载）。

- [ ] **Step 4: 更新 PROGRESS.md**

标记「UI 层 ✅ 完成」。

---

### Task 4: 端到端验证

**Files:**
- Modify: `E:\workspace\python\kronos-forecast\docs\PROGRESS.md`

**Interfaces:**
- Consumes: app.py + predictor.py 全部完成
- Produces: 验收证据（预测结果、耗时、图表路径）

- [ ] **Step 1: 预下载 Kronos-base 模型**

```powershell
cd E:\workspace\python\kronos-forecast
$env:HF_ENDPOINT="https://hf-mirror.com"
uv run python -c "from huggingface_hub import snapshot_download; p=snapshot_download('NeoQuasar/Kronos-base'); print(p)"
```

预期：下载成功（约 400MB），打印缓存路径。

- [ ] **Step 2: 命令行端到端测淳中科技**

```powershell
uv run python -c "import sys; sys.path.append('.'); from predictor import run_prediction, plot_result; r=run_prediction('603516', 60); print('close', r['last_close'], '->', r['pred_end'], f\"{r['pct']:+.2f}%\", 'truncated', r['truncated']); print(plot_result('603516', r['df_hist'], r['df_pred']))"
```

预期：打印预测价格与百分比、PNG 路径，无报错。

- [ ] **Step 3: 测剑桥科技**

同上，代码换 `603083`。预期打印预测结果。

- [ ] **Step 4: 浏览器手动验收**

`uv run app.py` 启动，浏览器依次：
1. 点「淳中科技」按钮 → 代码框出现 603516 → 点预测 → 等进度 → 看到图表/表格/摘要
2. 输入 600519 自定义 → 预测成功
3. 拖动预测天数滑条到 30 → 预测成功

- [ ] **Step 5: 更新 PROGRESS.md**

标记全部里程碑 ✅，记录实测数据、耗时、遗留问题。

---

## Self-Review 记录

- Spec 覆盖：FR-1(股票池) → Task1/3 ✓；FR-2(数据) → Task2 ✓；FR-3(预测) → Task2 ✓；FR-4(界面) → Task3 ✓；FR-5(文档) → 各任务 Step 收尾 ✓；NFR-1(性能) → Task4 实测 ✓。
- 无占位符：所有代码步骤含完整实现。
- 类型一致性：`run_prediction` 返回 dict 键名在 Task2 定义、Task3/4 引用一致（df_hist/df_pred/last_close/pred_end/pct/truncated/pred_high/pred_low/timings）。
- 新增需求：project 无 git，不创建 commit 步骤。
