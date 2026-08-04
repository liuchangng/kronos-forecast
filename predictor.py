import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import sys
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

LOOKBACK = 400
MAX_CONTEXT = 512
LIMIT_RATE = 0.10
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_NAME = "NeoQuasar/Kronos-base"

_loaded = {"predictor": None, "device": "cpu"}


def market_prefix(code: str) -> str:
    """根据代码判断沪深前缀"""
    return "sh" if code.startswith(("6", "9")) else "sz"


def fetch_daily(code: str, retries: int = 3, timeout: int = 15) -> pd.DataFrame:
    """腾讯后复权日线接口，返回 date/open/high/low/close/volume/amount"""
    import requests
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


def load_model():
    """惰性单例加载模型，避免重复加载"""
    if _loaded["predictor"] is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        model = Kronos.from_pretrained(MODEL_NAME)
        _loaded["predictor"] = KronosPredictor(
            model, tokenizer, device=device, max_context=MAX_CONTEXT)
        _loaded["device"] = device
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
        raise RuntimeError(
            f"{code} 数据不足: {len(df)} 行，需要至少 {LOOKBACK + 5}")

    predictor = load_model()
    t_load = time.time() - t0

    x_df = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_ts = df.iloc[-LOOKBACK:]["date"].reset_index(drop=True)
    y_ts = pd.bdate_range(
        start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=pred_len)

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
        "device": _loaded["device"],
        "timings": {"fetch": t_fetch, "load": t_load, "predict": t_pred},
    }


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK_FONTS = ["Microsoft YaHei", "SimHei", "SimSun", "PingFang SC"]
for _name in _CJK_FONTS:
    _paths = font_manager.findfont(font_manager.FontProperties(family=_name),
                                   fallback_to_default=False)
    if _paths and os.path.basename(_paths).lower() != "dejavusans.ttf":
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break


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