import os
import gradio as gr

from predictor import run_prediction, plot_result
from stocks import DEFAULT_STOCKS

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter")],
    spacing_size="md",
    radius_size="lg",
).set(
    body_background_fill="#f8fafc",
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    body_text_color="#1e293b",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #0ea5e9)",
    button_primary_text_color="#ffffff",
)

PRED_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
COL_LABELS = {
    "date": "日期", "open": "开盘价", "high": "最高价",
    "low": "最低价", "close": "收盘价", "volume": "成交量", "amount": "成交额(万)",
}


def predict_and_show(code: str, pred_len: int):
    """预测按钮回调：跑预测 + 画图 + 生成摘要文本"""
    try:
        result = run_prediction(code.strip(), pred_len)
        chart = plot_result(result["code"], result["df_hist"], result["df_pred"])
        flag = "⚠️ 预测已超出涨跌停，被截断" if result["truncated"] else "正常"
        summary = (
            f"### 预测摘要\n\n"
            f"- 最新收盘: **{result['last_close']:.2f}**\n"
            f"- 预测终点: **{result['pred_end']:.2f}** ({result['pct']:+.2f}%)\n"
            f"- 预测区间: {result['pred_low']:.2f} ~ {result['pred_high']:.2f}\n"
            f"- 限幅状态: {flag}\n"
            f"- 推理设备: **{result['device']}**\n"
            f"- 耗时: 拉数{result['timings']['fetch']:.1f}s / "
            f"模型加载{result['timings']['load']:.1f}s / "
            f"预测{result['timings']['predict']:.1f}s"
        )
        table = result["df_pred"][PRED_COLS].copy()
        table["date"] = table["date"].dt.strftime("%Y-%m-%d")
        # 价格/金额保留2位小数，减少列宽
        for col in ["open", "high", "low", "close"]:
            table[col] = table[col].round(2)
        table["volume"] = table["volume"].astype(int)
        table["amount"] = (table["amount"] / 1e4).round(2)  # 万元
        table = table.rename(columns=COL_LABELS)
        return chart, table, summary, "✅ 预测完成"
    except Exception as e:
        return None, None, f"❌ 预测失败: {e}", f"错误: {e}"


# 下拉选项: "名字 (代码)" 格式, value 存代码
STOCK_CHOICES = {f"{name} ({code})": code for name, code in DEFAULT_STOCKS.items()}

with gr.Blocks() as demo:
    gr.HTML(
        "<div id='app-header'>"
        "<div id='app-title'>KRONOS · A股智能预测</div>"
        "<div id='app-subtitle'>金融K线基础模型 · 实时行情 · 未来走势预测</div>"
        "</div>"
    )
    with gr.Row():
        code_input = gr.Dropdown(
            label="选择股票",
            choices=list(STOCK_CHOICES.keys()),
            value=list(STOCK_CHOICES.keys())[0] if STOCK_CHOICES else None,
            allow_custom_value=True,
            scale=3,
        )
        pred_len_slider = gr.Slider(10, 120, value=60, step=5, label="预测天数",
                                    scale=2)
        predict_btn = gr.Button("🚀 预测", variant="primary",
                                elem_id="predict-btn", scale=1)
    status = gr.Markdown("等待预测...")
    chart_out = gr.Image(label="预测走势", height=560,
                         buttons=["download", "fullscreen"])
    with gr.Row():
        table_out = gr.Dataframe(label="预测数据", interactive=False,
                                 wrap=True, scale=3)
        summary_out = gr.Markdown(label="摘要", elem_classes="summary-card",
                                  scale=1)

    def resolve_code(label: str) -> str:
        """下拉选中 '名字 (code)' 时提取 code; 自定义输入直接返回"""
        return STOCK_CHOICES.get(label.strip(), label.strip())

    predict_btn.click(
        lambda code, pl: predict_and_show(resolve_code(code), pl),
        inputs=[code_input, pred_len_slider],
        outputs=[chart_out, table_out, summary_out, status],
    )


if __name__ == "__main__":
    demo.title = "Kronos A股预测面板"
    demo.launch(
        theme=theme,
        css="assets/style.css",
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
