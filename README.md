# Kronos A 股预测面板

基于开源金融 K 线基础模型 **Kronos**（AAAI 2026）的本地 A 股预测 Web 面板。

## 快速开始

```powershell
cd E:\workspace\python\kronos-forecast
uv venv --python 3.12
uv pip install -e .
uv run app.py
```

浏览器打开 Gradio 本地地址，选择股票预测即可。

## 功能

- 内置淳中科技（603516）、剑桥科技（603083）一键预测
- 支持任意 A 股代码（自动识别沪深前缀）
- Kronos-base 模型，CPU 推理，预测未来 10-120 个交易日
- 腾讯后复权行情，涨跌停限幅

## 模型下载

首次运行自动从 HuggingFace 拉取 `NeoQuasar/Kronos-base` 与 `NeoQuasar/Kronos-Tokenizer-base`，已配置 hf-mirror 镜像。若需手动下载：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET="1"
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('NeoQuasar/Kronos-base'))"
```

> 注意：需禁用 xet（`HF_HUB_DISABLE_XET=1`），否则 hf-mirror 会返回 401。
