# Kronos A 股预测面板 设计文档

日期：2026-08-04
状态：已批准

## 1. 目标

在 `E:\workspace\python\kronos-forecast/` 新建一个本地金融预测 Web 应用：

- 使用 **uv** 管理 Python 依赖
- 使用 **Gradio** 作为 Web 界面
- 使用 **Kronos-base** 模型（102.3M 参数）做 A 股日线预测
- 内置**淳中科技（603516）**与**剑桥科技（603083）**两个股票，可在界面上添加更多
- 界面深色科技风，美观大方
- 需求与进度写入独立文档

## 2. 技术栈

| 项 | 选型 | 理由 |
|----|------|------|
| 包管理 | uv 0.11+ | 用户指定 |
| 界面 | gradio | 用户指定，Python 生态快速出 Web UI |
| 模型 | NeoQuasar/Kronos-base + Kronos-Tokenizer-base | 用户指定，本地 CPU 推理 |
| 数据源 | 腾讯行情接口 `web.ifzq.gtimg.cn`（后复权） | 已验证稳定可用，无需 token |
| 图表 | matplotlib（Agg 后端） | Kronos 仓库自带绘图栈，无额外依赖 |
| Python | 3.12 | gradio/torch 生态最稳；uv 自动装 |
| 框架 | 纯 Python 脚本 + `sys.path` 引入本地 `model/` | Kronos 模型非 pip 包，按仓库原样使用 |

## 3. 目录结构

```
E:\workspace\python\kronos-forecast\
├── pyproject.toml          # uv 项目定义
├── README.md               # 快速开始
├── app.py                  # Gradio 入口，全部界面代码
├── predictor.py            # 数据拉取 + 预测 + 画图（纯逻辑，无 UI）
├── stocks.py               # 内置股票池字典，可扩展
├── model/                  # 从 Kronos 仓库复制（kronos.py / module.py / __init__.py）
├── assets/                 # 界面装饰（渐变背景、logo，可选）
└── docs/
    ├── REQUIREMENTS.md     # 需求文档
    └── PROGRESS.md         # 进度文档
```

## 4. 数据流

```
用户选择股票 → predictor.fetch_daily(code) → 腾讯接口后复权日线(≤2000行)
        → 截取最近400交易日 → KronosTokenizer 量化 → Kronos-base 自回归预测60步
        → 涨跌停限幅(±10%) → matplotlib 出图 → Gradio 展示图表 + 预测表格 + 摘要指标
```

## 5. 界面设计（深色科技风）

- **主题**：Gradio 深色主题（`gr.themes.Base` + 自定义 CSS）
- **配色**：深蓝黑背景（#0a0e17），霓虹青（#00e5ff）/ 霓虹紫（#8b5cf6）渐变点缀，高亮卡片
- **布局**：
  - 顶部：标题 + 副标题（渐变文字）
  - 股票选择区：两个大按钮快速选淳中/剑桥 + 下拉/文本框自定义代码 + "添加并预测"按钮
  - 预测参数区：预测天数滑动条（默认60，范围10-120）
  - 主展示区：预测 K 线/价格对比图（大）+ 预测数据表格 + 摘要卡片（当前价/预测价/涨跌幅）
  - 底部：状态栏（模型加载状态、耗时）
- **交互**：预测按钮点击后显示进度（模型加载 + 逐步预测），完成后自动填充图表/表格

## 6. 错误处理

- 数据源不可用：重试 3 次，超时 15s；最终失败给出中文错误提示，不崩 UI
- 数据不足 400 行：明确报错
- 模型首次加载慢（下载 400MB + 加载）：启动时预热加载，界面显示 loading 状态
- 涨跌停限幅：预测值超出 ±10% 时截断并在结果标注

## 7. 测试策略

- 手动验证：运行 `uv run app.py`，浏览器打开，分别测淳中/剑桥/自定义股票
- 逻辑层单测：`predictor.fetch_daily` 返回字段完整、无负价、后复权；`apply_price_limit` 边界
- 不引入 pytest，用简单的 `if __name__ == "__main__"` 自检脚本（保持轻量）

## 8. 范围边界（YAGNI）

- 不做多股票同时预测队列
- 不做历史回测
- 不做持仓/盈亏管理
- 不做模型微调
- 不做部署（仅本地运行）
