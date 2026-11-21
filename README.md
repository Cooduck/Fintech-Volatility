# Fintech Volatility

当前研究入口为 V5，使用已归档的历史行情、新闻和评分。三股与九股回测、验收结果见 [V5 终版报告](V5终版实验报告.md)。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python src/run_experiment_pipeline.py --study v5 --phase a *> .log/v5_run.log
.\.venv\Scripts\python src/run_experiment_pipeline.py --study v5 --phase b --offline
```

Phase A 从本地输入重跑三股模型比较。Phase B 的离线入口核对历史输入摘要，使用评分归档重跑九股联合回测和常规日分治回测，并与已发布结果逐表对照。核验记录写入 `results/v5/phase_b/offline_reproduction.json`。这两个入口都无需 API 密钥。

代码在 `src/`，运行日志在 `.log/`，终版产物在 `results/v5/`。本项目的 Windows 盘前采集与次日核验任务已停用，终版只使用历史数据。

历史入口保留 `--study original --mode demo`、`--study improved --mode demo`、`--study v3` 和 `--study v4 --phase a`。九股付费评分的原入口保留 `--approved-usd` 费用保护和断点续跑能力。
