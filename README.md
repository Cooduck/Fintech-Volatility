# Fintech Volatility

项目通过 `src/run_experiment_pipeline.py` 运行。先在项目根目录创建 Python 虚拟环境并安装依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

复现 V3 离线实验并保存运行日志。

```powershell
.\.venv\Scripts\python src/run_experiment_pipeline.py --study v3 *> .log/v3_run.log
```

复现前两轮实验可将 `--study v3` 换成 `--study improved --mode demo` 或 `--study original --mode demo`。输入、缓存、结果和本地 `.env` 从项目根目录读取。运行日志保存在 `.log/`。
