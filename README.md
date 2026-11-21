# Fintech Volatility

统一入口位于 src/run_experiment_pipeline.py。研究使用本地历史行情、新闻和评分。研究设计、各轮迭代和最终验收见 [实验方案与结果汇总](实验方案与结果汇总.md)。

~~~powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python src/run_experiment_pipeline.py --phase a
.\.venv\Scripts\python src/run_experiment_pipeline.py --phase b --offline
~~~

Phase A 重跑三股模型对比、公告回放与特征审计。Phase B 的离线入口使用本地九股评分重跑联合回测、常规日分治和财报必审，核验历史输入并对照已发布结果。两种入口都无需 API 密钥。九股覆盖率与费用预检使用 --phase b --dry-run，新增评分须给出 --approved-usd。费用授权按整轮计入，续跑复用已完成的评分。

~~~powershell
.\.venv\Scripts\python src/run_experiment_pipeline.py clock-diagnostic
.\.venv\Scripts\python src/run_experiment_pipeline.py collect --help
~~~

| 模块 | 职责 |
| --- | --- |
| config、artifacts | 研究配置、路径与模型文件完整性 |
| market_data、features | 行情窗口、波动标签与时点特征 |
| models、evaluation | 候选模型、时序切分、评分指标与统计区间 |
| backtest、research、portfolio | 联合回测、三股比较和九股研究 |
| clock_diagnostic | 同一股票、同一窗口的时点口径诊断 |
| collection、news_sources | 历史行情、机构披露与媒体来源采集 |
| semantic、scoring | DeepSeek 闭卷评分、缓存、费用保护与二阶段细化 |

数据和结果沿用既有路径。运行日志在 .log/，终版结果在 results/v5/。Windows 盘前采集和次日核验任务均已停用。各轮原始方案、报告和旧版代码见 Git 提交 483b1b4。
