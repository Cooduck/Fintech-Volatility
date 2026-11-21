# 基于 V4 Phase A 实证诊断的完整优化方案（V4.5 / V5 落地方案）

## 一、V4 Phase A 核心诊断：已经解决什么、还卡在哪里

对 V4 Phase A 全部产物（`V4实验报告.md`、`results/v4/paired_auc.csv`、`results/v4/earnings_split.csv`、`results/v4/pooled_granger.csv`、`results/v4/event_replay.csv`、`results/v4/phase_b/estimate.json`）及 `src/v4_models.py` 源码核查后，项目当前处于**“方法学骨架已完备，剩余 4 个精确可修的工程与数据瓶颈”**阶段：

- **已取得的硬核成果**：
  1. **Granger 先导检验正式通过 5% BH 校正**：四项预注册主检验中，纯盘中早盘 B2 的 `ClusterImpactSum`（Lag=1–2 联合检验）取得 **`F = 5.955, p = 0.0028, q_primary = 0.0112 < 0.05`**；
  2. **财报 + VIX 基线成功挤出水分，验证了非财报常规日的真实语义价值**：加入 `^VIX` 与 SEC 8-K Item 2.02 财报窗口后，基准 A 从 `0.635` 跃升至 `0.696`；在有界首发时间戳下，三股联合 B1 完整模型 D 达到 **`0.733` AUC（PR-AUC `0.323`）**，且在**剔除全部财报日后的 711 个非财报常规日**上，仍取得统计显著的 **`D - A = +0.042`（95% 区间 `[+0.002, +0.090]`）**；合格股（NVDA + TSLA）在纯盘中 B2 上也取得显著的 **`+0.028`（`[+0.0001, +0.062]`）**。

- **暴露出的 4 个精确瓶颈（本次优化的全部靶点）**：

| 瓶颈编号 | 实查量化证据 | 代码/数据层根因 | 优先级 |
| :--- | :--- | :--- | :--- |
| **瓶颈 1：11 月 12 日 B1 公告后原始分反降** | `08:47 -> 08:48` 公告到达时 `cluster_impact` 从 `1.337` 跳升至 `1.722`，但 B1 原始分反降（`0.709 -> 0.694`，线性 `0.848 -> 0.837`、树 `0.384 -> 0.359`） | `src/v4_models.py` 中 `TrainQuantileWinsorizer` 用硬截断 `np.clip` 把 `>99%` 分位的 `ClusterImpactSum` 拍平；同时树模型仅对 5 列加了 `+1` 约束，未对 `MinSentiment` 加 `-1` 约束，且未约束 `NegRatio`/`UniqueEventCount` | **P0（30 分钟修复，$0）** |
| **瓶颈 2：历史时间戳滞后抹平 `+0.042` 增益** | 保守口径 `pooled_3 B1` 非财报日 `D-A = +0.003`（不显著），有界首发口径跃升至 `+0.042`（显著） | 28.8% 的历史文章因后期修订被保守口径推迟，无法在历史快照中彻底消除；唯有通过真实前瞻采集锁定 `first_seen_utc` 才能同时满足“零泄露”与“实时性” | **P0（立刻挂载定时任务）** |
| **瓶颈 3：Phase B 新增 6 股覆盖率严重不足** | `estimate.json` 显示 `GOOGL 28.8%`、`AVGO 32.4%`、`AMD 36.9%`、`META 44.6%`、`AMZN 46.1%`、`AAPL 49.3%`、`MSFT 62.4%`，均低于 70% 准入线 | 现有媒体源（CNBC Archive + 24/7 Wall St. 等）对 NVDA/TSLA 覆盖高，但对其他 7 股检索漏采严重；若此时直接付费跑 Phase B，将浪费预算且无法扩大合格面板 | **P1（跑 Phase B 前必做）** |
| **瓶颈 4：财报日与常规日先验异常率相差近 9 倍** | `earnings_split.csv` 中 3 股财报窗口仅 12 条（B1 异常率 **75.0%**），常规日 711 条（异常率 **8.9%**） | 两种分布截然不同的样本混在同一个线性截距和树分裂里训练，会扭曲常规日的概率尺度与阈值稳定性 | **P2（9 股达标后落地）** |

---

## 二、四大核心优化模块（精确到代码逻辑与验收标准）

### 模块 1（P0 · 零费用立刻修复）：软饱和截尾 + 全风险特征单调约束，根治 B1 公告反降

#### 1. 将 `TrainQuantileWinsorizer` 升级为“软饱和截尾（Soft Tail Compression）”
硬截断 `np.clip(x, lo, hi)` 会导致超过训练集 99% 分位的极端黑天鹅（如 `cluster_impact` 从 `1.337` 飙升至 `1.722`）全部被拍平为同一个常数 `hi`，使得线性模型对“极端之上更极端”的突发公告失去边际灵敏度。
- **修改 `src/v4_models.py` 中的 `TrainQuantileWinsorizer.transform`**：
  ```python
  def transform(self, X):
      check_is_fitted(self, "lo_")
      values = np.asarray(X, dtype=float).copy()
      sub = values[:, self.positions_]
      # 下尾做硬截断，上尾超过 99% 分位后改用对数软压缩，既防外推爆炸又保留严格单调递增
      sub = np.maximum(sub, self.lo_)
      over = sub > self.hi_
      sub[over] = self.hi_[np.where(over)[1]] + np.log1p(sub[over] - self.hi_[np.where(over)[1]])
      values[:, self.positions_] = sub
      return values
  ```

#### 2. 补齐风险特征的双向单调约束与负向情绪解耦
- **新增单向负情绪强度特征（写入 `src/v4_features.py`）**：
  - `NegSentIntensity = np.maximum(0.0, -SentMean)`（消除正面情绪与负面情绪在单调树中难以统一约束的歧义）；
  - `NegMinIntensity = np.maximum(0.0, -MinSentiment)`（将 `MinSentiment` 转为非负单调递增风险指标）。
- **更新 `src/v4_models.py` 的单调约束映射表（同时覆盖 B1 分类树与 B2 回归树）**：
  ```python
  MONOTONE_MAP = {
      "ClusterImpactSum": 1,
      "NewsShock": 1,
      "HasLevel3Event": 1,
      "UniqueEventCount": 1,
      "NegRatio": 1,
      "NegSentIntensity": 1,
      "NegMinIntensity": 1,
      "Interaction": 1,
      "is_earnings_window": 1,
      "vix_z20": 1,
  }
  ```
- **特征符号审计（写入 `src/v4_models.py` 诊断输出）**：
  - 在 `fit_route` 后记录 L2 各语义特征的标准化权重至 `results/v4/linear_weights.csv`，核查是否存在因多重共线性导致的反向权重（例如 `SentStd` 与 `NoveltyRatio` 在极端恐慌日反而下降拖累总分）；若在训练期内层验证中 `NoveltyRatio` 权重为负且无 AUC 增益，将其移出 B1 线性主特征集，仅保留在独立诊断表。
- **验收标准**：在 11 月 12 日回放中，`08:47 -> 08:48 -> 08:49` 的 **B1 原始分与 B2 原始分双双严格正向跃升**（不再出现 `0.709 -> 0.694` 倒挂）。

---

### 模块 2（P0 · 立刻部署）：启动 `v4_forward_watcher.py` 每日定时任务，开启真前瞻验证（Phase C）

V4 Phase A 已经用数据证明：**历史回测中保守时间戳（`0.698`）与有界首发时间戳（`0.733`）之间存在高达 `+0.035` AUC 的时间戳损耗**。与其在历史快照上反复纠结，不如立刻把已写好的前瞻监测器挂上定时任务，积累零泄露、零修订污染的真实前瞻样本。

#### 1. Windows 任务计划程序自动化部署（3 股先跑，日耗 `< $0.001`）
在项目根目录以 PowerShell 注册两个每日定时任务（自动避开周末，仅在美股交易日执行；若本机为北京时间，美东冬令时 `09:22 EST` = 北京时间 `22:22`）：
```powershell
# 任务 1：每日美东 09:22（北京时间 22:22）运行盘前首发采集与实时预警落盘
schtasks /create /tn "FintechVol_Forward_Predict" /tr "D:\path\to\project\.venv\Scripts\python.exe D:\path\to\project\src\v4_forward_watcher.py --phase a --run-once" /sc weekly /d MON,TUE,WED,THU,FRI /st 22:22 /f

# 任务 2：次日美东 16:30（北京时间次日 05:30）回填真实 B1/B2 波动率标签并更新前瞻审计表
schtasks /create /tn "FintechVol_Forward_Verify" /tr "D:\path\to\project\.venv\Scripts\python.exe D:\path\to\project\src\v4_forward_watcher.py --phase a --verify" /sc weekly /d TUE,WED,THU,FRI,SAT /st 05:30 /f
```

#### 2. 真前瞻运行期防篡改规范
- 每次 `--run-once` 仅写入 `09:30 EST` 之前的首发新闻快照（锁死 `first_seen_utc`、`original_title`、`sha256`）与模型原始预测分；
- 连续积累 **15–20 个交易日**（3 股共 45–60 个全新早盘，若 Phase B 完成后切至 9 股则可达 135–180 个全新早盘），生成独立章节《真前瞻实时验证报告》，彻底终结关于“回顾性留出集”与“历史修订标题”的所有质疑。

---

### 模块 3（P1 · 启动 Phase B 的硬前置）：补齐 7 股财经新闻覆盖率至 $\ge 70\%$，再触发 DeepSeek 评分

当前 `results/v4/phase_b/estimate.json` 显示：`AAPL (49.3%)`、`AMZN (46.1%)`、`META (44.6%)`、`AMD (36.9%)`、`AVGO (32.4%)`、`GOOGL (28.8%)`、`MSFT (62.4%)` 全部低于 70% 覆盖门槛。**在补齐覆盖率之前，严禁对这批残缺语料发起付费 DeepSeek 评分。**

#### 1. `src/v4_collect.py` 增补高覆盖多源财经采集通道
为所有 9 只标的统一接入以下 3 个互补源（均带精确 UTC 时间戳且按 URL + 标题标准化去重）：
1. **GDELT 2.0 Doc API（免费、覆盖 2024–2026 全区间、无配额上限）**：
   - 按标的构造精确布尔查询（如 `("Apple" OR "AAPL") (domainis:reuters.com OR domainis:cnbc.com OR domainis:marketwatch.com OR domainis:wsj.com OR domainis:barrons.com OR domainis:finance.yahoo.com OR domainis:bloomberg.com)`）；
   - 将 GDELT 返回的 `seendate`（UTC 精确到秒）作为首发可用时间。
2. **Finnhub Company News API（免费 Key 放 `.env` 中 `FINNHUB_API_KEY`，支持按 Ticker 精确拉取带 Unix 时间戳的财经新闻与导语摘要）**：
   - 直接按 `symbol`（如 `GOOGL`, `AVGO`, `AMD`, `MSFT`）分段拉取近 1–2 年公司新闻，天然附带 `summary` 字段，可直接供给 Stage 2 高影响导语精评，省去二次网页抓取失败率！
3. **SEC EDGAR 8-K / 10-Q / 10-K 全量实质披露补齐**：
   - 确保 9 只标的按 CIK 完整拉取所有 Item 1.01（重大协议）、Item 2.02（财报）、Item 5.02（高管变动）、Item 7.01/8.01（重大披露）并提取首段正文。

#### 2. Phase B 启动前的硬性门禁检查（Gate Check）
运行 `.\.venv\Scripts\python src/run_experiment_pipeline.py --study v4 --phase b --dry-run` 后，必须满足以下两个条件才允许去掉 `--dry-run` 发起真实 DeepSeek 评分：
- **覆盖率门禁**：9 只标的至少 **7 只以上**半日覆盖率 $\ge 70\%$（且 `NVDA, MSFT, TSLA, AAPL, AMZN, META, GOOGL` 全部达标）；
- **费用门禁**：总预估费用（含 Stage 1 新增标题 + Stage 2 Tier 1 高影响导语）严格控制在项目剩余预算（`$2.00 - $0.4843 = $1.5157`）以内，优先在**空闲时段**（北京时间非工作日或工作日 12:00–14:00、18:00–次日 09:00）执行以享半价。

---

### 模块 4（P2 · 9 股达标后落地）：财报窗口与非财报常规日“分治建模”

#### 1. 问题根源
`earnings_split.csv` 显示：财报早盘 B1 异常率高达 **75.0%**（9/12），非财报常规早盘仅 **8.9%**（63/711），先验概率相差近 **9 倍**。混在同一个模型中会造成两个扭曲：一是线性模型的截距和阈值被财报日拉偏；二是树模型把有限的 5 个叶节点优先浪费在切分 `is_earnings_window == 1` 上，挤占了常规日黑天鹅的分裂容量。

#### 2. 分治架构设计（仅在 9 股面板上启用，保证样本量充足）
- **常规日专用风控模型（Regular-Day Model）**：
  - 仅在 `is_earnings_window == 0` 的样本（9 股约 **~2,100 个样本外早盘、~180+ 个异常**）上训练与评估，彻底移除 `is_earnings_window` 特征，让全部树分裂容量与线性自由度专注学习**突发公司治理、监管制裁、供应链冲击、宏观政策与 VIX 异动**。
- **财报窗口专用语义分支（Earnings-Window Evaluator）**：
  - 对 `is_earnings_window == 1` 的样本（9 股约 **36 个样本外财报早盘**），直接标记为**已知高危事件必审窗口**，重点用 Stage 2 的“财报标题 + 导语精评”输出财报指引方向与波动风险等级，不再与常规日混算同一套 85%/95% 分位阈值。

---

## 三、分步执行清单与最终验收标准

| 执行步骤 | 操作命令与动作 | 预计耗时与费用 | 量化通过标准 |
| :--- | :--- | :--- | :--- |
| **Step 1：修 B1 反降 Bug + 重跑 Phase A** | 按模块 1 修改 `src/v4_features.py` 与 `src/v4_models.py`，运行 `--study v4 --phase a` | 30 分钟 / **$0** | 11 月 12 日 `08:47 -> 08:48 -> 08:49` B1 与 B2 **原始分双双严格递增**；B2 预注册 Granger 保持 `q < 0.05` |
| **Step 2：立刻挂载 Phase C 真前瞻定时任务** | 按模块 2 配置每日 `09:22 EST` 采集预警与次日 `16:30 EST` 自动核验 | 15 分钟 / **日耗 < $0.001** | 连续运行 15–20 个交易日，产出零修改污染的 `results/v4_forward/summary.csv` |
| **Step 3：补齐 7 股语料覆盖率（GDELT / Finnhub）** | 按模块 3 升级 `src/v4_collect.py`，运行 `--study v4 --phase b --dry-run` 审计覆盖率 | 2–3 小时 / **$0** | 9 股中至少 7 股（含 MSFT）半日覆盖率 $\ge 70\%$，剩余预算充足 |
| **Step 4：空闲时段执行 Phase B 全量 9 股 + 分治对照** | 去掉 `--dry-run` 执行 9 股评分与回测，并跑通模块 4 常规日独立建模对照 | 1 小时 / **$0.50–$1.00** | 1. 合格面板样本外早盘异常数 $\ge 180$；<br>2. **非财报常规日**上 `D_v4 - A_v4` 在保守或真前瞻口径下实现统计显著正向增益；<br>3. 预注册 Granger 检验在 9 股面板上持续显著（`q < 0.05`） |
