# V3 实验深度评估与 V4 进阶优化详细实施方案

## 一、V3 实验总评与 V4 升级边界

V3 在 **$0 新增 API 费用**下完成了关键的统计突破，核心实证结果已达到高质量课程研究与定量实证报告的标准：
- **三标的联合早盘（B1）首次实现统计显著的语义增益**：保守时间戳下，V3 完整集成 AUC 从量价基准 `0.635` 升至 **`0.688`**（配对增量 `+0.052`，95% 区间 `[+0.003, +0.100]`），PR-AUC 从 `0.165` 跳升至 **`0.263`**（+59%），Lift@Top5% 达 **`3.77x`**；有界时间戳下，单调 LightGBM 突破 **`0.702`**（`+0.085`，`[+0.042, +0.131]`），完整集成达 **`0.699`**（PR-AUC `0.270`）。
- **纯盘中早盘（B2，完全剥离隔夜跳空）同样检出显著增益**：有界口径 HuberLightGBM 达到 **`0.653`**（相对 A `+0.057`，`[+0.010, +0.105]`）；保守口径 Ridge 达到 **`0.642`**（PR-AUC `0.213`，Lift@Top5% `3.36x`），证明大模型语义并非仅解释开盘前已发生的跳空。
- **Model D 相对 Model C 的倒挂彻底消除**：在 `conservative | filtered` 联合早盘中，L2 与集成均实现 D 显著优于 C（`+0.010 ~ +0.013`，95% 区间严格大于零）；11 月 12 日 08:48 CEO 公告到达后，纯盘中 B2 原始集成分清晰地从 `0.369` 跃升至 `0.473`（08:49 达 `0.489`）。
- **Granger 先导检验逼近显著线**：联合纯盘中 B2 的 `SubstantiveImpactSum` 在滞后 1、2、3 阶均取得原始 `p < 0.01`（最低 `p = 0.0028`），仅因 36 项多重检验 BH 校正停在 `q = 0.0745`。

### 严格方法论边界：V4 做什么、绝不做什么
- **严禁事项（防止回顾性过拟合）**：项目已围绕同一批 469 个 NVDA/MSFT/TSLA 交易日迭代了 V1 $\to$ V2 $\to$ V3 三轮。V4 **不再**在现有 3 只股票的冻结特征上微调集成比例（如 0.7/0.3）、树深度或时间衰减系数 $\lambda$，避免把测试集演变为隐性验证集。
- **V4 核心发力点**：全部聚焦于**结构性信息增量**（财报日历、VIX、高影响导语与事件去重）、**横截面统计效能升级**（3 股扩充至 8–9 股）、**模型数值稳定性修复**（Winsorize 防饱和 + B1/B2 分目标路由）以及**真正零触碰的前瞻样本外验证（2026-11-20 起）**。

---

## 二、V4 总体代码架构与目录映射

延续你本地项目清晰的版本隔离风格（`run_experiment_pipeline.py --study original | improved | v3`），V4 通过独立模块接入，**零破坏已有 V1–V3 复现链路**：

| 模块文件 | 职责与核心新增能力 | 关键输入 / 输出 |
| :--- | :--- | :--- |
| `v4_collect.py` | 1. 扩展采集 8–9 只科技权重股行情（1d / 1h）与财经新闻；<br>2. 拉取 `^VIX` 日线与各标的财报日历（Yahoo / SEC 8-K Item 2.02）；<br>3. 对 Tier 1 高影响新闻抓取首段导语（$\le 150$ 词） | 输出冻结快照至 `data/v4/` |
| `v4_features.py` | 1. 在 V3 48 列特征基础上并入 VIX 与财报日历（严守 `known_at <= cut`）；<br>2. 实现同窗事件聚类（TF-IDF 余弦 $\theta = 0.65$）与跨窗新颖度特征；<br>3. 合并二阶段导语精评结果 | 输出 `results/v4/*_features.parquet` |
| `v4_models.py` | 1. 插入训练期内生 `TrainQuantileWinsorizer(0.01, 0.99)` 修复 L2 外推饱和；<br>2. 按目标路由主模型（B1 分类集成 / B2 连续回归集成）；<br>3. 增加“财报窗口 vs. 非财报常规日”分层评估与精简版 Granger 检验 | 输出 `results/v4/ablation.csv`、`earnings_split.csv`、`paired_auc.csv`、`pooled_granger.csv` |
| `v4_pipeline.py` | 编排 `--study v4` 全流程（支持 `--phase a` 零费用快改、`--phase b` 多股全量、`--dry-run` 估价） | 汇总看板 `results/v4/dashboard.png` 与 `audit.json` |
| `v4_forward_watcher.py` | 2026-11-20 起每日 `09:25 EST` 准实时抓取首发新闻、记录精确 `first_seen_utc`、输出前瞻预警并次日自动核验真实 B1/B2 标签 | 追加写入 `results/v4_forward/daily_log.jsonl` |

---

## 三、四大核心优化方向详细设计（可直接落地至代码）

### 方向 1（P0 · 信息维度突破）：引入“财报日历 + VIX 期权恐慌基准”与“非财报日”纯黑天鹅切片

#### 1. 为什么必须加？
美股科技股早盘异常波动中，约 25%–40% 集中在季度财报发布日或全市场宏观恐慌日。缺少这两个事前已知的公开基准变量，会带来两个问题：一是纯量价基线 A 偏弱，拉低整体预警天花板；二是评审会质疑“DeepSeek 只是通过标题里的 `earnings` 词学会了认财报日，而非真正捕捉突发事件”。

#### 2. 新增 5 维零泄露基准特征（并入组别 A）
所有新增基准特征均强制校验 `known_at <= cut`（以早盘 `09:30 EST` 截断为例，只使用前一交易日 `16:00 EST` 之前的 VIX 收盘价与事前公开的财报日程）：

| 特征名 | 数学定义与获取方式 | 零泄露校验规则 | 金融含义 |
| :--- | :--- | :--- | :--- |
| `is_earnings_window` | 二值变量 $\in \{0, 1\}$：标的季度财报发布时刻是否落在 $(T_{t-1}\text{ 16:00 EST},\ T_t\text{ 09:30 EST}]$ 内（联合校验 SEC 8-K Item 2.02 接受时间与 Yahoo Earnings Calendar） | 财报日期为事前数周公布；Item 2.02 接受时间必须 $\le$ `cut` | 显式分离“已知财报催化”与“未知突发事件” |
| `is_earnings_pre1d` | 二值变量 $\in \{0, 1\}$：下一交易日是否为财报发布窗口（财报前一日） | 仅依赖事前公开财报日历 | 捕捉财报前资金博弈与期权押注导致的波动抬升 |
| `vix_close` | 前一交易日 CBOE `^VIX` 收盘点位 | `vix_date < target_date` | 反映全市场期权隐含波动率水平（宏观风险温度计） |
| `vix_ret1` | $\ln(\text{VIX}_{t-1} / \text{VIX}_{t-2})$ | 仅用 $t-1$ 与 $t-2$ 日收盘 | 捕捉前一日全市场恐慌情绪的边际跳升 |
| `vix_z20` | $(\text{VIX}_{t-1} - \text{MA}_{20}(\text{VIX}_{t-2})) / \text{STD}_{20}(\text{VIX}_{t-2})$ | 严格使用前 20 日滚动统计量 | 衡量当前 VIX 相对近一月常态的偏离程度 |

#### 3. 消融升级与“非财报常规日”硬核切片
- **组别定义升级**：`A_v4` = 量价 + QQQ + VIX + 财报日历；`C_v4` = `A_v4` + 去噪情感/新颖度；`D_v4` = `C_v4` + 簇级累积冲击 + 最低情感 + Tier 1 三级事件标志。
- **强制输出分层评估表（`results/v4/earnings_split.csv`）**：在测试集上将早盘样本拆分为：
  1. **全量早盘（All AM）**；
  2. **非财报常规日（`is_earnings_window == 0`）**：专门检验剔除已知财报日后，`D_v4 - A_v4` 的 AUC 与 PR-AUC 增量是否依然显著为正；
  3. **财报窗口（`is_earnings_window == 1`）**：检验 DeepSeek 对财报超预期/指引下调的语义解读，能否在已知“今天发财报”的基础上进一步区分“温和财报”与“暴雷/暴涨财报”。

---

### 方向 2（P0 · 统计效能突破）：扩充至 8–9 只科技权重股 + 预注册 Granger 检验族 + 真前瞻留出集

#### 1. 横截面标的池扩充方案
单标的 NVDA 样本外仅 241 个早盘（28 个异常），检验功效严重受限；扩充至 **9 只高流动性科技与半导体权重股**，可将联合早盘样本外观察量从 723 个提升至 **~2,169 个**，样本外异常事件数从 72 个跃升至 **~210–250 个**（配对自助法置信区间宽度预计收窄约 40%）：

| 标的 | 公司 / 板块属性 | SEC CIK | 纳入目的与语料预期 |
| :--- | :--- | :--- | :--- |
| `NVDA`、`MSFT`、`TSLA` | 原有 3 只核心标的（AI 算力、云软件、新能源/自动驾驶） | `0001045810` / `0000789019` / `0001318605` | 保持与 V1–V3 纵向可比；为 `MSFT` 补充主流财经源使其覆盖率 $\ge 70\%$ |
| `AAPL`、`AMZN`、`META`、`GOOGL` | 超大盘科技龙头（消费电子、电商云、社交广告、搜索 AI） | `0000320193` / `0001018724` / `0001326801` / `0001652044` | 财经媒体报道密度极高、SEC 8-K/反垄断/产品发布事件丰富 |
| `AMD`、`AVGO` | 半导体与算力产业链同业（与 NVDA 同板块） | `0000002488` / `0001730168` | 与 NVDA 形成同行业横向对照，放大算力产业链事件样本量 |

- **语料质量准入机制**：每只标的仍按方案三项门槛独立审计（半日覆盖率 $\ge 70\%$、全语料负面 $\ge 15\%$、各折训练期实质事件 $\ge 10\%$）。结果表中并列输出 **Full-9 Pooled（全部 9 股）** 与 **Qualified-Only Pooled（仅全部达标标的）**，保证学术透明性。

#### 2. 预注册 Granger 先导检验族（解决 36 项多重惩罚稀释）
V3 中 `SubstantiveImpactSum` 在纯盘中 B2 的 Lag=1, 2, 3 原始 $p$ 值均小于 `0.01`（最低 `0.0028`），却因同时检验了 36 项（包含已知噪音过高的 PM 与次要特征）导致 BH 校正后 `q = 0.0745`。
- **V4 预注册主检验族（仅 4 项核心假设）**：
  - 范围：联合面板早盘（AM），股票固定效应 + 交易日聚类标准误；
  - 变量：`ClusterImpactSum`（簇级累积冲击）与 `SentStd`（实质新闻情绪分歧度）；
  - 目标：`B1`（含跳空）与 `B2`（纯盘中）的 Lag=1–2 联合 $F$ 检验；
  - 校正：对这 4 项预注册主假设做 Benjamini–Hochberg 校正（`q_primary`），其余 PM / Lag=3 / 单标的检验归入探索性附录表单独校正，避免探索性切片稀释核心假设的统计效能。

#### 3. 真前瞻样本外验证协议（True Out-of-Time Forward Holdout）
为彻底消除 V1–V3 同一历史测试集的回顾性迭代偏差，以及“历史修订文章缺少首发标题快照”的局限：
1. **冻结截止日**：以 **2026 年 11 月 19 日**为历史开发截止线，冻结 V4 全部代码、特征公式、超参数与提示词版本，将配置摘要写入 `results/v4/frozen_manifest.json`（含代码 SHA-256）。
2. **准实时前瞻采集与打分（`v4_forward_watcher.py`）**：
   - 从 **2026 年 11 月 20 日**起，在每个美股交易日美东 `09:25 EST` 自动运行一次（Windows 任务计划程序或 cron）：
   - 抓取 9 只标的在 $(T_{t-1}\text{ 12:30 EST},\ T_t\text{ 09:25 EST}]$ 内的新闻，记录本地首次抓取时间戳 `first_seen_utc` 与原始首发标题（锁定不变）；
   - 调用 DeepSeek 闭卷评分（复用缓存，单日 9 股约 30–60 条新闻，日耗不足 **$0.003**）；
   - 用截至 $T_{t-1}$ 的冻结模型输出当日 `09:30` 的 B1/B2 风险分数与预警清单，落盘至 `results/v4_forward/predictions.jsonl`；
   - 次日收盘后自动回填真实 `sigma`、`z` 与 `y`，连续累积 10–20 个交易日（约 90–180 个全新早盘观察）作为**零触碰真前瞻检验报告**。

---

### 方向 3（P1 · 语义深度升级）：高影响新闻两阶段精评 + 事件聚类与新颖度（Novelty）特征

#### 1. 两阶段分级语义解析（控成本、提精度）
- **Stage 1（全量标题粗筛）**：沿用 V3 的 `deepseek-flash`（`thinking={"type": "disabled"}`, `temperature=0.0`, 严禁 `tools`, `response_format={"type": "json_object"}`），100% 复用既有 `cache/deepseek_cache.jsonl`。
- **Stage 2（高影响报道“标题 + 导语”精评）**：
  - **触发条件**：仅对 `source_tier == 1`（SEC filing / CNBC / AP / 官方新闻稿）且 Stage 1 判为 `impact_level >= 2` 的实质性报道（约占总语料 10%–15%）触发；
  - **输入内容**：标题 + 抓取自原页面首段或 SEC 8-K Item 摘要的导语文本（硬截断至 $\le 150$ 词，防止长文噪声与 token 膨胀）；
  - **独立缓存与提示词**：写入 `cache/deepseek_lead_cache.jsonl`（键为 `sha256(MODEL + LEAD_PROMPT_V4 + title + lead)`），除精校 `sentiment_score` 与 `impact_level` 外，新增输出 `catalyst_certainty`（`confirmed` 已落地事实 vs. `speculative` 传闻/预期）字段。

#### 2. 本地事件聚类与新颖度（Novelty）特征构造（零 API 成本）
同一重大消息在 21 小时早盘窗口内常被 10–20 家媒体反复转述，V3 直接累加导致“媒体复读次数”与“独立冲击事件数”混为一谈。在 `v4_features.py` 中新增轻量本地文本聚类层：
1. **同窗事件去重聚类**：
   - 仅在过滤掉 `market_noise` 后的实质性新闻集合内，使用字符/词元 1–2 gram TF-IDF 计算窗口内两两标题的余弦相似度；
   - 将余弦相似度 $> 0.65$ 的报道通过连通分量合并为同一**事件簇** $C_k$（$k = 1, \dots, K$）。
2. **新增 3 维抗复读特征（替换或补充 V3 计数特征）**：
   - `UniqueEventCount` $= \ln(1 + K)$：窗口内独立实质事件数量；
   - `ClusterImpactSum` $= \ln\left(1 + \sum_{k=1}^{K} \max_{i \in C_k}\left(w_i \cdot \max(0,\ \text{impact\_level}_i - 1)\right)\right)$：每个事件簇只取其最高有效冲击（按信源权重与时间衰减加权）再跨簇求和，彻底杜绝同一旧闻被 20 家媒体复读时把冲击分数虚增 20 倍；
   - `NoveltyRatio`：计算当前窗口每个事件簇 $C_k$ 与前 3 个同类窗口历史实质新闻标题的最大余弦相似度，定义最大相似度 $< 0.55$ 的簇为“新发事件”，计算新发事件数占 $K$ 的比例（窗口无实质事件时记 `0`）。

---

### 方向 4（P1 · 建模收口）：训练期内生 Winsorize 截尾 + 按目标物理特性路由主模型

#### 1. 修复 11 月 12 日 L2 线性外推饱和（零泄露内生截尾）
- **V3 暴露的缺陷**：11 月 12 日 08:47 时 B1 的 L2 预测分数已外推至 `0.810`（落入 sigmoid 饱和区），导致 08:48 正式 CEO 辞职公告进入窗口后，原始集成分反而从 `0.810` 微降至 `0.802`。
- **精确代码修复（写入 `v4_models.py`）**：
  在 `StandardScaler` 之前插入严格只在每折训练集拟合的 `TrainQuantileWinsorizer`：
  ```python
  class TrainQuantileWinsorizer(BaseEstimator, TransformerMixin):
      def __init__(self, low=0.01, high=0.99, cols=None):
          self.low, self.high, self.cols = low, high, cols

      def fit(self, X, y=None):
          X_arr = np.asarray(X, dtype=float)
          self.lo_ = np.nanquantile(X_arr, self.low, axis=0)
          self.hi_ = np.nanquantile(X_arr, self.high, axis=0)
          return self

      def transform(self, X):
          X_arr = np.asarray(X, dtype=float).copy()
          return np.clip(X_arr, self.lo_, self.hi_)
  ```
  - 作用：对右偏重尾特征（`ClusterImpactSum`、`NewsShock`、`volume_ratio`、`vix_z20`、`Interaction`）按训练折 `[1%, 99%]` 分位截尾后再标准化，既保留极端高风险排序，又避免单个重尾特征直接将线性 logit 顶死在饱和区；
  - 验收检查：在 11 月 12 日 08:47 $\to$ 08:48 $\to$ 08:49 回放中，核验 B1 与 B2 的**原始模型分数（未叠加规则前）**均呈现单调正向跃升。

#### 2. 按目标物理特性路由主模型（B1 vs. B2 差异化架构）
基于 V3 清晰一致的实证规律，不再强求 B1 与 B2 使用同一架构：
- **含跳空早盘（B1）主模型**：`0.7 L2(Winsorized) + 0.3 MonotoneLightGBM` 分类集成。
  - 单调约束列：`ClusterImpactSum`、`NewsShock`、`HasLevel3Event`、`is_earnings_window`、`vix_z20`（均为 `+1` 非递减约束）。
- **纯盘中早盘（B2，无隔夜跳空）主模型**：`0.5 Ridge(Winsorized) + 0.5 HuberLightGBM` 连续波动率回归集成。
  - 训练目标：`clip(z_intraday, -2.0, 6.0)`；
  - 集成方式：两模型在每折训练期后 35% 前向验证集上分别做 Z-Score 标准化（统一量纲）后按 `0.5 / 0.5` 等权融合，直接用于计算测试集 `ROC-AUC`、`PR-AUC`、`Recall@Top15%` 与 `Lift@Top5%`，预警线与简报线仍由训练验证集的 85% / 95% 分位数内生确定。

---

## 四、分阶段实施计划、费用预算与量化验收标准

| 阶段 | 范围与核心任务 | CLI 入口（建议） | 预计新增 API 费用 | 量化验收标准 |
| :--- | :--- | :--- | :--- | :--- |
| **Phase A：3 股零费用精修（半天，适合课程快速收口）** | 1. 接入 `^VIX` 与 3 股财报日历；<br>2. 新增非财报常规日切片表；<br>3. 上线同窗事件聚类与 `NoveltyRatio`；<br>4. 上线 `TrainQuantileWinsorizer` + B2 回归集成路由 | `.\.venv\Scripts\python run_experiment_pipeline.py --study v4 --phase a` | **$0**（100% 复用现有缓存） | 1. 11 月 12 日 08:47 $\to$ 08:48 B1/B2 原始分数双双正向跃升；<br>2. 非财报常规日切片上 `D_v4 - A_v4` 保持正向增量；<br>3. 联合 B1 保守 AUC 突破 **`0.70+`**，B2 回归集成突破 **`0.66+`** |
| **Phase B：9 股面板扩充 + 高影响导语精评（1–2 天，适合论文级实证）** | 1. 补齐 MSFT 语料并扩充至 9 只科技龙头（~20,000–25,000 条新闻）；<br>2. 对 Tier 1 `impact >= 2` 报道运行二阶段导语精评；<br>3. 重跑 9 股联合消融与 4 项预注册 Granger 检验 | `.\.venv\Scripts\python run_experiment_pipeline.py --study v4 --phase b --dry-run`<br>确认估价后去掉 `--dry-run` | **约 $0.50–$0.90**（空闲时段，仍受 $2 保护线约束） | 1. 联合早盘样本外异常数 $\ge 200$；<br>2. 保守口径 `D_v4 - A_v4` 95% 区间显著大于零且区间宽度收窄；<br>3. 预注册 4 项 Granger 检验至少 1 项通过 BH 校正（**`q < 0.05`**） |
| **Phase C：真前瞻留出验证（2–4 周自动运行）** | 冻结 V4 全部代码与模型配置，自 2026-11-20 起每日 `09:25 EST` 自动执行 `v4_forward_watcher.py`，记录首发时间戳并次日自动核验 | `.\.venv\Scripts\python v4_forward_watcher.py --run-once` | **约 $0.05**（20 个交易日累计） | 输出零触碰真前瞻评估表（含实时 `first_seen_utc` 与首发标题快照），验证线上排序能力与回测一致 |

---

## 五、V4 关键输出产物清单（`results/v4/`）

为保证与 V3 报告无缝衔接及审计可复现，V4 统一输出以下核心文件：
- `results/v4/ablation.csv`：9 股联合（Full / Qualified）与单标的在 B1/B2、保守/有界口径下的完整消融表（含 `ROC_AUC`、`PR_AUC`、`Recall`、`Precision`、`Recall_Top15`、`Lift_Top5`、`Brier`）；
- `results/v4/earnings_split.csv`：**财报窗口 vs. 非财报常规日**独立分层消融对比表；
- `results/v4/paired_auc.csv`：5 交易日分块配对自助法（1,000 次）置信区间表（`D_v4 - A_v4` 与 `D_v4 - C_v4`）；
- `results/v4/pooled_granger.csv`：区分为 `primary_preregistered`（4 项主检验，单独 BH 校正）与 `exploratory` 两层的面板 Granger 检验结果；
- `results/v4/event_replay.csv`：Winsorize 修复后的 11 月 12 日 08:47 / 08:48 / 08:49 / 09:30 原始分与规则叠加分明细；
- `results/v4/audit.json` 与 `results/v4/frozen_manifest.json`：API 费用账本校验、SHAP 最大重构误差、语料三门槛审计及前瞻冻结哈希。
