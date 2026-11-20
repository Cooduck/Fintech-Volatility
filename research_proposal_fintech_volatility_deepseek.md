# 结合 DeepSeek 大模型情感与事件分析的股市异常波动预警与归因研究方案

## 一、研究动机（Research Motivation）

### 1. 现实背景：突发舆情与“黑天鹅”事件对金融市场的剧烈冲击
金融市场并非仅由冰冷的财务报表和历史 K 线驱动，重大宏观政治事件与核心企业突发新闻往往在数小时内重塑市场预期，引发剧烈的**异常波动（Abnormal Volatility）**。以 2026 年 11 月的真实市场事件为例：
- **公司治理突发事件冲击**：2026 年 11 月 9 日，[英伟达董事会发布独立调查声明](https://nvidianews.nvidia.com/news/statement-from-nvidia-board-of-directors)；随后在 2026 年 11 月 12 日，[英伟达官方宣布创始人黄仁勋辞去 CEO 职务并由首席财务官 Colette Kress 担任临时 CEO](https://nvidianews.nvidia.com/news/nvidia-announces-ceo-transition)。这一突发高管变动消息迅速引爆全球科技与财经媒体，导致[英伟达市值在短时间内跌破 4 万亿美元关口](https://www.aljazeera.com/news/2026/11/12/ai-titan-jensen-huang-out-as-nvidia-ceo)，并引发算力供应链板块的连锁震荡。
- **宏观政策与政治周期扰动**：2026 年 11 月初[美国中期选举相关结果与国会席位格局变化](https://www.cfr.org/articles/2026-election-by-the-numbers)公布后，市场对财政支出、新能源补贴、医药监管与科技反垄断政策的预期发生快速重构，相关行业指数出现显著的跳空与波动率跃升（如需核实各选区官方计票结果，可参阅 [AP News 2026 众议院选举结果追踪](https://apnews.com/projects/elections-2026/us-house-results/) 与 [AP News 2026 参议院选举结果追踪](https://apnews.com/projects/elections-2026/us-senate-results/)）。

### 2. 面向小白的直观拆解：为什么不直接“预测涨跌”，而是做“异常波动预警与归因”？
很多初学者做金融 AI 项目时，第一反应是“用大模型预测明天股价涨多少”。但在学术界与工业界，直接预测涨跌面临两大死穴：
- **信噪比极低（随机游走困境）**：日常无重大事件时，股价小幅涨跌受高频交易与随机噪声主导，强行预测涨跌往往准确率接近 50%（抛硬币水平），课堂汇报时很难解释为什么模型有时候准、有时候不准。
- **“波动率聚集”与事件驱动规律更显著**：金融学中著名的 **波动率聚集效应（Volatility Clustering）** 指出，虽然涨跌方向难测，但**“市场剧烈震荡的程度（波动率）”具有极强的可预测性与事件伴随性**。就像地震预警一样，当董事会调查声明、反常舆情分歧或政策转向信号密集出现时，系统不需要赌明天涨 3% 还是跌 3%，只要精准发出**“下一交易时段或交易日将出现超过历史均值 1.5~2 倍标准差的异常波动（高危风险预警）”**，就能为风控减仓、期权对冲和保证金管理提供极高商业价值。

### 3. 从传统词典、本地 FinBERT 到 DeepSeek API 的范式升级
在原方案的文本处理路线之上，我们将核心语义引擎从“本地部署的小模型”全面升级为**云端 DeepSeek 平台 API**，其核心动因与能力跨越体现在三个层面：
- **突破纯量价与传统词典的局限**：传统 GARCH 或纯量价模型属于“看着后视镜开车”；早期的 [Loughran-McDonald 金融情感词典](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) 仅统计正负面词频，面对“业绩超预期但下调下季指引”“尽管面临调查风波，但 Blackwell 订单依然饱满”等复杂转折与反讽语境极易误判。
- **突破本地小模型（如 FinBERT）的部署与语义瓶颈**：以 [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) 为代表的 BERT 类模型虽然能输出正/负/中性三分类概率，但在课程实践与真实应用中存在三大痛点：① 需要本地安装数 GB 的 PyTorch/Transformers 依赖并下载模型权重，无 GPU 电脑推理缓慢；② 仅支持 512 token 截断长度，且主要针对英文语料，难以无缝处理中英双语财经快讯与长篇公告；③ 只能输出单一情感标签，无法判断**事件类型**与**冲击烈度**，更无法生成人类可读的归因解释。
- **DeepSeek API 的轻量化与高维语义优势**：通过调用 [DeepSeek 开放平台 API](https://api-docs.deepseek.com/zh-cn/)（原生兼容 OpenAI SDK），使用者**无需本地 GPU、无需下载任何权重，只需在项目本地 `.env` 文件中配置 `DEEPSEEK_API_KEY`**，即可直接调用当前主力的 `deepseek-flash`（由 DeepSeek-V4.1-Flash 驱动）或 `deepseek-v4-pro` 模型（均支持 1M 超长上下文与 [JSON Output 结构化输出](https://api-docs.deepseek.com/zh-cn/guides/json_mode)）。大模型在单次纯离线推理调用（**严格关闭联网搜索**）中即可同步提取**连续情感得分、事件冲击等级、事件分类标签与归因摘要**，实现“低门槛部署 + 高维特征抽取 + 自然语言归因报告”的一体化闭环。

---

## 二、研究目标（Research Objectives）

本项目旨在面向“金融科技与人工智能”课程实践，构建一套**零本地算力门槛（仅需本地 `.env` 提供 APIKEY）、防因果倒置且兼顾盘中即时冲击、可复现、强可解释的“DeepSeek 驱动多模态股市异常波动预警与归因系统”**，具体包含以下四个可量化子目标：

1. **构建“防因果倒置 + 降衰减”的多粒度行情—新闻对齐数据集**：基于开源的 [FNSPID 大规模金融新闻与股价时间序列数据集](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) 及 Yahoo Finance 实时/小时级行情接口，构建重点科技股（如 NVDA、MSFT、TSLA）与宏观指数（SPY/QQQ）数据集；同时支持**半日双时段滚动预测（AM/PM Session-Split，捕捉即时盘中冲击）**与**时间衰减加权跨日预测（捕捉跨日波动延续）**，并在 2026 年 11 月重大事件（如[黄仁勋卸任英伟达 CEO 事件](https://nvidianews.nvidia.com/news/nvidia-announces-ceo-transition)）中引入 **15 分钟分时事件研究**。
2. **设计“快慢双引擎 + 严禁联网”DeepSeek API 财经舆情解析模块**：
   - **批量结构化打分引擎**：调用 `deepseek-flash`（**关闭联网搜索** + 关闭思考模式 + 强制 JSON 输出 + 批量打包 + 自动上下文缓存），将原始中英文新闻转化为**加权净情感均值、情绪分歧度、突发热度冲击、平均事件冲击等级、高危事件触发占比**等 8 维数值特征，并内置本地哈希缓存确保“一次调用、永久复用、零重复计费”。
   - **深度黑天鹅归因引擎**：在系统触发高风险预警时，按需调用 `deepseek-flash` 思考模式（或 `deepseek-v4-pro`，同样关闭联网搜索），将 SHAP 特征贡献与关键新闻融合生成人类可读的**风控归因简报**。
3. **建立融合量价与 DeepSeek 语义特征的异常波动预警模型**：对比正则化逻辑回归（L2-Logistic Regression）、随机森林（Random Forest）与梯度提升树（LightGBM/XGBoost）在不同特征组合与时间粒度下的预警表现，系统验证 DeepSeek 结构化特征（情感 + 事件冲击）相比纯量价、传统词典基线的 AUC-ROC、F1-Score 与 Recall 提升幅度。
4. **实现“统计先导 + 特征贡献 + LLM 语义简报”三层可解释归因**：结合 **Granger 因果检验**、**TreeSHAP 局部特征归因** 与 **DeepSeek 自然语言事件综述**，完成从“数据预警”到“决策可读解释”的完整风控演示。

---

## 三、核心研究问题（Research Questions）

本项目围绕以下三个层层递进的研究问题（RQs）展开实证检验：

- **RQ1（特征增益与粒度对比：DeepSeek 多维语义特征在半日盘中与跨日口径下是否均显著优于基线？）**
  在严格无时间泄露的前提下预测下一交易时段（AM/PM）或下一交易日是否发生“异常波动”时，相比纯量价（Model A）与传统金融词典（Model B），引入 DeepSeek 基础情感（Model C）与“情感 + 事件冲击等级”（Model D）能否显著提升 AUC-ROC 与 Recall？相比日度口径，**半日双时段切分（AM/PM）**能在多大程度上缓解早盘新闻被当日下午提前消化导致的信号衰减？
- **RQ2（信息反应机制问题：情绪极性、情绪分歧与事件冲击等级，谁是波动的真正推手？）**
  引发股市剧烈震荡的究竟是“一边倒的负面情绪（低情感均值）”、“多空激烈分歧（高情绪分歧度）”，还是 DeepSeek 识别出的“高危公司治理/宏观政策冲击（高冲击等级）”？这些舆情特征在时间序列上是否显著领先于真实波动率跃升（Granger 因果先导）？
- **RQ3（黑天鹅分时因果与防泄露问题：在重大突发事件中，系统如何证明“新闻在先、波动在后”并量化归因？）**
  以 [2026年11月英伟达 CEO 突发变动](https://nvidianews.nvidia.com/news/nvidia-announces-ceo-transition) 及宏观政策窗口为例，在关闭 LLM 联网搜索并遵循信息隔离提示词的前提下，结合 **15 分钟分时级时间戳验证**，系统能否排除盘后复盘新闻干扰、在主冲击波全面展开前发出高置信度预警，并通过 SHAP 量化权重与 DeepSeek 归因简报精准定位核心诱因？

---

## 四、研究方法与技术框架（Research Methodology）

为了确保方案逻辑透明且易于在汇报时讲清技术细节，我们将系统拆解为五个核心模块，并配以直观类比与具体数值示例。

### 模块 1：异常波动目标定义与“防倒置、抗衰减”三层时间对齐架构

#### 1. 深入剖析两大时间对齐难题：因果倒置 vs. 信号提前消化
在金融舆情建模中，时间窗口切分直接决定实验的科学性，必须同时规避两个方向的陷阱：
- **陷阱一：同日因果倒置（Intraday Reverse Causality）**：若用第 $t$ 日全天新闻预测第 $t$ 日波动率，上午 10:00 暴跌、下午 14:30 才发布的“暴跌复盘新闻”会被模型当成预测特征，导致严重的虚假高准确率。
- **陷阱二：日度跨天信号衰减（Priced-In / Lag Attenuation）**：若仅做粗粒度日度跨天预测（第 $t$ 日收盘前新闻 $\to$ 第 $t+1$ 日波动），则第 $t$ 日上午 10:00 发布的新闻往往在 $t$ 日下午就已被市场消化并引发了当日波动，到 $t+1$ 日时：
  1. 常规新闻的**即时冲击信号已大幅衰减**；
  2. 该新闻引发的波动已经体现在当日量价特征 $\sigma_t$ 中，导致模型容易把功劳归给“历史波动率惯性”，反而低估了新闻语义的真实边际贡献。

#### 2. 完善后的三层时间对齐方案（半日双时段为主 + 时间衰减跨日为辅 + 15 分钟案例显微镜）
为兼顾**即时冲击捕捉、零因果倒置**与**开源数据可获取性**，本项目构建以下三层对齐架构：

| 方案层级 | 时间粒度与截断规则 | 特征新闻窗口（严格在先） | 预测目标窗口（严格在后） | 核心金融学含义与数据来源 |
| :--- | :--- | :--- | :--- | :--- |
| **口径 B（推荐核心：半日双时段滚动预测 AM / PM）** | **早盘时段（AM）**：截断于当日开盘 `09:30 EST` | 昨日收盘后至今日开盘前 $(T_{t-1}\text{ 16:00},\ T_t\text{ 09:30}]$（隔夜公告、盘后财报、盘前快讯） | 当日**早盘** `09:30–12:30 EST` 已实现波动率 $\sigma_{t,\text{AM}}$（含隔夜跳空） | 捕捉隔夜与盘前重大公告引发的开盘即时巨震；Yahoo Finance 1 小时 K 线免费支持近 730 天（约 **1,000 个半日样本**） |
| **口径 B（推荐核心：半日双时段滚动预测 AM / PM）** | **午盘时段（PM）**：截断于午间 `12:30 EST` | 今日早盘及午间 $(T_t\text{ 09:30},\ T_t\text{ 12:30}]$（早盘突发新闻、盘中研报、政策讲话） | 当日**午盘/尾盘** `12:30–16:00 EST` 已实现波动率 $\sigma_{t,\text{PM}}$ | 彻底解决“上午新闻作用于下午市场”的捕捉难题，同时 100% 切断下午复盘新闻反向污染上午行情 |
| **口径 A（优化版：时间衰减跨日预测）** | **日度收盘截断**：截断于 `16:00 EST` | $(T_{t-1}\text{ 16:00},\ T_t\text{ 16:00}]$，对新闻施加**指数时间衰减权重** $w_i = e^{-\lambda \Delta\tau_i}$（临近收盘权重高、早盘已消化旧闻降权） | 未来第 $t+1$ 个交易日综合波动率 $\sigma_{t+1}$ | 适用于多年期开源日度面板（如 FNSPID 2019–2023），专门检验重大事件的**跨日波动率持续与余震效应（Volatility Persistence）** |
| **案例显微镜层（高频事件研究）** | **15 分钟 K 线对齐** | 精确到分钟的突发新闻时间戳（UTC 统一转 EST） | 新闻发布前后 $[-2\text{h}, +4\text{h}]$ 分时已实现波动率 | 聚焦近 60 天内的 2026 年 11 月英伟达与中期选举窗口，从微观时间轴验证“新闻在先、波动在后” |

> **跨市场适配说明**：美股常规交易时段为 `09:30–16:00 EST`（连续交易 6.5 小时，无物理午休），华尔街微观结构研究通常以 `12:30 EST`（或 `13:00 EST`）作为早盘（AM）与午盘（PM）分界点；若项目后续扩展至 **A 股（09:30–11:30 / 13:00–15:00）** 或 **港股（09:30–12:00 / 13:00–16:00）**，则可直接沿用天然的物理午休作为截断边界——上午盘依赖前日收盘至今日开盘前新闻，下午盘依赖今日早盘盘中与午间休市新闻，代码仅需切换市场时区配置即可无缝复用。

#### 3. 异常波动率标签的数学定义（同时兼容半日时段与日度口径）
- **直观拆解**：平时某只股票每个半日时段（或每个交易日）的波动幅度在历史正常区间内属于“正常呼吸”；若未来时段突然出现超过过去 20 个同类时段均值 $1.5$ 倍标准差以上的剧烈震荡，就像突发高烧一样，记为“异常波动”。
- **统一数学公式**（记任意预测时段为 $s$，在口径 B 中为第 $t$ 日早盘或午盘，在口径 A 中为第 $t+1$ 日）：
  1. **时段综合波动率 $\sigma_s$**：
     - 对于**半日时段（口径 B）**：利用时段内的 1 小时/15 分钟 K 线收益率 $\{r_{s,1}, \dots, r_{s,m}\}$ 与时段极差 $(H_s, L_s)$，结合隔夜跳空（早盘包含 $r_{\text{gap}} = \ln(O_{t,\text{AM}} / P_{t-1,\text{close}})$），定义：
       $$\sigma_s = \max\left(\sqrt{\sum_{k=1}^{m} r_{s,k}^2},\ \sqrt{\frac{1}{4\ln 2}\left(\ln\frac{H_s}{L_s}\right)^2}\right)$$
     - 对于**日度时段（口径 A）**：$\sigma_t = \max\left(|r_t|,\ \sqrt{\frac{1}{4\ln 2}\left(\ln\frac{H_t}{L_t}\right)^2}\right)$。
  2. **同类时段标准化 Z-Score（消除盘中 U 型周期干扰）**：由于美股早盘天然比午盘波动更大，在口径 B 中**早盘与午盘分别对其过去 $W=20$ 个同类时段（即过去 20 个 AM 或过去 20 个 PM）**计算滚动均值 $\mu_{\sigma, s-1}$ 与标准差 $s_{\sigma, s-1}$：
     $$Z_s = \frac{\sigma_s - \mu_{\sigma, s-1}}{s_{\sigma, s-1}}$$
  3. **预警标签 $Y$**：若未来目标时段满足 $Z_{s_{\text{next}}} > \tau$（默认 $\tau = 1.5$ 或历史 $85\%$ 分位数），标记为异常波动正样本 $Y = 1$，否则为 $0$。
- **具体数值举例**：假设英伟达过去 20 个午盘（12:30–16:00 EST）的平均已实现波动率 $\mu_{\text{PM}} = 1.1\%$，标准差 $s_{\text{PM}} = 0.4\%$。若某日早盘 10:40 突发公司治理调查新闻，当日午盘波动率飙升至 $2.9\%$，则 $Z_{t,\text{PM}} = (2.9\% - 1.1\%) / 0.4\% = 4.5 > 1.5$，该午盘被准确标记为异常波动正样本（$Y = 1$），且触发该预警的正是 `09:30–12:30` 窗口内的早盘突发新闻。

---

### 模块 2：基于 DeepSeek API 的多维财经舆情与事件特征提取
设目标截断窗口内收集到关于某标的的 $N_s$ 条带精确时间戳的新闻 $\{d_{s,1}, d_{s,2}, \dots, d_{s,N_s}\}$。

#### 1. 模型选型与调用策略（结合 2026 年 11 月 DeepSeek 官方接口规范）
根据 [DeepSeek 官方更新日志](https://api-docs.deepseek.com/zh-cn/updates/) 与 [模型与价格文档](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)，旧版模型名 `deepseek-chat` 与 `deepseek-reasoner` 已逐步退役，本项目直接采用官方最新模型标识与分层调用策略：

| 任务场景 | 推荐模型 ID | 思考模式配置（`thinking`） | 联网搜索配置（`tools`） | 核心作用与优势 |
| :--- | :--- | :--- | :--- | :--- |
| **大批量历史/分时段新闻打分**（主力引擎） | `deepseek-flash`（DeepSeek-V4.1-Flash） | `{"type": "disabled"}`（关闭思考，直出 JSON） | **严格关闭（不传 `tools`）** | 响应极快、并发上限高（[最高 2500 并发](https://api-docs.deepseek.com/quick_start/rate_limit/)）、成本极低，适合批量特征抽取 |
| **高危预警时段黑天鹅深度归因**（按需触发） | `deepseek-flash`（开启思考）或 `deepseek-v4-pro` | `{"type": "enabled"}`（开启思维链推理） | **严格关闭（不传 `tools`）** | 仅在预警概率 $P > 0.7$ 时调用，严格基于本地传入新闻与 SHAP 权重生成风控简报 |

> **关键工程细节（防踩坑）**：根据 [Chat Completions API 文档](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion)，若请求中省略 `thinking` 与 `reasoning_effort` 参数，DeepSeek 模型**默认开启思考模式**。因此在批量提取数值特征时，必须显式传入 `extra_body={"thinking": {"type": "disabled"}}` 并设置 `temperature=0.0`，同时**不传入任何联网搜索工具**，确保低成本、低延迟与零外部信息泄露。

#### 2. 单条/批量新闻的结构化 JSON 输出设计
通过设置 `response_format={"type": "json_object"}`（[JSON Output 指南](https://api-docs.deepseek.com/zh-cn/guides/json_mode)），并在 System Prompt 中固定输出 Schema，每次可将同一截断窗口内的 $10\sim 15$ 条新闻打包为一个 Batch 发送，要求 DeepSeek 为每条新闻返回以下 5 个字段：
```json
{
  "results": [
    {
      "id": 1,
      "sentiment_score": -0.85,
      "confidence": 0.92,
      "impact_level": 3,
      "event_type": "governance",
      "rationale": "董事会启动独立调查引发高管稳定性担忧，短期抛压风险极高"
    }
  ]
}
```
- `sentiment_score` $\in [-1.0, +1.0]$：连续情感极性得分（$-1$ 为极度利空，$0$ 为中性，$+1$ 为极度利好）。
- `confidence` $\in [0.0, 1.0]$：模型对情感与影响判断的置信度。
- `impact_level` $\in \{0, 1, 2, 3\}$：事件对短期股价波动的预期冲击烈度（$0$=无影响/噪音，$1$=轻度常规消息，$2$=中度重要事件，$3$=重大黑天鹅/突发冲击）。
- `event_type`：标准金融事件分类枚举（`governance` 公司治理/高管变动、`earnings` 财报/指引、`macro_policy` 宏观/政策、`regulatory` 监管/诉讼、`product_supply` 产品/供应链、`market_noise` 常规市场杂讯）。
- `rationale`：不超过 25 字的单句影响归因摘要。

#### 3. 聚合构造截断窗口多维舆情与事件算子（引入时间衰减权重）
为缓解同一截断窗口内“较早新闻已被部分消化、临近截断时刻的新闻冲击更强”的问题，定义每条新闻的综合权重为**置信度 × 指数时间衰减**：
$$w_{s,i} = c_{s,i} \cdot \exp\left(-\lambda \cdot \Delta \tau_{s,i}\right)$$
其中 $\Delta \tau_{s,i} \ge 0$ 为该条新闻发布时刻距离当前截断时刻（如 `09:30`、`12:30` 或 `16:00`）的小时数，$\lambda$ 为衰减系数（例如半衰期设在 4~6 小时，口径 A 下显著降权早盘旧闻、突出临近收盘新冲击）。据此聚合得到 **8 维语义与事件特征**：
1. **时间衰减加权净情感均值（Time-Decayed Weighted Sentiment Mean）**：
   $$\text{SentMean}_s = \frac{\sum_{i=1}^{N_s} w_{s,i} \cdot s_{s,i}}{\sum_{i=1}^{N_s} w_{s,i} + \epsilon}$$
2. **情绪分歧度（Sentiment Dispersion）**：衡量媒体与市场观点是否激烈撕裂：
   $$\text{SentStd}_s = \sqrt{\frac{1}{N_s}\sum_{i=1}^{N_s} (s_{s,i} - \bar{s}_s)^2}$$
3. **近期情感动量（Sentiment Momentum）**：当前窗口净情感相对前 3 个同类窗口均值的变化量 $\Delta \text{SentMean}_s$。
4. **极端负面新闻占比（Extreme Negative Ratio）**：满足 $s_{s,i} < -0.5$ 的新闻占比 $\text{NegRatio}_s$。
5. **突发舆情热度冲击（News Shock Intensity）**：当前窗口新闻量相对于过去 20 个同类窗口均值的激增倍数：
   $$\text{NewsShock}_s = \frac{N_s}{\text{MA}_{20}(N_{s-1}) + 1}$$
6. **情绪-热度交互项（Sentiment-Shock Interaction）**：$|\text{SentMean}_s| \times \text{NewsShock}_s$。
7. **平均事件冲击等级（Mean Impact Level，DeepSeek 新增）**：$\text{ImpactMean}_s = \frac{\sum w_{s,i} \cdot \text{impact\_level}_{s,i}}{\sum w_{s,i} + \epsilon} \in [0, 3]$，直接过滤掉“情绪偏激但无关痛痒的八卦噪音”。
8. **高危事件触发占比（High-Impact Event Ratio，DeepSeek 新增）**：窗口内 `impact_level == 3` 或属于 `governance / regulatory / macro_policy` 突发类别的新闻占比 $\text{HighImpactRatio}_s$。

#### 4. 学术严谨性护栏：四重机制防范 LLM 前瞻偏差（Look-Ahead Bias）
由于通用大模型兼具参数记忆与可选联网检索能力，直接使用 LLM 评估历史财经数据时必须严防“提前知道未来信息”。本项目建立四重硬性隔离护栏：
1. **强制关闭智能联网搜索（Pure Closed-Book API Call）**：根据 [DeepSeek Chat Completions API 文档](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion)，联网搜索依赖在请求中挂载 `deepseek_web_search` / `deepseek_read_url` 工具。本项目在代码封装层**硬性禁止传入任何 `tools` 参数**（锁定 `ENABLE_WEB_SEARCH = False`），并对响应体做校验（确保未触发任何 `server_tool_calls` 或网页引用 `annotations`），从物理通道彻底切断模型通过联网搜索查询未来股价走势、事后财报或复盘报道的可能。
2. **严格文本隔离提示词（Information-Isolation Prompt）**：在 System Prompt 中明确指令——*“你必须仅根据当前输入的新闻文本进行即时判断，严禁调用任何关于该事件后续股价走势或事后结果的外部记忆，假设当前正处于新闻发布后的第一分钟。”*
3. **历史数据日期脱敏选项（可选稳健性检验）**：对历史训练集（如 2019–2023 年 FNSPID 数据）支持 `--mask-dates` 参数，在送入 DeepSeek 前隐去新闻正文中的具体年份日期，交叉验证情感打分不受模型参数时间记忆污染。
4. **近实时外推样本窗口**：2026 年 11 月的英伟达高管变动与美国中期选举案例处于最新时间窗口，天然具备零历史记忆泄露属性，适合作为最严苛的样本外泛化检验。

---

### 模块 3：本地 `.env` 密钥配置、项目内虚拟环境与缓存架构

为了让使用者**仅在项目本地 `.env` 文件中填写 APIKEY、并在项目独立虚拟环境中一键部署运行**，同时做到**安全不泄露密钥、省钱不重复扣费、报错不中断流程**，系统采用以下 4 层工程规范：

1. **统一本地 `.env` 文件读取 APIKEY**：
   - 项目根目录提供 `.env.example` 模板文件（内容为 `DEEPSEEK_API_KEY=your_api_key_here`），使用者只需将其复制为 `.env` 并填入自己的密钥即可。
   - 程序启动时通过 `python-dotenv` 的 `load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)` 显式加载项目本地 `.env`；若未检测到 `.env` 文件或密钥为空，程序立即输出清晰的中文配置提示并安全退出。
   - `.gitignore` 默认将 `.env` 与 `.venv/` 加入忽略列表，杜绝误提交密钥或环境目录至代码仓库。
2. **项目内独立虚拟环境（`venv`）与轻量化依赖**：
   - 使用 Python 标准库内置的 `venv` 在项目根目录创建隔离环境（`.venv/`），避免污染全局 Python 环境或与其他课程项目产生依赖冲突。
   - 彻底移除 `torch`、`transformers` 等数 GB 级重型框架，所有依赖通过 `requirements.txt`（核心为 `openai`、`python-dotenv`、`pandas`、`numpy`、`scikit-learn`、`lightgbm`、`shap`、`statsmodels`、`matplotlib`、`yfinance`）在 `.venv` 内数秒完成安装。
   - 固定 `base_url="https://api.deepseek.com"`，底层完全复用 OpenAI 官方 Python SDK。
3. **本地持久化哈希缓存 + 断点续传（防重复计费）**：
   - 每条新闻在调用 API 前，先计算确定性主键：`cache_key = sha256(model_id + prompt_version + normalized_text)`。
   - 优先查询本地 `cache/deepseek_cache.jsonl`（或 SQLite 数据库）：命中缓存则直接读取本地结果（耗时 $<1\text{ms}$，费用 $0$ 元）；未命中才发起 API 请求并实时落盘。
   - 即使网络中断或后续反复切换口径 A / 口径 B、调整下游 LightGBM/SHAP 超参数，**绝不会对同一条新闻重复调用 API**。
4. **成本控制与异常兜底机制**：
   - **自动上下文缓存红利（Context Caching）**：所有请求共享完全一致的 System Prompt 前缀（约 400–600 tokens），自动触发 DeepSeek 服务端上下文缓存命中，根据 [官方定价](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)，缓存命中输入单价可低至未命中的约 2%。
   - **空内容与格式兜底**：依据 [JSON Output 注意事项](https://api-docs.deepseek.com/zh-cn/guides/json_mode)，当模型输出非法 JSON 时接口可能返回纯空格内容（`U+0020`），代码中内置 `content.strip()` 判空 + JSON Schema 校验 + 最多 3 次指数退避重试。
   - **错误分级处理**：对 `408 / 429 / 5xx` 网络与限流错误自动退避重试；对 `401`（APIKEY 无效）与 `402`（账户余额不足）立即中止并输出中文排查指引；支持 `--dry-run` 参数在发请求前预估总 token 消耗与费用。

---

### 模块 4：多模态特征融合与预警分类模型（Multimodal Fusion & Prediction）
将两类特征拼接为特征向量 $\mathbf{x}_s = [\mathbf{x}^{\text{tech}}_s \parallel \mathbf{x}^{\text{deepseek}}_s]$（口径 B 下额外附带 1 维时段标识 `is_afternoon_session`）：
- **基础量价特征（$\mathbf{x}^{\text{tech}}_s$，6 维）**：前两期滞后收益率、过去 5 期与 20 期同类时段滚动波动率、成交量异常放大倍数（Volume Ratio）、相对强弱指标（RSI-14）、上一时段极差幅度（High-Low Spread）。在口径 B 预测午盘时，这些特征实时更新至当日 `12:30 EST` 的早盘已实现状态。
- **DeepSeek 语义与事件特征（$\mathbf{x}^{\text{deepseek}}_s$，8 维）**：时间衰减加权净情感均值（$\text{SentMean}_s$）、近期情感动量（$\Delta \text{SentMean}_s$）、情绪分歧度（$\text{SentStd}_s$）、极端负面占比（$\text{NegRatio}_s$）、新闻热度冲击（$\text{NewsShock}_s$）、情绪-热度交互项（$|\text{SentMean}_s| \times \text{NewsShock}_s$）、平均事件冲击等级（$\text{ImpactMean}_s$）、高危事件占比（$\text{HighImpactRatio}_s$）。
- **分类器与防泄露切分**：采用 **LightGBM**、**随机森林（Random Forest）** 与 **L2 正则化逻辑回归（L2-Logistic Regression）** 进行横向对比，严格采用**时间前向滚动切分（Walk-Forward Split）** 划分训练/测试集；同时对比树模型在极端高新闻量（突发黑天鹅日）下的阈值饱和特性与线性模型的外推风险。

---

### 模块 5：三层可解释归因机制（Granger 因果 + SHAP 量化 + DeepSeek 归因简报）
1. **宏观时序先导检验（Granger Causality）**：建立双变量自回归模型（VAR），检验加入滞后 $1\sim 3$ 阶的 DeepSeek 舆情/冲击特征后，波动率方程残差平方和是否显著下降（$F$ 统计量与 $p$-value），用统计学证明“舆情领先于波动”。
2. **中观特征贡献归因（TreeSHAP）**：将单时段预警概率 $f(\mathbf{x}_s)$ 精确拆解为基准概率与各特征 Shapley 值之和：
   $$f(\mathbf{x}_s) = \phi_0 + \sum_{j \in \text{Tech}} \phi_j + \sum_{k \in \text{DeepSeek}} \phi_k$$
   直观输出诸如“量价惯性贡献 28%，DeepSeek 突发高危事件与负面情绪贡献 72%”的量化拆解。
3. **微观自然语言风控简报（DeepSeek Attribution Brief）**：当预警概率 $P(Y=1) > 0.7$ 时，系统自动将 Top-3 正向推高风险的 SHAP 特征及截断窗口内 `impact_level == 3` 的关键新闻 `rationale` 汇总，调用一次 `deepseek-flash`（开启思考模式 `thinking={"type": "enabled"}`、**保持关闭联网搜索**）生成 150 字以内的结构化风控简报，直接附在可视化看板下方供决策阅读。

---

## 五、实验方案（Experimental Plan：步骤、成本预算与预期结果）

### 1. 实验步骤拆解（5 步闭环）

| 阶段 | 实验步骤名称 | 具体执行内容 | 产出物 |
| :--- | :--- | :--- | :--- |
| **Step 1** | 三层粒度数据获取与防因果倒置对齐 | 获取目标标的（如 NVDA、QQQ）日度 OHLCV（FNSPID / Yahoo Finance）、近 2 年 **1 小时级 K 线**与带精确时间戳新闻，分别构建：**口径 B 半日双时段数据集（AM: 昨日盘后–09:30 预测早盘；PM: 09:30–12:30 预测午盘）**、**口径 A 时间衰减跨日数据集**，以及 2026 年 11 月案例 **15 分钟 K 线** | `aligned_session_am_pm.csv`、`aligned_daily_decay.csv`、`intraday_15m_case.csv` |
| **Step 2** | 双轨文本特征提取与本地缓存 | 分别运行 **(A) 传统金融词典法（Baseline）** 与 **(B) DeepSeek API 批量结构化解析（`deepseek-flash`，关闭联网搜索 + `thinking=disabled` + JSON 输出 + 本地哈希缓存）**，两套口径复用同一份新闻缓存 | `cache/deepseek_cache.jsonl` 与特征表 |
| **Step 3** | 四组消融实验 + 粒度对比实验 | 在严格 Walk-Forward 划分下对比四组特征：<br>① **Model A（Tech-Only）**：仅 6 维量价<br>② **Model B（Tech + Lexicon）**：量价 + 传统词典<br>③ **Model C（Tech + DeepSeek-Sent）**：量价 + DeepSeek 6 维基础情感<br>④ **Model D（Tech + DeepSeek-Full，Ours）**：量价 + DeepSeek 8 维（含事件冲击等级与时间衰减）<br>并横向对比 **口径 B（AM/PM 半日） vs. 口径 A（跨日）** 的语义特征增量增益 | 消融实验与粒度对比表（AUC、F1、Precision、Recall、Brier Score） |
| **Step 4** | Granger 因果与防泄露稳健性检验 | 检验 Lag=1, 2, 3 下 `NewsShock`、`SentStd`、`ImpactMean` 对未来波动率的 Granger 因果显著性；测试不同波动阈值（$Z > 1.2, 1.5, 1.8$）、不同时间衰减系数 $\lambda$ 及 `--mask-dates` 去日期提示词下的稳健性 | Granger 统计表、ROC 曲线与稳健性检验表 |
| **Step 5** | 2026 年 11 月重点事件分时因果与三层归因复盘 | 聚焦 **2026 年 11 月上旬宏观政策波动期** 与 **11 月 9 日–13 日黄仁勋辞去英伟达 CEO 事件窗口**，结合 **15 分钟分时 K 线**验证新闻时间戳领先于盘中波动跃升，绘制四联图、SHAP 瀑布图并输出 DeepSeek 自动生成的风控归因简报 | 课堂汇报看板（`volatility_dashboard.png`、`intraday_event_study.png`）与归因简报 |

### 2. DeepSeek API 成本与耗时预算分析（课程项目友好型）
以 [DeepSeek 官方定价](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) 为基准（`deepseek-flash` 空闲时段输入未命中 $0.15 / 百万 tokens、缓存命中 $0.003 / 百万 tokens、输出 $0.60 / 百万 tokens；北京时间周一至周五 9:00–12:00 及 14:00–18:00 为高峰时段，价格翻倍，其余时段及周末全天为空闲时段）：
- **典型课程实验规模估算**：由于口径 A 与口径 B **共享同一份新闻本地哈希缓存**，每条新闻仅需调用一次 DeepSeek API。假设分析 1~2 只核心标的、总计 **3,000 条财经新闻**，每 10 条打包为 1 次请求（共 300 次 API 调用）：
  - System Prompt 约 500 tokens（第 2~300 次调用触发上下文缓存命中，成本可忽略不计）；
  - 新闻正文输入约 180,000 tokens，JSON 输出约 150,000 tokens；
  - **空闲时段总成本**：输入约 $\$0.027$ + 输出约 $\$0.090 \approx \mathbf{\$0.12}$**（不足人民币 1 元）**；即便在高峰时段全量重跑，总成本也仅约 **$\$0.24$（不足人民币 2 元）**。
  - **运行耗时**：配合 5~10 并发异步请求，300 次调用可在 **1~2 分钟内全部完成**，且结果自动写入本地缓存，后续切换半日/日度口径或反复调参零额外费用。

### 3. 实验预期结果与课堂汇报分析要点
- **预期结果 1（四组消融与半日 vs. 跨日粒度增益）**：
  - 在**口径 B（AM/PM 半日双时段）**下，由于隔夜公告直击早盘、上午盘中新闻直击午盘，避免了早盘新闻被下午提前消化的问题，DeepSeek 语义特征相对纯量价基线的 AUC 增量预计**显著高于**粗粒度日度口径（Model D 相比 Model A 预计 AUC 提升 $0.12\sim 0.18$，F1-Score 提升 $20\%\sim 30\%$）。
  - 在**口径 A（跨日 + 时间衰减加权）**下，临近收盘新闻加权后仍能有效捕捉重大事件的跨日波动延续效应（Model D AUC 预计达到 $0.82\sim 0.88$），且时间衰减加权优于等权聚合。
- **预期结果 2（特征重要性与因果检验）**：
  - 在滞后 1~2 阶上，`ImpactMean`、`NewsShock` 与 `SentStd` 对未来波动率的 Granger 因果检验显著（$p < 0.01$）。
  - TreeSHAP 全局归因中，**“情绪-热度交互项”**、**“平均事件冲击等级”**与**“情绪分歧度”**位居前列，说明“高影响级别的重大事件叠加市场分歧”才是剧烈波动的核心催化剂。
- **预期结果 3（2026 年 11 月英伟达 CEO 离职事件分时验证与三层归因）**：
  - 在 2026 年 11 月 9 日[董事会调查声明](https://nvidianews.nvidia.com/news/statement-from-nvidia-board-of-directors)与 11 月 12 日[黄仁勋卸任公告](https://nvidianews.nvidia.com/news/nvidia-announces-ceo-transition)的 15 分钟分时对齐图中，可清晰观测到新闻发布时间戳严格先于盘中成交量与波动率跃升，排除“先大跌后出新闻”的倒置干扰。
  - 接入 DeepSeek 的系统在对应半日截断窗口内即识别出高危 `governance` 事件（`impact_level = 3`），预警概率迅速拉升至 $0.85$ 以上，成功覆盖[市值跌破 4 万亿美元](https://www.aljazeera.com/news/2026/11/12/ai-titan-jensen-huang-out-as-nvidia-ceo)的主冲击窗口，SHAP 量化归因与 DeepSeek 自动生成的风控简报在风险诱因上高度一致。

---

## 六、本地虚拟环境部署与快速运行指南

### 第一步：在项目内创建并激活虚拟环境（`venv`）
```bash
# 1. 在项目根目录创建独立虚拟环境 .venv
python -m venv .venv

# 2. 激活虚拟环境
# macOS / Linux:
source .venv/bin/activate

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 3. 在虚拟环境中安装轻量级依赖（无需 PyTorch / GPU）
pip install -r requirements.txt
# （requirements.txt 包含: openai, python-dotenv, pandas, numpy, scikit-learn, lightgbm, shap, statsmodels, matplotlib, yfinance）
```

### 第二步：在本地 `.env` 文件中配置 DeepSeek APIKEY
在项目根目录复制模板文件 `.env.example` 为 `.env`（已被 `.gitignore` 忽略），写入你的密钥：
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 第三步：运行实验管线（双模式 + 多口径支持）
- **离线演示模式（无 APIKEY 也可跑通全流程）**：
  ```bash
  python run_experiment_pipeline.py --mode demo
  ```
  自动加载内置缓存样本与 2026 年 11 月英伟达/宏观选举案例库，秒级输出消融结果与可视化看板。
- **DeepSeek 真实 API 分析模式（自动读取项目内 `.env`）**：
  ```bash
  # 第一步：先预估本次分析的 token 消耗与费用（不发起真实扣费请求）
  python run_experiment_pipeline.py --mode deepseek --ticker NVDA --dry-run

  # 第二步（推荐核心口径 B）：半日双时段滚动预测（AM: 隔夜->早盘；PM: 上午+午间->午盘）
  python run_experiment_pipeline.py --mode deepseek --ticker NVDA --model deepseek-flash --granularity halfday --intraday-case 15m

  # 可选（口径 A）：跨日收盘截断 + 指数时间衰减加权（用于多年期日度历史数据集）
  python run_experiment_pipeline.py --mode deepseek --ticker NVDA --model deepseek-flash --granularity daily --time-decay 0.15
  ```

### 附：核心 DeepSeek API 调用最小代码示例（关闭联网搜索 + 仅读 `.env`）
```python
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 严格仅从项目根目录本地 .env 文件加载 APIKEY
env_path = Path(__file__).resolve().parent / ".env"
if not env_path.exists():
    raise FileNotFoundError("未找到本地 .env 文件，请先复制 .env.example 为 .env 并填写 DEEPSEEK_API_KEY。")

load_dotenv(dotenv_path=env_path, override=True)
api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not api_key:
    raise ValueError(".env 文件中未配置有效的 DEEPSEEK_API_KEY。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

# 严禁传入 tools（不启用 deepseek_web_search / deepseek_read_url），确保纯离线闭卷推理
response = client.chat.completions.create(
    model="deepseek-flash",  # 官方主力高性价比模型（DeepSeek-V4.1-Flash）
    temperature=0.0,         # 保证打分确定性与可复现性
    response_format={"type": "json_object"},  # 强制结构化 JSON 输出
    extra_body={"thinking": {"type": "disabled"}},  # 批量打分显式关闭思考模式以提速降本
    messages=[
        {
            "role": "system",
            "content": (
                "You are a quantitative financial risk analyst. Evaluate ONLY the provided news text "
                "assuming it was just published 1 minute ago; NEVER use any future market outcome memory. "
                "Return strict json with keys: sentiment_score (-1.0 to 1.0), confidence (0 to 1), "
                "impact_level (0 to 3), event_type (governance/earnings/macro_policy/regulatory/"
                "product_supply/market_noise), and rationale (concise Chinese summary <= 25 chars)."
            ),
        },
        {
            "role": "user",
            "content": "News headline: NVIDIA announces CEO transition; CFO Colette Kress named interim CEO.",
        },
    ],
)

content = (response.choices[0].message.content or "").strip()
result = json.loads(content)
print(result)
```
