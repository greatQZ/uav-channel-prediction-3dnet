# 3D-NET DFKI 路线规划与论文布局计划

## 1. 总体策略
将 DFKI 在 `AP1.3 / AP2.1 / AP2.2 / AP2.3 / AP3.2 / AP3.4 / AP5.1 / AP5.4` 的任务拆成 **3 条主路线**，按“底座 -> 控制 -> 验证”推进，避免同时铺太开。

默认路线划分：
1. **路线 A：数据与信道智能底座**
   - 覆盖 `AP1.3.5 + AP2.1.1 + AP5.1(部分)`
2. **路线 B：预测驱动的网络控制与资源编排**
   - 覆盖 `AP2.1.2 + AP2.2 + AP2.3 + AP3.2 + AP5.1(部分)`
3. **路线 C：群体通信与最终验证**
   - 覆盖 `AP3.4 + AP5.4 + AP5.1(最终验证指标)`

整体依赖固定为：
- **先做路线 A**
- 再做 **路线 B**
- 最后做 **路线 C**

这 3 条路线分别对应：
- **路线 A**：回答“真实数据不够时，怎样构造可信的 synthetic + pretraining 底座？”
- **路线 B**：回答“如何把预测能力变成 handover / topology / frequency / multipath 的控制能力？”
- **路线 C**：回答“如何在 UAV swarm 与 HIL/field 条件下证明这些能力真的可用？”

## 2. DFKI Arbeitsschritt 与 3 条路线的映射
| AP | DFKI Arbeitsschritt | 路线 | 在整体中的角色 |
|---|---|---|---|
| `AP1.3.3` | Hybrides O-RAN Architekturdesign für NTN | B | 为 AI control/xApp 提供系统接口背景 |
| `AP1.3.4` | KI/ML Integration und Datenpipeline-Design im SMO für NTN | A/B | 定义数据流、训练流、部署流 |
| `AP1.3.5` | Datengetriebene Kanalsynthese für 3D-Netze | A | 路线 A 的核心任务 |
| `AP2.1.1` | Datenaugmentierung und Modell-Pretraining | A | 路线 A 的直接输出任务 |
| `AP2.1.2` | Deep-Learning-basiertes Prädiktives Handover | B | 路线 B 的第一落地点 |
| `AP2.2.1` | ML-gestützte dynamische Topologieanpassung | B | topology control |
| `AP2.3.1` | KI-gestützte Frequenzzuteilung als O-RAN xApp | B | spectrum/resource control |
| `AP2.3.2` | O-RAN Architektur-Evaluierung der xApps | B | system-level xApp evaluation |
| `AP3.2.1` | Analyse asymmetrischer Link-Bedingungen | B | multi-link path state modeling |
| `AP3.2.2` | KI-gestützte Datenflusssteuerung für Multi-Link | B | multipath steering/control |
| `AP3.4.1` | Semantische Datenkompression bei Deep Fades | C | swarm resilience strategy |
| `AP3.4.2` | Proaktive Link-Adaption für den Schwarm | C | swarm-wide proactive control |
| `AP5.1.3` | KPI-Definition für KI-Steuerungssysteme | A/B/C | 三条路线共享评价层 |
| `AP5.4.1` | Hardware-in-the-Loop Integration der KI-Module | C | 实验验证入口 |
| `AP5.4.2` | Feldtests der O-RAN xApps im Drohnenschwarm | C | 最终 demonstrator / field proof |

### 2.1 AP 全覆盖矩阵（主线 + Support）
说明：`主线覆盖` 表示已在 A/B/C 里有明确任务；`背景支撑` 表示已识别但暂不作为主线主角；`部分覆盖` 表示仅覆盖该 AP 家族的部分子项；`未覆盖` 表示在当前 PLAN 中尚无明确落点。

| AP 范围 | 当前状态 | 当前落点 | Support 子线 | 下一步最小交付物 |
|---|---|---|---|---|
| `AP1.3.3` | 背景支撑 | B（接口背景） | S1 | NTN O-RAN 架构约束清单（输入/输出/时序） |
| `AP1.3.4` | 背景支撑 | A/B（数据与集成背景） | S1 | SMO 数据/训练/部署流接口说明 |
| `AP1.3.5` | 主线覆盖 | A | - | 合成器与 realism 验证闭环 |
| `AP2.1.1` | 主线覆盖 | A | - | augmentation + pretraining 实验资产 |
| `AP2.1.2` | 主线覆盖 | B | - | predictive handover/QoS 评估结果 |
| `AP2.2.1` | 主线覆盖 | B | - | topology adaptation 策略与回放验证 |
| `AP2.3.1` | 主线覆盖 | B | - | frequency allocation 策略与接口抽象 |
| `AP2.3.2` | 主线覆盖 | B | - | xApp 架构评估结果（replay/simulator 级） |
| `AP3.2.1` | 主线覆盖 | B | - | path asymmetry 建模与指标定义 |
| `AP3.2.2` | 主线覆盖 | B | - | multi-link steering baseline |
| `AP3.4.1` | 主线覆盖 | C | - | semantic compression 策略与 gate |
| `AP3.4.2` | 主线覆盖 | C | - | proactive swarm adaptation 策略与 gate |
| `AP5.1.3` | 主线覆盖 | A/B/C（共享） | S2 | 统一 KPI 字典与口径冻结 |
| `AP5.1`（除 `5.1.3`） | 部分覆盖 | 未明确 | S2 | KPI 子项盘点与 owner 分配 |
| `AP5.4.1` | 主线覆盖 | C | - | HIL gate 协议与证据模板 |
| `AP5.4.2` | 主线覆盖 | C | - | field gate 协议与证据模板 |
| `AP1.3 / AP2.1 / AP2.2 / AP2.3 / AP3.2 / AP3.4 / AP5.4` 中未在本表列出的子项 | 未覆盖 | 未明确 | S3 | AP 覆盖补全清单（AP-ID -> 路线/Support/owner） |

### 2.2 Support 子线（用于补齐未覆盖 AP）
Support 子线原则：
- 不替代 A/B/C 主线，不改变主线依赖顺序（A -> B -> C）。
- 以“补位 + 对齐 + 治理”为目标，优先产出可复用规范，而不是先做大规模实现。

#### S1：架构与 SMO 接口支撑线（针对 `AP1.3.3 / AP1.3.4`）
目标：
- 把 O-RAN/SMO 相关要求从“背景描述”提升为可执行约束。
最小交付物：
- O-RAN/SMO 接口约束清单（输入输出、调用时序、时延预算、失败回退）。
- Route B/C 使用的接口兼容检查表。
触发时机：
- 在 Route B 控制策略进入代码实现前冻结 v1。

#### S2：KPI 与评测治理支撑线（针对 `AP5.1` 未覆盖部分）
目标：
- 补齐 `AP5.1.3` 之外的 KPI 体系，避免 A/B/C 指标口径漂移。
最小交付物：
- KPI 字典（定义、统计窗口、聚合方法、通过阈值、责任人）。
- A/B/C KPI 映射矩阵和统一报告模板。
触发时机：
- 在 Route B 完成评审前形成 v1，在 Route C field GO 前冻结。

#### S3：AP 覆盖补全支撑线（针对“未覆盖子项”）
目标：
- 对未进入主线的 AP 子项做系统补全，形成可追踪 backlog。
最小交付物：
- `AP Coverage Backlog`：`AP-ID -> 当前状态 -> 归属（A/B/C/S1/S2/S3）-> owner -> target date`。
- 每周一次覆盖率更新（新增、关闭、延期原因）。
当前基线文件：
- `docs/refs/ops/20260330_dfki_3dnet_roadmap_ap_coverage_backlog_v0_1.md`
- `docs/refs/ops/support_s3_ap13_uncovered_register.md`
触发时机：
- 立即启动，持续到所有目标 AP 至少有明确归属和最小交付物定义。

## 3. 当前基础：哪些已经有，哪些还缺
### 已有基础（可直接复用）
- `OAI + USRP + UAV + 5G modem` 的真实测量平台已经具备。
- 真实 `CSI + telemetry + timestamp` 同步链路已经建立。
- 已完成的分析包括：
  - hardware-inclusive CSI 的物理观察
  - `guided / unguided` ridge 对比
  - `raw / cfo_linear_only / stronger compensation` 对照
  - 旧项目中已有 predictor benchmark（`Naive / GRU / Transformer`）
  - 旧项目中已有 retrospective replay（`BLER / throughput`）
- 现有代码资产可作为路线 A 起点：
  - `src/CPE_analyse.py`
  - `src/raw_vs_compensated_experiment.py`
  - 必要时可参考旧 predictor / replay 代码，但它们不再作为当前主工作流起点

### 当前缺口
- 没有统一的数据协议与 mission-level manifest。
- 没有 measurement-calibrated synthetic channel generator。
- 没有 `synthetic-only / pretrain+fine-tune` 标准实验闭环。
- 没有统一的“预测状态表示”供 AP2/AP3 控制算法复用。
- 没有面向 swarm / HIL 的验证接口定义。

### 当前执行重置（重要）
为避免继续被旧论文项目遗留的 predictor / replay / 图表结果牵着走，当前仓库的实际执行基线重置为：
1. 先只推进 **路线 A / A1：统一真实数据协议**
2. 当前主工作区只围绕：
   - 原始测量数据与 mission 记录
   - `src/CPE_analyse.py`
   - `src/raw_vs_compensated_experiment.py`
   - `docs/refs/data/route_a_data_protocol.md`
3. 旧项目复制来的训练权重、回放结果、论文图表、演示脚本统一视为 **legacy archive**，只在确有需要时回看，不再作为“下一步从哪里开始”的依据。

## 4. 路线 A：数据与信道智能底座（优先实施）
### 目标
把现有 UAV A2G 5G 实测资产升级成可用于 3D-NET 的标准数据底座，并完成测量校准的合成增强与预训练起点。

### 工作内容
#### A1. 统一真实数据协议
固定数据单元：
- `complex CSI`
- `telemetry`
- `timestamp`
- `mission metadata`

固定约束：
- chronological `train/val/test`
- 预处理版本统一：`raw`、`cfo_linear_only`
- realism 辅助标签统一：`guided/unguided Doppler consistency`

当前第一步只做三件事：
1. 盘点现有测量资产，区分 `raw source / derived artifact / legacy output`
2. 冻结最小 mission-level manifest
3. 用 `CPE_analyse.py` 和 `raw_vs_compensated_experiment.py` 验证协议字段能被现有分析流程消费

#### A2. 轻量级测量校准合成器
输入条件：
- 速度
- 径向速度
- 距离
- 飞行阶段
- 采样率
- mission/高度标签

输出对象：
- 与现有预测模型兼容的复数 CSI 时间窗

结构固定三层：
1. 大尺度功率趋势
2. 小尺度时频选择性
3. 硬件层（`CFO/CPE/phase jitter/中心脊偏置`）

固定两类合成器基线：
- `physics-informed analytic + residual noise`
- `measurement-conditioned generative residual model`

#### A3. synthetic -> pretrain -> real fine-tune 入口
固定三种实验臂：
- `real-only`
- `synthetic-only`
- `synthetic pretrain + real fine-tune`

### 交付物
- 数据说明书
- 合成器设计说明
- realism 指标表
- 预训练数据生成脚本与数据版本定义

### 完成标准
synthetic data 进入预训练前必须过：
- 功率分布对齐
- 相位抖动统计对齐
- 时域/频域相关结构对齐
- `guided/unguided` ridge gap 同量级
- mission 间统计差异可由条件变量解释

### 对应论文
**Paper A**：测量校准的合成数据预训练是否提升真实 UAV-5G 信道预测与链路决策

固定章节：
1. 真实测量数据不足与 sim-to-real gap
2. 已有实测观察：hardware-inclusive CSI 与 Doppler 偏差
3. 测量校准的轻量级信道合成器
4. synthetic pretrain + real fine-tune
5. 实验：
   - realism 对齐
   - predictor `NMSE / delta vs naive`
   - `BLER / throughput` replay
6. 讨论：
   - 哪些统计特征必须保留
   - 哪类 synthetic data 最有用

## 5. 路线 B：预测驱动的网络控制与资源编排
### 目标
把路线 A 的预测能力转换为网络控制状态量，用于 mobility、topology、frequency、multi-link 决策。

### 工作内容
#### B0. 统一预测状态表示
作为 AP2/AP3 控制输入，统一输出：
- future CSI quality
- outage / BLER risk
- path asymmetry score
- link degradation trend

#### B1. AP2.1.2 predictive handover / QoS forecasting
指标聚焦：
- `BLER`
- signaling proxy
- handover trigger quality

#### B2. AP2.2 topology adaptation
- relay/path/topology 重配置策略
- 依赖预测状态表示，不直接从原始 CSI 做决策

#### B3. AP2.3 frequency allocation / xApp abstraction
- 先做算法核
- 再做 xApp-compatible 接口抽象
- system-level 验证保留到 replay/simulator 层

#### B4. AP3.2 multipath traffic steering
- 分析链路不对称性
- 基于 `path asymmetry` 做分流、冗余、容错策略

### 默认实现顺序
1. predictive handover
2. topology / frequency control
3. multipath traffic steering

### 交付物
- 统一预测状态接口定义
- replay/simulator 中的控制策略验证结果
- O-RAN xApp 级算法说明书
- 多链路 path steering baseline

### 完成标准
控制验证至少包含：
- handover trigger quality
- outage / retransmission proxy
- path asymmetry handling
- frequency allocation efficiency
- xApp timing feasibility（至少在仿真或 replay 级别）

### 对应论文
- **Paper B1**：预训练驱动的 predictive handover / QoS forecasting
- **Paper B2**：面向 TN/NTN 的 AI-native mobility / topology / frequency / multi-link control

## 6. 路线 C：群体通信与最终验证
### 目标
把前两条路线的算法能力落到 UAV swarm 通信与 HIL/field test 验证。

### 工作内容
#### C1. AP3.4.1 语义数据压缩
- 深衰落下只保留 missions-critical payload
- 建立 fallback 通信机制

#### C2. AP3.4.2 proactive link adaptation for swarm
- 基于 channel / QoS 预测做群体级链路自适应
- 面向 swarm 的整体资源协同，而非单链路优化

#### C3. AP5.4.1 HIL integration
- 把 AP2/AP3 模型迁移到 SDR/HIL 环境
- 提前暴露延迟、接口、硬件约束

#### C4. AP5.4.2 field validation
- 在 UAV swarm 条件下验证语义通信与 proactive link adaptation
- 形成可用于项目汇报与高水平论文的实验素材

### 交付物
- swarm communication 算法说明
- HIL 验证报告
- field-test KPI 报告
- demonstrator 级图表、日志、实验数据说明

### 完成标准
- semantic compression 下的关键任务有效载荷保真
- swarm connectivity continuity
- HIL latency / integration overhead
- field-test `BLER / throughput / mission success proxy`

### 对应论文
- **Paper C1**：semantic-aware resilient communication for UAV swarms
- **Paper C2**：HIL-to-field validation of hardware-aware AI-native swarm communication

## 7. 论文路线图
### 论文优先级（固定）
1. **Paper A**
2. **Paper B1**
3. **Paper C**

### 各论文的角色
- **Paper A**：建立“真实数据 -> 合成增强 -> 预训练 -> real-world gain”的闭环，是后面所有任务的学术入口。
- **Paper B1**：把预测从“看得准”推进到“控制得更好”。
- **Paper C**：展示系统级与群体级的最终工程价值。

## 8. 下一步实施顺序（可直接执行）
### 严格 Mimir 模式（已启用：2026-03-30）
- 严格按依赖执行：`A -> B -> C`，不再使用并行例外。
- 同一时刻仅允许上游主线处于执行态；下游主线必须标记为 `Blocked` 并等待解锁。
- 只有上游主线达到 `Complete` 并完成评审后，才允许切换下游主线为 `Active`。

### 立即开始
- 优先细化 **路线 A**，只做三件事：
  1. 数据协议
  2. 合成器设计
  3. 预训练实验设计
- 同步启动 Support 轻量动作（不与主线抢资源）：
  1. S2：先冻结 KPI 字典草案 v0.1
  2. S3：先完成 AP Coverage Backlog v0.1（补齐未覆盖 AP 子项，见 `docs/refs/ops/20260330_dfki_3dnet_roadmap_ap_coverage_backlog_v0_1.md`）

### 暂不并行启动
- 不立刻进入 AP2.2 / AP2.3 / AP3.2 的完整控制算法实现
- 不立刻进入 AP3.4 / AP5.4 的 swarm / HIL / field 验证
- 不把 `AP1.3.3 / AP1.3.4` 单独拉成第一篇论文主角

## 9. Assumptions / Defaults
- 默认采用 **3 路线划分**，不压缩成 2 条
- 当前唯一真实数据源为现有 UAV A2G 5G SDR 实测数据
- `AP1.3.3 / AP1.3.4` 暂作为路线 B/C 的系统接口背景，不作为第一篇论文主角
- 现有代码资产默认继续复用：
  - `src/CPE_analyse.py`
  - `src/raw_vs_compensated_experiment.py`
  - 现有 replay / predictor / analysis 脚本
- 下一阶段不把精力投入在新的论文排版或 track 选择上，而是先把“合成器 + 预训练闭环”做扎实
- Support 子线默认采用“轻量治理优先”，先文档和约束，后代码实现。
