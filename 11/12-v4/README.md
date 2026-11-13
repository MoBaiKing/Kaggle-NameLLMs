# 11/12-v4 — A1 + 公开 0.684 集成 + A2 + B

v4 位于独立目录 `11/12-v4/`，不改 11/11、11/12、v2、v3 的代码和输出。seed42，四卡协作。每次提交都包含我们自己的增量，**没有公开兜底**。

```text
           ┌──────────── 公开 0.684 集成（LoRA 成员、comp、GSQ、融合、Sinkhorn 全部不变）────────────┐
 A1 ──► native 槽位（40%）                                                                     │
           └───────────────────────────────────── 最终概率 ─────────────────────────────────┘
                                                      │ 前 4 个候选（96% 召回）
                                    A2：类条件似然证据 ──► B：兄弟感知门控重排 ──► 提交
```

## 依据（都在 `analysis/` 里可复现，只读已保存的 OOF）

- **错误结构**（`error_structure.py`）：0.684 规则在干净 OOF 上 top-1 66.1%，top-2 84.2%，top-3 91.9%，top-5 98.3%。61% 的错误发生在同家族兄弟版本之间：gemini 3.6/3.7/3.8-flash（gemini 错误里 179/183 在家族内）、gpt-5.6 三兄弟、glm-5.3 与 flash；另有跨家族的 claude-fable-5 ↔ glm-5.3（约 90 次）。
- **代理必须锚定公榜**（`proxy_anchor.py`、`b_without_a2.py`）：三成员 LoRA OOF parent 相对 native 比真实完整集成弱得多。代理上 native 占比越高分越高（平台 0.70–0.825），公榜上 0.40 才是最优（0.30、0.50 都更低）。早期带 native 特征的 B 在 0.40 代理上看起来 +2.63 BA，锚定后只剩 −0.30，是纯代理假象。因此：
  - v4 的训练、验证、拟合都在**锚定代理**上进行：代理 native 占比 0.75，即代理自身最优平台的中心，对应部署时的 0.40。这一点在任何 v4 训练之前固定（`scripts/proxy_v4.py`）。部署不变，仍是 0.40。
  - B 没有 native 特征，只能利用集成本身没有的新证据。B 不含 A2 时在两个代理上都约等于 0（−0.07 / +0.07），说明 B 的价值完全取决于 A2。
- **公开实验的反面证据**：同族门控、加更多 LoRA 成员、Hungarian/prompt 分配都已被公开笔记本证明在公榜上无效。v4 不做这些，也不用强化学习。

## 三个组件

**A1 — HSC-Native（Kaggle 上零额外时间）**：`a1_train.py`、`a1_objective.py`

沿用 v3 引擎，但目标在锚定代理上：OOF 加权集成感知 CE + KL，权重来自 `oof_targets_v4.py`。另外加入：
- **兄弟结构回放**：外部 prompt-clean 数据按"混淆组"（一个家族或一对固定的跨家族易混组合）组成 cell，与官方 cell 1:1 交替。
- **分级 SupCon**：在 native 投影上做层级监督对比，同模型权重 1，同家族或易混对权重 0.25（多层对比思路来自 DeTeCtive, NeurIPS 2024）。
- **兄弟 margin**：真实类别要比兄弟和易混类别的 log-sum-exp 高出 1 nat。

部署用 WiSE α=0.5，公开校准冻结。

**A2 — CCL-Gen 对比式类条件似然专家（新专家类型）**：`ccl_gen.py`、`a2_train.py`、`a2_score.py`
- 结构：冻结的 Qwen3-1.7B-Base（HF `ea980cb`，与 Kaggle 模型 `qwen-3/1.7b-base` 同一份权重），加一个共享 LoRA（rank 16，7 类投影全覆盖）和每类 8 个可学习前缀向量。
- 打分：类别 y 的分数 S(x,y) = Σ log p(x_t | prefix_y, x_<t)，是生成式分类器（ATLAS, arXiv 2610.19583）。
- 训练：生成式 NLL，加上候选集（真类 + 兄弟/易混 + 随机）上的判别式 CE，把似然质量往兄弟版本之间推开。这是相对 ATLAS 的原创改动。
- 只在外部 prompt-clean 数据上训练（56,866 条，每类 ≤3,200）。剔除了与官方 150 个 prompt 话题相近的 2,163 条，所以 2,699 条官方行都是 A2 从没见过的干净验证集。
- Kaggle 上只对门控行的 top-4 候选打分，双 T4 fp16。

**B — SAGE 兄弟感知门控重排（CPU，数秒）**：`sage.py`
- 门控：top-2 是兄弟或易混，或者 margin 处于该数据集下 50% 分位时才重排，否则保留 top-1。
- 特征（全部是行内相对值）：集成相对 log 概率（按当前数据集做 z 标准化，消除代理与真实集成锐度不同的影响）、排名、长度条件化的排版似然（只用外部数据估计）、是否兄弟、A2 证据（Σlog p 差 / √tokens）及其与"兄弟"的交互。
- 模型：条件 logit，L2 正则；C 由官方 prompt 折做嵌套交叉验证选择。
- 没拿到 A2 分数的行改用无 A2 的权重。同一份 `sage.py` 既用于拟合，也原样嵌入 Kaggle，保证一致。

## 验证（`decide_v4.py`，CPU）

在 2,699 条官方行、锚定代理上逐级比较：
- S0：公开 native，仅作参照，从不提交
- S1：+A1
- S2：S1 + B（无 A2）
- S3：S1 + B（含 A2）= v4 完整版

每级给出配对 prompt-bootstrap 95% 区间和逐折诊断。提交候选永远是 v4 完整版，验证数字作为证据附上；明显有害（95% 上界 < 0）时会提示，由你决定是否提交。这些都是官方训练集上的验证代理，不是隐藏测试，也不是公榜。

## 运行（服务器，4 张 GPU）

环境和固定资产沿用 [11/11](../11/README.md)。在 `11/12-v4/` 下执行：

```bash
python ../11/scripts/assets.py all
python scripts/assets_v4.py
python scripts/run.py --dry-run
python scripts/run.py --smoke
python scripts/run.py
```

- `assets_v4.py` 下载并校验 A2 底座（3.4 GB，SHA256 固定）。
- `run.py` 依次执行：外部数据和锚定 OOF 权重（CPU）→ A1 五折（每折训练后立即做四卡 native 验证）→ A2 训练 → A2 对官方行全 18 类打分 → B 嵌套 CV、证据和最终 bundle。
- 退出码 75 表示窗口到期并已存断点，用同一条命令恢复。可以用 `--stages a2` 等只跑部分阶段。
- 粗略耗时（4×RTX 5000 Ada）：A1 每折约为 v3 的 2 倍（默认窗口 150 分钟）；A2 约 34M token 的前后向（默认窗口 360 分钟，预计 1 小时左右）；A2 打分为 2,699 行 × 18 类 ≈ 4.9 万条序列的前向，几分钟。

## 构建与提交

```bash
python scripts/build_submission.py --username YOUR_KAGGLE_USERNAME
python scripts/publish.py
python scripts/publish.py --submit   # 只在明确同意后执行一次
```

- **构建器断言**：Notebook 相对固定的 0.684 只有以下改动，其余一律不变：
  - setup cell（A1 权重、A2/B bundle 哈希、A2 底座重新算哈希）
  - 两个新 cell（`dyl_ccl.py`、`dyl_sage.py`）
  - `infer.py` 的三处锚点：native 成功后调用 B；native 跳过或失败时删除输出并报错
- **B 阶段失败**：保留已写好的 A1 决策（那也是我们的增量）。
- **Kaggle 额外时间**：A2 约 4,000×0.85×4 个序列（中位约 130 token），双 T4 估计 5–15 分钟，在 native 截止（8 h）之后、9 h 上限之内完成，截止时间为 native 截止 + 30 分钟。超时的行改用无 A2 的 B。

## 检查（CPU，只用合成小模型或已有 OOF）

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_a1_v4.py
CUDA_VISIBLE_DEVICES='' python tests/test_a2_v4.py
CUDA_VISIBLE_DEVICES='' python tests/test_sage_v4.py
CUDA_VISIBLE_DEVICES='' python tests/test_decide_v4.py
```

真实的 A1/A2 训练和打分都还没有运行，以上任何内容都不代表已经提分。

来源：ATLAS（arXiv 2610.19583）、DeTeCtive（NeurIPS 2024）、Idiosyncrasies in LLMs（Sun et al., 2025）、WiSE-FT（CVPR 2022）、[公开方案归属](../11/THIRD_PARTY.md)。公开 native 权重、0.684 Notebook 和 Open Pool 数据归原作者所有。
