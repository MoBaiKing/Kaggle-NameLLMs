# 11/12-v6 — 按 LLM 赛常见金牌套路：数据为王 + 成熟配方 + 种子汤 + 对齐公榜的验证

## 为什么要换方向

v2–v5 都在**同一个公开 native 专家、同一批信息**上做改动（续训、头重写、加弱证据），公榜一直是 0.669–0.680，没有超过公开版的 0.684。能抬分的只有集成里还没有的新信息。LLM 赛金牌方案最一致的规律有三条：

1. **数据为王**：在更多、更干净、分布对齐的外部数据上训练。本赛论坛也一致：单模型 0.65–0.68 的增益几乎都来自数据。
2. **复用经过验证的强配方，不在架构上折腾**：换数据，保持已在公榜验证过的模型和训练流程。
3. **集成稳定性**：多种子训练后做权重平均（model soup），推理成本不变，方差更小。

外加一条工程纪律：**验证指标必须能预测公榜方向**。v4 和 v5.0 的结果表明，"native 单独的折外 BA"两次都对；"三成员集成代理"一次都没对。

## v6 做什么

把 0.684 集成里占比最大的 native 专家（40%）**用同一个结构、同一套公开配方重新训练，只换成更好的数据**，再做种子汤。Kaggle 推理代码、运行时间和显存都不变，只替换 5 个折权重和它们的校准文件。

### 1. 数据（`scripts/data_v6.py`，CPU）

数据源：Open Pool **v15**（Kaggle `crtproduction/name-that-llm-open-pool`，2026-11-13 版，226,191 条，SHA256 固定）。公开 native 当时只用了 19.6 万条旧版数据。

规则在训练前固定，每条都有数据集作者的实测依据：

| 规则 | 依据（作者在真实 prompt 上的对照实验） |
|---|---|
| 采样权重 = `match_weight`（排版、长度与真实文本对齐），截断到 [0.05, 10] | 同等数据量 +0.026 BA |
| kimi-k3 / glm-5.3 / qwen3.8-max 按生成设置的保真度降权（相对该类最好设置，下限 0.3） | kimi-k3 推理开启时识别率 25%，关闭时 8% |
| 去掉 `max_sim_train_prompt ≥ 0.25` 的行（6,560 条） | 保证官方 OOF 门槛干净 |
| 训练时类别均衡采样，类内按权重 | 公开配方 `per_update_class_sequence_matched` |

结果：保留 219,631 条，payload SHA256 `ce7f0e08…`。

### 2. 训练（`scripts/train_v6.py`，4 卡 torchrun）

结构 = 公开 native（Qwen3-Embedding-8B，8 个深度，LoRA r16/α32 覆盖全部投影，plain + centered 两个头）。超参照抄公开 `study.json`：

| 阶段 | 内容 |
|---|---|
| **pretrain**（五折共享） | 只用外部数据，所以不会把任何一折的官方标签带进别的折。从零初始化 LoRA/头；全局 batch 64；adapter lr 2e-4，head lr 5e-4；label smoothing 0.05；centered 系数 0；10% 预热 + 余弦；默认 40 万次样本（公开 39.2 万） |
| **finetune**（每折 × 3 个种子） | 从 pretrain 开始；训练行 = 本折训练集去掉 18 个校准 prompt；每个 cell 4 个完整 prompt 组；2 个 epoch；adapter lr 5e-5，head lr 5e-4；centered 系数 0.5 |
| **soup**（每折） | 3 个种子权重取平均（同一 pretrain 初始化，均值合法）→ 导出公开格式 → 11/12 `evaluate.py`：在公开 native_q8 运行时上做一致性检查，并在校准行上拟合公开校准字段 |

每个阶段都存断点，到 `--deadline` 以退出码 75 暂停，用同一条命令恢复。

### 3. 门槛（`scripts/decide_v6.py`，CPU）

在全部 2,699 条官方行上（每行由没见过它的折模型打分），比较 **native 单独 BA**：我们的 vs 公开 native（作者的 OOF）。prompt 级配对 bootstrap，按以下规则判定：
- 每组 ≤2 篇（模拟隐藏测试）：Δ > 0 且 95% 下界 > 0
- 单篇：Δ ≥ 0
- 五折一致性检查全部通过

全部满足才得到 `decision = v6`，构建器才会构建，否则 `not_ready`。没有公开兜底：native 没跑成时，Notebook 删除输出并报错。

### 预期与风险

- 这一版押的是数据。作者实测数据改进 +0.02–0.03 BA，但那是在 1.7B 小模型上；对 8B native 是否同样成立，要看门槛结果。
- pretrain 约 40 万次 8B 前后向，4 卡粗估 6–12 小时，可以跨多个窗口续跑。每折精调 × 3 种子约 10–20 分钟。

## 运行

环境沿用 [11/11](../11/README.md)。在 `11/12-v6/` 下执行：

```bash
python ../11/scripts/assets.py all
mkdir -p ../11/external/open_pool_v15
kaggle datasets download crtproduction/name-that-llm-open-pool -f ntl_open_pool.parquet -p ../11/external/open_pool_v15 --unzip
kaggle datasets download crtproduction/name-that-llm-open-pool -f fidelity_by_reasoning.csv -p ../11/external/open_pool_v15 --unzip
python scripts/run.py --dry-run
python scripts/run.py --smoke
python scripts/run.py --minutes 600
```

`data_v6.py` 会校验 v15 文件的 SHA256（`5a3dc822…`）。如果 Kaggle 上的数据集又更新了，SHA 会对不上并停止；这时要指定旧版本下载，或者重新固定新版本。

## 构建与提交

```bash
python scripts/build_submission.py --username YOUR_KAGGLE_USERNAME
python scripts/publish.py
python scripts/publish.py --submit
```

- 构建器只接受门槛通过的 decision，并断言 Notebook 相对固定的 0.684 只改了 setup cell 和 `infer.py` 两行。
- 上传约 900 MB（5 个 180 MB 的折权重 + 5 个校准文件）。
- `--submit` 会消耗一次提交。

## 检查（CPU，只用合成小模型或已有 OOF）

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_data_v6.py
CUDA_VISIBLE_DEVICES='' python tests/test_train_v6.py
CUDA_VISIBLE_DEVICES='' python tests/test_decide_v6.py
```

真实 8B 训练还没有运行，这里不代表已经提分。

来源：Open Pool v15 数据集说明（crtproduction，CC BY-SA 4.0）；公开 native 配方与权重（CHEN Han `study.json`）；Model soups（Wortsman et al., ICML 2022）。
