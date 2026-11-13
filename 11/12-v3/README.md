# 11/12-v3 — OOF 加权集成感知续训 + 与部署一致的验证门槛

v3 位于独立目录 `11/12-v3/`，不修改 11/11、11/12、11/12-v2 的代码和输出。起点仍是公开 Qwen3-Embedding-8B native 五折权重，seed42，四卡协作一个折，五折依次训练。

## 为什么 v2 掉分（public 0.684 → 0.669）

1. **部署时改了公开融合方式。** 0.684 的 `infer.py` 在 log 空间做几何混合：`rowsoftmax(decide_scores(rowsoftmax(0.6·logP + 0.4·lp)))`；v2 改成概率空间的算术混合 `0.6·P + 0.4·exp(lp)`。用干净 OOF、真实 Sinkhorn 复算，算术混合在各种设置下低 0.1–0.5 BA。
2. **3:2:1 显式投票重复计数。** 这三个成员本来就在 P 里，再占 25% 只是加重它们，代理上无收益。
3. **验证代理和部署不一致。** v2 的代理没有 Sinkhorn，parent 只有 3 个成员、温度 1（真实 `sk_div` 下等效 1.75），而且 original 用几何、候选用算术，+0.67 不可比。
4. **GRPO 信号是训练集内的。** fold-k 公开 native 已在同一批 `training_rows` 上训练 2 个 epoch，策略过度自信；枚举式 GRPO 在 one-hot 奖励下等价于梯度约 ∝ √(q(1−q)) 的加权 CE，真实概率接近 0 的难样本几乎没有梯度。
5. **native 失败时直接报错**，隐藏测试重跑超时会让整次提交失败。

## v3 做了什么

**部署（`build_submission.py`）**：固定的公开 0.684 Notebook 只替换五个 native 折权重文件。`infer.py` 与公开版逐字节相同（构建时断言只有 setup cell 改变）：几何融合、每折 Sinkhorn 后平均、失败时回退公开结果都保持原样；没有显式投票，也不改混合方式。权重文件在 Notebook 内先校验 SHA256 和张量形状，校准沿用公开文件。

**训练目标（`objective_v3.py`、`train_v3.py`）**：

```text
policy  = log_softmax(0.6 · log P_parent + 0.4 · log p_native)          # 与部署的几何规则一致
loss    = Σ_r w_r · CE_ls(policy_r, y_r) / (|batch| · mean_w) + β · KL(p_native ‖ p_native_public)
w_r     = class_balance(y_r) · (ε + (1−ε)(1 − p_oof(r)))                  # ε = 0.1
```

- `p_oof(r)`（`oof_targets.py`，CPU）：作者公开的**折外** native 输出（每行由没见过它的折模型打分）与三成员 LoRA 折外 parent（T=1.75）按部署规则融合（含 Sinkhorn、≤2 的测试式分组、10 次抽样）后，真实类别的概率。所以权重反映的是部署集成在样本外真正答错的行，而不是训练集内的过度自信。
- 用 CE 而不是枚举 GRPO：难样本梯度为 (1−q) 而不是 √(q(1−q))。
- 以全部训练行的平均权重归一化，而不是逐 batch 归一化，容易的 batch 只贡献小梯度。
- LoRA 学习率 1e-5，heads/norms 1e-4，前 10% 线性 warm-up，然后余弦衰减到 0；1 个 epoch，全局 batch 16 条文本，label smoothing 0.05，KL β=0.05，梯度裁剪 1.0。所有窗口按 token 加权聚合，公开校准冻结，dropout 关闭。
- 训练结束同时导出续训权重（α=1）和 **WiSE-FT** 插值 `0.5·公开 + 0.5·续训`（`wise.py`）。部署候选预先固定为 α=0.5，α=1 只做报告。

**验证门槛（`decide_v3.py`、`proxy_v3.py`）**：2,699 条官方训练行，每行用没见过它的折模型打分，按 0.684 Notebook 的规则融合（`decide_scores` 逐字取自固定的 `vendor/infer.py`，测试会比对）。候选与同一规则下的公开 native 配对比较，按 prompt 做 bootstrap。

- 采用顺序（训练前固定）：`v3_wise050` → 方案 A 回退 `v2_wise050` → `v2_full` → `public_only`。
- 通过条件：ΔBA > 0，95% 下界 > −0.30，五折全部通过一致性检查，native 单独 BA 下降不超过 0.50（防止过拟合三成员代理 parent；它不是完整公开集成）。
- 同时报告敏感性（parent T=1、分组 ≤1/≤3、无 Sinkhorn）和逐折 Δ，只作参考。

代理只是官方训练集上的验证，不是隐藏测试，也不是公榜。同一 Notebook 重复提交在 LB 上约有 ±0.009 的波动，0.01 以内的变化无法确认。

## 运行

环境和固定资产沿用 [11/11](../11/README.md)。在 `11/12-v3/` 目录执行：

```bash
python ../11/scripts/assets.py all
python scripts/run.py --gpus 0,1,2,3 --dry-run
python scripts/run.py --gpus 0,1,2,3 --smoke
python scripts/run.py --gpus 0,1,2,3 --v2-runs ../12-v2/runs_grpo
```

- 第一步在 CPU 上生成 `cache/oof_targets.npz`，只读已有 OOF，不跑模型。已存在时自动 `--check`，输入哈希不符就停止。本机生成的 payload SHA256 记录在 `artifacts/verification.json`，服务器重建后可以对照。
- 每折先做 v3 续训，结束后立即四卡 native GPU 验证。`--minutes`（默认 90）是每折的保存/暂停窗口：到时保存断点并以 75 退出，用同一条命令恢复，已完成的折和验证会核验后复用。
- `--v2-runs` 可选：有已完成的 v2 权重时，生成方案 A 的回退候选（不修改 v2 目录）并做 GPU 验证。没有就只评估 v3。
- smoke 用独立的 `runs_v3_smoke/`，fold0、2 个 batch、48 行验证，不能当正式结果。
- 最后 `runs_v3/decision.json` 给出决定和全部对照。

## 构建与提交

```bash
python scripts/build_submission.py --username YOUR_KAGGLE_USERNAME
python scripts/publish.py
python scripts/publish.py --submit   # 只在明确同意后执行一次
```

`decision` 为 `public_only` 时不构建，结论就是保持公开 0.684 Notebook。构建产物：私有 Notebook `YOUR_KAGGLE_USERNAME/ntl-11-12-v3` 和数据集 `YOUR_KAGGLE_USERNAME/ntl-11-12-v3-native`。Kaggle 上 native 推理次数与 0.684 相同（五折），运行时间不变。

## 检查（CPU，只用合成小模型或已有 OOF）

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_objective_v3.py
CUDA_VISIBLE_DEVICES='' python tests/test_proxy_v3.py
CUDA_VISIBLE_DEVICES='' python tests/test_pipeline_v3.py
CUDA_VISIBLE_DEVICES='' python tests/test_decide_v3.py
CUDA_VISIBLE_DEVICES='' python tests/test_notebook_v3.py
```

这些不是 8B 训练或推理。真实四卡 smoke、训练和验证都还没有运行，上面任何内容都不代表已经超过 0.684。

来源：WiSE-FT（Wortsman et al., CVPR 2022, arXiv 2109.01903）、[公开方案归属与资源](../11/THIRD_PARTY.md)。公开 native 权重、校准与 0.684 Notebook 归原作者所有。
