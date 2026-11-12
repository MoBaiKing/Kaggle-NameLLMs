# GRPO v2 — 四卡训练与多强模型软投票

v2 位于独立目录 `11/12-v2/`。继续从公开 Qwen3-Embedding-8B native 五折权重起步，四卡协作一个折，五折依次训练，固定 seed42。v1 的代码、输出和断点保持原样；v2 使用自己的训练、验证、Notebook 和 Kaggle dataset。

这版针对两处实现限制：v1 的16次随机类别采样可能整组奖励相同；v1 的策略目标只看 native 分类器。v2 改用18类全枚举的裁剪期望，并在冻结的多模型 OOF 投票代理上优化。新增显式多模型概率投票，复用公开管线已经计算的强模型结果，不增加这些骨干的前向次数。

当前只有代码、CPU合成小模型及只读公开资产检查，尚未完成真实8B训练或新公榜提交。下面的改动是待验证方案，不能认定已经超过0.684。

## 多模型投票

显式投票成员为公开 `q14v5`（Qwen3-14B）、`q8v5`（Qwen3-8B）、`ll8v5`（Llama-3.1-8B）的既有 LoRA/readout。各成员先独立 softmax，再按3:2:1做概率平均。构建的 Kaggle 推理严格要求三个成员都存在，不能把失败成员静默丢弃。

设完整公开集成概率为 P，三个强模型的显式投票为 V，某一 native 折的校准概率为 N：

```text
parent = 0.75 P + 0.25 V
prediction = 0.60 parent + 0.40 N
```

native 仍沿用公开按折输出、原有 `decide_scores`/Sinkhorn 和折平均。平衡操作之前各分支的概率系数固定如下：

| 分支 | 系数 |
| --- | ---: |
| 完整公开集成（保留其他成员） | 0.450 |
| Qwen3-14B 显式软投票 | 0.075 |
| Qwen3-8B 显式软投票 | 0.050 |
| Llama-3.1-8B 显式软投票 | 0.025 |
| GRPO v2 native | 0.400 |

三个显式成员也参与完整公开集成，因此分支之间有关联。`grpo_vs_same_vote` 对照用完全相同的投票规则、原始公开 native 权重，区分投票变化与 GRPO 训练的贡献。

训练只使用固定 `data/reference/{q14v5,q8v5,ll8v5}.npz`，校验文件哈希、官方训练行顺序和类别排列。OOF 文件只覆盖这三个成员，不能重建完整公开集成：训练时 P 由既有三成员几何 OOF stack 代理。OOF/推理中显式投票的数学计算已做一致性检查；整体代理仍未覆盖其他成员、真实测试分组和 Sinkhorn，验证报告明确标为 proxy，不将其当作隐藏测试成绩。

## 枚举式 categorical GRPO

对每条文本，冻结本轮更新前的投票策略 q_old(c)。奖励 r(c)=1[c=y]，枚举全部18类：

```text
mu = sum_c q_old(c) r(c)
sigma² = sum_c q_old(c) (r(c)-mu)²
A(c) = (r(c)-mu) / max(sigma, 1e-4)
J = sum_c min(q_current(c) A(c),
              clamp(q_current(c), (1-epsilon)q_old(c),
                                   (1+epsilon)q_old(c)) A(c))
```

这是 **全支持、旧策略加权的 GRPO 分类适配**。它用精确奖励矩替代有限采样组的均值/标准差，因此与有限组原始GRPO不同；不能把18类简单等权平均，也不人为插入正确采样动作。概率质量写法避免除以极小的 q_old。饱和策略仍可能有退化方差，保留对应统计与0.1权重的监督 NLL。

损失为 `-J + 0.02 KL(native_current || native_public) + 0.1 NLL(vote_policy, y)`。公开 native 的 KL、两头温度、中心化混合权重和多模型投票系数均冻结。按训练折类别频次加权；默认学习率2e-6、1 epoch、每轮旧策略两次更新。没有 critic、生成式推理或额外种子。

全文 native 窗口先按有效 token 数汇总，再计算两个分类头的公开校准策略。沿用 v1 的四进程全局目标、梯度缓存和 SUM 归约，micro batch 按每卡窗口计数；全局文本 batch 为16。奖励仅使用当前公开折的 training_rows，外折 prompt 不参与训练；不读取隐藏测试标签或用公榜调整权重。

## 运行

环境和固定资产沿用 [11/11](../11/README.md)。在 `11/12-v2/` 目录执行：

```bash
python ../11/scripts/assets.py all
python scripts/run.py --gpus 0,1,2,3 --dry-run
python scripts/run.py --gpus 0,1,2,3 --smoke
python scripts/run.py --gpus 0,1,2,3
```

默认五折，必须四个不同 GPU ID。每折90分钟是 GRPO 工作进程保存/暂停窗口，未完成 epoch 不提前导出；工作进程退出75后启动器停止。同一命令用于恢复。不同配置使用 `--out` 新目录。部署前需保证 GPU 预约覆盖训练及随后的 native 验证；此入口不代替预约管理。

预处理 tokens 可在指纹匹配时复用 v1 的 `11/12/cache/grpo_tokens/`；已完成 v1 的公开 native 参考通过模型/窗口/执行协议、代码与输出哈希核验后可由符号链接复用；v1 学生权重不参与初始化。其余参考、更新前策略、优化器、训练输出及完成记录放在 v2 独立目录。中途恢复保留全部18类旧策略概率与优势，同一断点保持四进程和协议参数不变。

每折训练结束立即 GPU native 验证。`runs_grpo/decision.json` 同时报告 `overall_vs_original`（完整改动相对原公开 OOF 代理）及 `grpo_vs_same_vote`（GRPO相对相同投票的原公开权重），包含配对 prompt bootstrap 区间。预先指定的优化折全部部署；负结果如实记录，技术失败明确停止。

## 构建与提交

```bash
python scripts/build_submission.py --username YOUR_KAGGLE_USERNAME
python scripts/publish.py
# 需要实际上传、运行并提交一次时：
python scripts/publish.py --submit
```

独立 Notebook 为 `YOUR_KAGGLE_USERNAME/ntl-11-12-grpo-v2`，输出 `submission_grpo/`。构建检查 v2 方法身份、真实权重更新和结果指纹；Notebook 校验上传哈希，强制三个强模型投票、五折 native 推理，超时或失败会停止。原公开的其他成员与配置保留。

## 检查

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_objective_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_parent_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_distributed_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_pipeline_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_notebook_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_selection_v2.py
```

这些检查只用CPU小模型/既有OOF数据，不是8B CPU补测。

来源：[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)、[TRL GRPO](https://huggingface.co/docs/trl/en/grpo_trainer)、[公开方案归属与资源](../11/THIRD_PARTY.md)。枚举目标与这次概率投票属于本项目适配，不将已有GRPO或公开大模型作为原创成果。
