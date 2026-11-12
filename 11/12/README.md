# 11/12 — public 权重上的四卡 GRPO 深度优化

直接以公开 0.684 集成中的 **Qwen3-Embedding-8B native 五折已训练状态**为初始化和冻结参考，继续优化全部语言侧 rank16 LoRA、八层 LayerNorm、plain/centered 双分类头。默认五折依次训练，每折始终由 **4 张 GPU 协作一个模型**，固定 seed42，不再从随机分类头开始训练。公开集成的其他成员及原有融合权重继续保留。

公开方案、原始数据、模型 revision、tokenizer 与依赖沿用 [11/11](../11/README.md) 和 [资源来源](../11/THIRD_PARTY.md)。`train.py` 中原有监督实现保留为工具与历史对照，默认入口已改为 `grpo_train.py`。

## 训练目标

这是适配 18 类作者识别的 **categorical GRPO**，不生成推理文本，也不将 embedding 模型传给生成式 `GRPOTrainer`。

1. 严格加载对应公开折的 LoRA、LayerNorm、两个分类头；逐文件核验公开 manifest、训练数据、类别顺序、划分与 tokenizer 哈希。
2. 一个全局 batch 最多16条文本，读取每条文本的**全部 native 窗口**，先按有效 token 数汇总两个头的分数，再按公开温度、同 prompt 中心化和双头混合公式生成分类策略。singleton 及组大小边界沿用公开实现。训练与部署使用相同窗口汇总和双头公式。
3. 每条文本从当前策略独立采样16个类别动作，正确奖励1，否则0；每条文本自己的动作构成 GRPO group。使用组内标准化优势、0.2裁剪、每组 rollout 两次优化、相对冻结公开策略的精确 categorical KL。零方差组的策略梯度为0，记录其占比；不人为插入正确类别。
4. 按训练折类别频次加权，对齐比赛 Balanced Accuracy。另加0.1权重的监督 NLL 防止稀疏奖励失去学习信号；默认 KL 权重0.02、学习率2e-6、1 epoch。这些是待验证的预设，不代表已经证明有效。
5. 原生分类状态与冻结参考只保留一个8B骨干；提前缓存公开模型在训练窗口上的两个头分数。四卡汇总窗口输出后计算完整的全局目标、重放本卡样本并 **SUM** 合并梯度，支持显存不足时缩小 micro batch。

GRPO 奖励只用公开 split 的 training_rows；validation_rows 及对应 prompt 均不进入训练。公开 inner-only 校准参数全程冻结，不重新拟合；不读取隐藏测试标签，不根据公榜选折、epoch或奖励。公开权重已有训练历史，因此这些 OOF 结果只报告为继续训练对照，不包装成全新未接触的研究基准。

## 四卡运行

先在有权限的四张卡上准备固定资源和环境：

```bash
# 在仓库 11/12 目录；Python/CUDA 环境见 ../11/README.md
python ../11/scripts/assets.py all
python scripts/run.py --gpus 0,1,2,3 --dry-run
python scripts/run.py --gpus 0,1,2,3 --smoke
python scripts/run.py --gpus 0,1,2,3
```

默认依次处理公开折0、1、2、3、4，每折90分钟的运行窗口。这里的90分钟是保存/暂停上限，**不是缩减 epoch 或提前导出的条件**。未完成保存 `checkpoint.pt` 并退出75；延长可用窗口后运行同一命令续跑。待执行的 rollout、组内动作、旧策略概率、更新位置、优化器和各 rank RNG 一起保存；同一断点须保持相同四卡进程数与实验参数。

卡数固定4，全局文本 batch 不随卡数放大。`--micro-batch` 表示每卡每次计算的窗口数量，默认2。需要缩小验证规模时可在训练前固定 `--folds 0,1`；两折仍为四卡协作，不表示完整五折结果。不同超参数须另用输出目录，不覆盖既有断点。

训练结束立即在四卡做公开 native GPU 验证和导出一致性检查，之后产生 `runs_grpo/decision.json`，记录各折验证、相对公开参考的配对差及 prompt bootstrap 区间。**预先声明的优化折全部进入构建**；负结果仍原样记录，性能诊断不自动替换权重。缺失、变更或技术校验失败的状态会明确停止。

## 构建与提交

```bash
python scripts/build_submission.py --username YOUR_KAGGLE_USERNAME
python scripts/publish.py --directory submission_grpo
# 核对构建后，实际上传、运行并提交一次：
python scripts/publish.py --directory submission_grpo --submit \
  --message '11/12 GRPO: public native continuation, seed42, four-GPU training'
```

输出独立 `submission_grpo/`，Kaggle Notebook 为 `YOUR_KAGGLE_USERNAME/ntl-11-12-grpo`。构建强制挂载优化状态，检查训练/验证/校准指纹及真实权重更新；Notebook 再核对上传状态的 SHA256 与形状。指定部分折时，其他折来自对应公开状态，保持五个不同折；不会重复某一公开折来填槽。

这次实现尚未进行真实8B GRPO训练或新的公榜提交；0.684是公开参考分数，不能当作新方法成绩。Kaggle 隐藏测试仅在冻结模型后由平台推理。

## 本地检查

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_grpo.py
CUDA_VISIBLE_DEVICES='' python tests/test_grpo_distributed.py
CUDA_VISIBLE_DEVICES='' python tests/test_grpo_pipeline.py
CUDA_VISIBLE_DEVICES='' python tests/test_method.py
CUDA_VISIBLE_DEVICES='' python tests/test_distributed.py
CUDA_VISIBLE_DEVICES='' python tests/test_pipeline.py
CUDA_VISIBLE_DEVICES='' python tests/test_selection.py
```

以上使用CPU小模型检查数学、四进程梯度、断点与提交格式，不是8B CPU补测。

算法参考：[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)、[TRL GRPO 说明](https://huggingface.co/docs/trl/v0.22.2/en/grpo_trainer)。精确18类 KL、窗口汇总与双分类头策略是本任务适配；不宣称原始GRPO论文使用此分类结构。
