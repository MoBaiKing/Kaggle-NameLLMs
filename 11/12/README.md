# 11/12

同题成对判别 + native 全文窗口训练。骨干、数据、LoRA 和两阶段训练沿用 [11/11](../11/README.md)。

对同一 prompt、标签不同的回答 $(x_i,y_i),(x_j,y_j)$，令 $s_{i,c}$ 为普通分类头分数：

$$
m_{ij}=\tfrac12(s_{i,y_i}-s_{i,y_j}+s_{j,y_j}-s_{j,y_i}),\qquad
L_{pair}=\frac{\sum_{ij}\sqrt{w_iw_j}\,\operatorname{softplus}(-m_{ij})}{\sum_{ij}\sqrt{w_iw_j}}.
$$

$L=L_{11/11}+0.2L_{pair}$。同 prompt 共享的类别偏移在成对分差中抵消；推理不进行成对指派。

训练窗口严格沿用公开 native 切分，每次按有效文本 token 数抽取一个窗口；验证与提交仍汇总全部窗口。训练固定 seed42，四卡协作一个折，默认顺序训练折0、1。短批次直接跨卡反传，显存不足时使用全局梯度缓存。

```bash
# 环境沿用 11/11；在 11/12 目录执行
python ../11/scripts/assets.py all
python scripts/run.py --gpus 0,1,2,3 --smoke
python scripts/run.py --gpus 0,1,2,3
python scripts/build_submission.py --username YOUR_KAGGLE_USERNAME
python scripts/publish.py --submit
```

固定验证门控比较折0、1 / 折0–3的 OOF 代理；要求收益、prompt bootstrap 区间与独立使用保护均通过，否则保留公开基线。公开框架与资源来源见 [THIRD_PARTY](../11/THIRD_PARTY.md)，成对 logistic 排序参考 [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)。

```bash
# 消融；单独保存结果，不覆盖主方法
python scripts/run.py --lambda-pair 0
python scripts/run.py --sampling first --lambda-pair 0

# CPU 小模型与数值检查
CUDA_VISIBLE_DEVICES='' python tests/test_method.py
CUDA_VISIBLE_DEVICES='' python tests/test_distributed.py
CUDA_VISIBLE_DEVICES='' python tests/test_pipeline.py
CUDA_VISIBLE_DEVICES='' python tests/test_selection.py
```

新方法尚无正式训练/公榜结果；既有0.684提交选择的是公开基线。
