# 11/12-v5 — 从神经元级改写开始

## v5.0 SNR：native 专家输出神经元的兄弟结构改写（已提交 58328855）

native 专家的 plain/centered 两个头都是线性层（18 个输出神经元 × 32768 维特征）。把它们改写为

```text
W' = (I + D) W,   b' = (I + D) b + d,   D 只在对角线和兄弟/易混类别对上非零（families.NEIGHBORS）
```

对任意文本、窗口、折模型，头输出都精确变为 z′ = (I + D) z + d。作者公开的折外头输出就是每行在"没见过它的折模型"下的 z，所以 D、d 可以直接在干净 OOF 上拟合，再写回五个折文件。Kaggle 上公开 native_q8 原样加载，推理代码不改。

- 拟合目标：native 自身经公开校准后的交叉熵（plain/centered 温度和 centered 权重冻结，centered 头用"每 prompt ≤2 篇"的测试式分组），加 ODIR 式 L2 正则（Dirichlet calibration, Kull et al., NeurIPS 2019，这里限制在兄弟结构上）。λ 由 prompt 折 CV 选择，最终 λ=0.003。
- 目标函数刻意不用代理集成，避免利用代理 parent 的失真。

### 嵌套验证（官方训练行，外折 λ 由内折 CV 选，`scripts/evaluate_snr.py` → `artifacts/snr_evidence.json`）

| 变体 | native 单独 BA | native 熵 | 部署 @0.40 | 锚定 @0.75 |
|---|---:|---:|---|---|
| 公开 | 67.17 | 1.20 | — | — |
| **SNR 兄弟结构（提交）** | 66.72 | 1.06 | **+0.22** [−0.30, 0.74] | **+0.33** [−0.22, 0.89] |
| SNR 熵匹配 | 66.65 | 1.17 | 0.00 | 0.00 |
| 只调温度 | 67.13 | 0.97 | +0.33 [0.04, 0.67] | −0.26 [−0.56, 0.04] |
| 只调 bias | 66.94 | 1.20 | +0.04 | −0.04 |
| 对角缩放 | 66.87 | 1.09 | +0.52 [0.11, 0.96] | +0.11 |
| 完整 18×18 | 66.50 | 1.12 | +0.11 | −0.56 [−1.04, −0.15] |

**诚实结论**：把尖锐度匹配回公开水平后，SNR 的提升为 0。它在两个代理上的正值来自让 native 更尖锐，而不是新信息：对 18 个 logit 做线性映射本身不能产生新信息。兄弟结构是唯一在两个代理上都非负的结构，但证据很弱。这次提交可以视为一个公榜读数：在锚定代理预测 +0.3 BA 的前提下，"native 适度变尖锐"在公榜上的真实效果。

### 复现

```bash
CUDA_VISIBLE_DEVICES='' python scripts/evaluate_snr.py      # 嵌套证据
CUDA_VISIBLE_DEVICES='' python scripts/fit_final.py         # 最终改写 + 每折头权重（upload/）
CUDA_VISIBLE_DEVICES='' python tests/test_snr.py            # 权重 = logit 改写；真实公开文件上合并；篡改拒绝；Notebook 最小改动
python scripts/build_submission.py --username YOUR_KAGGLE_USERNAME
```

Notebook：固定 0.684 + setup cell 内合并头权重（公开折文件先对清单校验 SHA256，头文件也校验 SHA256，其余张量逐位不变）+ infer.py 两处 native 跳过改为删除输出并报错（没有公开兜底）。上传只有 5 个头文件（共 24 MB）。

## 下一步（v5.1 起）

logit 级改写已经证明没有信息增量。要真正改写"有问题的神经元"，必须作用在特征上：先在 GPU 上提取 native 的 32768 维多深度特征，再把训练集 prompt 话题方向从两个头的输入中投影掉，W′ = W(I − P)（INLP, Ravfogel et al., ACL 2020；测试 prompt 全部没见过，话题方向是伪相关）。用同样的嵌套方式验证。
