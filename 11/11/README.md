# 11/11

目标是在 **4 张 NVIDIA RTX 5000 Ada** 上协同训练一个模型，固定 **seed 42**，不做多种子重复。
使用 Qwen3-Embedding-8B 的 36 层骨干，读取 3/6/9/12/18/24/30/36 层状态，训练 LoRA、层归一化和 plain/centered 分类头。
训练采用类别先验调整、样本加权交叉熵、Top-5 分类损失、监督对比学习、真实 prompt 组采样与官方数据收尾阶段。

四卡进程共享同一个全局 cell（默认最多16条）。跨卡汇总输出后计算完整的对比学习和 prompt 组统计，再重算本卡样本、合并梯度。
代码不会把全局对比损失拆成四个互不相干的局部损失，也不会把 batch 隐式扩大四倍。
验证按样本分到四卡，复用公开 native 推理代码，并核对导出与公开参考的一致性。

这是从已有方法整理的独立复现实现，新增了四卡协同训练、路径可移植性和提交工具；没有修改原机器上的排队实验。
已通过四CPU进程的梯度一致性、微型36层模型断点续跑/导出，以及三种Notebook组合构建检查。真实四卡吞吐量与正式公榜结果需要运行后报告，不能预先声称达到公开 Notebook 标题中的分数。

## 1. 环境

训练使用 Python 3.10/3.11 和 CUDA 12.1。推荐先建立独立环境：

```bash
git clone https://github.com/MoBaiKing/Kaggle-NameLLMs.git
cd Kaggle-NameLLMs/11/11
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Kaggle CLI 2.2.5 要求 Python >=3.11，可单独安装，让训练环境通过 PATH 调用：

```bash
python3.12 -m venv .kaggle
.kaggle/bin/python -m pip install kaggle==2.2.5
export PATH="$PWD/.kaggle/bin:$PATH"
```

按 Kaggle 官方方式配置你自己的凭据，并先加入比赛、接受比赛规则。不要把 token、`kaggle.json` 或个人环境上传到 Git。

## 2. 数据和模型

仓库直接提供：

- `data/original/train.csv`：原始 2,699 条训练记录，18 类，保留原始文本和顺序。
- `data/original/test.csv`：比赛提供的可见测试示例，**不是完整隐藏测试集**。
- `data/original/sample_submission.parquet`：原始提交格式示例。
- `data/folds.json`：固定的五折 prompt 划分。
- `data/reference/*.npz`：三个公开 LoRA 成员在官方训练记录上的参考预测，用于原方法的 OOF 融合验证。不是隐藏测试预测。

下载并校验增量训练流、预训练模型、公开 native 参考状态和固定 Notebook：

```bash
python scripts/assets.py all
```

也可以分别运行：

```bash
python scripts/assets.py verify
python scripts/assets.py augmentation
python scripts/assets.py backbone
python scripts/assets.py native
python scripts/assets.py notebook
python scripts/assets.py stage
```

骨干为 `Qwen/Qwen3-Embedding-8B`，固定 revision：

```text
1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
```

等价 Kaggle 模型来源是 `qwen-lm/qwen-3-embedding/Transformers/8b/1`。
`stage` 将公开 native 资产中的 tokenizer/config 与骨干权重组成 `base_staged`，保证空输入追加 token 151643；直接使用另一份 tokenizer 会改变输入协议。

增量数据下载自本仓库 Release，约77 MB，153,770条，SHA256：

```text
8a456ad6ce571bad412c178527210173ea727b6fe4584c7971be6955bdcc41bd
```

直接用 GitHub CLI 下载也可以：

```bash
gh release download '11/11' --repo MoBaiKing/Kaggle-NameLLMs \
  --pattern stream_8b_cap9500.parquet --dir data
python scripts/assets.py augmentation
```

该流来自 `crtproduction/name-that-llm-open-pool` 的固定212,623行快照；移除官方文本、固定保留集、近官方 prompt 与再生成记录，按类最多9,500条进行加权选择。
构建种子为20261110，数据选择种子不是额外模型训练种子。来源列表见 `data/augmentation_sources.csv`。
若需要从原始增量池重建：

```bash
python scripts/build_augmentation.py --tag 8b_cap9500 --cap 9500 \
  --pool /path/to/exact/ntl_open_pool.parquet
```

该脚本已在原快照上重建并得到相同文件哈希。最新版 Open Pool 可能已经改变，不能直接替代。
找不到固定资产时，使用 [复现 Agent 指令](REPRODUCE_AGENT.md) 的增量数据搜寻部分；它要求按哈希定位历史快照。

## 3. 四卡、单种子训练

先检查设备和使用权限，再做一次短预检：

```bash
nvidia-smi
python scripts/run.py --gpus 0,1,2,3 --smoke
```

快速、可进入原融合门控的流程：

```bash
python scripts/run.py --gpus 0,1,2,3 --folds 0,1 \
  --pool-minutes 90 --finish-minutes 45
```

这两个交叉验证折都使用 seed 42；每次四卡协同训练一个折，折之间顺序执行。
原来的融合判定至少需要两个通过验证的折，因此快速入口默认两个折。
如果只检查一个模型，可用 `--folds 0`；该检查不能满足原融合门控，不能冒充完整复现。

完整原四折覆盖用：

```bash
python scripts/run.py --gpus 0,1,2,3 --folds 0,1,2,3 \
  --pool-minutes 120 --finish-minutes 60
```

时间参数是每折预算，不是承诺完成时间。不同硬件和时间预算产生的更新次数不同，不能保证逐位复现。
默认省去中途反复验证，只在训练完成后做必要验证；单独调用 `train.py --monitor` 可保留中途监测。
遇到中断或收尾阶段预算耗尽会保留断点，退出码75；延长可用时间后重复同一命令续跑，不导出未完成模型。

`run.py` 内部使用：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train.py ... --seed 42
```

训练目录 `runs/foldN` 保存导出状态；每折训练结束即进行 native 验证和内层校准，最后运行 `decide.py`。
外层验证行不参与训练或内层温度拟合。部署组合只按验证数据选择，不根据 Kaggle 分数再选组合。

## 4. 构建和提交

验证结束会生成 `runs/decision.json`，选择 `ours2`、`ours4` 或 `public_only`。
收益需满足预先定义的 BA、prompt-bootstrap 区间和独立 guard；不满足时使用公开基线，并如实报告。

```bash
python scripts/build_submission.py --username YOUR_KAGGLE_USERNAME
python scripts/publish.py
```

第二条只显示将要提交的配置。要实际上传状态、运行 Notebook 并提交一次：

```bash
python scripts/publish.py --submit
kaggle competitions submissions name-that-llm
```

Notebook 运行在 Kaggle 支持的 GPU 环境，保持互联网关闭，读取平台替换后的隐藏测试数据，输出 `submission.parquet`。
本地四张 RTX 5000 Ada 用于训练；不用为 Kaggle Notebook 假设四卡硬件。
提交工具检查构建成功、输出结构及 native 状态挂载标记；技术检查失败会停止。
`submission/receipt.json` 记录提交编号；响应不确定时不会自动重试，先查历史，避免消耗重复提交额度。

## 5. 本地代码检查

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_distributed.py
CUDA_VISIBLE_DEVICES='' python tests/test_pipeline.py
python scripts/run.py --dry-run --smoke
```

这些是CPU小模型/格式检查，不是完整8B训练，也不是比赛成绩。更多来源与许可信息见 [THIRD_PARTY.md](THIRD_PARTY.md)。
