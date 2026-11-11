# Kaggle NameLLMs — 11/11

可复现代码、原始比赛数据和固定划分位于 **[11/11](11/11/README.md)**。

- 一个固定训练种子：**42**。
- **4 张 NVIDIA RTX 5000 Ada 协同训练同一个模型**，使用 `torchrun`；交叉验证折依次执行。
- 原始 `train.csv`、可见的 `test.csv` 示例、`sample_submission.parquet` 已直接包含在仓库中。
- 153,770 条已选增量数据通过 [11/11 Release](https://github.com/MoBaiKing/Kaggle-NameLLMs/releases/tag/11%2F11) 下载；预训练模型按固定 revision 下载。
- [完整复现与提交流程](11/11/README.md)，[可直接交给其他 AI 的复现指令](11/11/REPRODUCE_AGENT.md)。

本仓库发布方法和复现工具，不宣称已完成该四卡实现的正式训练或获得某个公榜分数。
