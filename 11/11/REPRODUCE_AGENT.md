# 可直接交给其他 AI 的复现指令

请在获得授权的机器上完整复现 https://github.com/MoBaiKing/Kaggle-NameLLMs 的 `11/11` 方法。
优先快速完成训练和有效提交，只用 **seed 42**，不做任何多种子搜索。硬件是 **4张 NVIDIA RTX 5000 Ada**，四卡协同训练同一个模型，不能将四个独立种子误当成四卡并行。

执行要求：

1. 克隆仓库，进入 `11/11`，先读 README。原始 train/test示例/sample_submission 和固定 folds 已在 `data/`，先运行 `python scripts/assets.py verify`，保持文本、标签、行顺序和 prompt 划分不变。隐藏测试数据只在 Kaggle 正式评分时可见。
2. 用独立环境安装 CUDA 12.1 对应的 torch 2.2.1 和 requirements.txt。Kaggle CLI 2.2.5 使用 Python>=3.11 的单独环境；凭据仅在本机配置，不写进代码、日志或仓库。先确认账户已接受比赛规则。
3. 运行 `python scripts/assets.py all`。预训练骨干是 `Qwen/Qwen3-Embedding-8B`，revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`，也对应 Kaggle 模型 `qwen-lm/qwen-3-embedding/Transformers/8b/1`。下载公开 native 折状态、校准、OOF、tokenizer，并运行 stage。必须使用 native 资产的 tokenizer/config 与该骨干权重组合，不自行改模板或末尾 token。
4. 增量训练数据优先下载本仓库 Release：`gh release download '11/11' --repo MoBaiKing/Kaggle-NameLLMs --pattern stream_8b_cap9500.parquet --dir data`。核对153,770条和 SHA256 `8a456ad6ce571bad412c178527210173ea727b6fe4584c7971be6955bdcc41bd`。不要为了省事改用一个不同的数据快照。
5. 确认GPU使用权限，运行 `python scripts/run.py --gpus 0,1,2,3 --smoke`。通过后优先快速流程：`python scripts/run.py --gpus 0,1,2,3 --folds 0,1 --pool-minutes 90 --finish-minutes 45`。每折使用四卡，两个折共用seed42，交叉验证折不等于不同随机种子。需要完整原四折覆盖时使用 `--folds 0,1,2,3`；只跑 `--folds 0` 是单模型检查，达不到原融合门控的两折要求。
6. 训练中断时使用原目录续跑。训练结束立即完成 native 验证、内层校准与导出一致性检查，再运行固定验证门控。不得用隐藏测试或公榜选择 checkpoint、阈值、折组合或随机种子。门控不通过时保留 public_only，并如实说明。
7. 用 `python scripts/build_submission.py --username <本人的Kaggle用户名>` 构建 Notebook；检查 `runs/decision.json`、模型状态、Notebook和附带资源。运行 `python scripts/publish.py` 预览，再按本条任务授权执行 `python scripts/publish.py --submit` 上传、运行和提交一次。状态记录在 `submission/receipt.json`；若提交响应不确定，先查询 `kaggle competitions submissions name-that-llm`，禁止盲目重试。
8. 报告实际训练和验证状态、最终部署组合、提交编号、评分状态和真实公榜分数。分清公开 Notebook 的历史标题分数、当前验证结果和本次提交成绩，不把预检当成训练完成。保留运行命令、数据哈希、checkpoint和冻结记录。

## 固定增量数据或上游资产找不到时，使用下面的搜寻任务

请检索 Kaggle 官方数据页、版本历史、公开 Notebook 附件及作者说明，优先找回
`crtproduction/name-that-llm-open-pool` 的**212,623行历史快照**，文件 `ntl_open_pool.parquet`。
原始快照精确 SHA256 和字节数在 `config/assets.json -> augmentation`；确认后用
`python scripts/build_augmentation.py --tag 8b_cap9500 --cap 9500 --pool <文件路径>` 重建。
最终输出必须具有153,770行、28,898个prompt、相同标签顺序和上述训练流 SHA256。
该筛选包含官方文本排除、固定保留集排除、近官方prompt与再生成标记排除、按类上限9,500、选择种子20261110、权重截断[0.1,3.0]。
不要省略排除步骤，也不要重新随机划分官方数据。

若 native 资产或 Notebook 内容发生变化，定位
`lightningv08/ntl-olmo2-native24-final-inference-assets` 中
`ntl-native-q8-multidepth-fivefold-0679/` 的匹配版本，以及
`lbjyang/ntl-684-ozertuu-mirror` 的对应历史 Notebook。逐文件匹配 `config/assets.json` 的哈希。
给出找到的官方URL、作者、版本号、许可、下载命令、文件哈希和校验结果。
无法找到精确版本时，明确报告无法精确复现；如仅找到新版，另存为扩展实验，不覆盖本次冻结复现，也不伪造一致的哈希或成绩。
