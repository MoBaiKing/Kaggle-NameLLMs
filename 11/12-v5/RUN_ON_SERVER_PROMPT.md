请实际运行 Kaggle Name That LLM 的 v5.1：先预检、CPU 测试、真实四卡 smoke，再按协议完成训练、GPU 打分、CPU 验证。只做 EB 提示中心化与三组 8B 判别式专家；不等待 v5.0 公榜分，不重启旧 A1/A2、native 续训、SNR、生成式专家或强化学习。

仓库：https://github.com/MoBaiKing/Kaggle-NameLLMs
分支：codex/v5.1-eb-sibling-experts
方法目录：11/12-v5/
先记录 git rev-parse HEAD 完整 SHA，读取目标服务器适用的 AGENTS.md、11/12-v5/V51.md 与 artifacts/v51_plan.json。只使用本次拉取的固定 SHA；不要混用 v3-oof-ensemble-ce 分支上的旧运行入口。保留真实日志和失败状态，处理可恢复错误。

1. 核对 GPU 授权、Python/CUDA/驱动、磁盘空间、四张卡的实际空闲显存。用户指定队友 4×RTX 5000 Ada 或 #510（2026-11-15 北京时间 04:00–14:00）。时间文字不是授权凭据：有预约系统时持续读取本人最新有效预约，适配新增/改期/缩短/取消和 GPU 范围变化。本工作区最新 AGENTS 指明预约本人为廖子杰，勿拿邓郁霖的预约代替。不要修改、缩短或释放预约记录；进程按实际有效窗口保存退出。没有可用授权卡就动态等待，不用 CPU 跑 8B 前向或训练。

2. 有 FocusForge 时，运行前读 /data/dyl/focusforge/README.md，用 /data/dyl/focusforge/focusforge_task_feed.py 按当地契约登记稳定任务 ID kaggle_name_llm_v51_seed42，以 runs_v51/status.json 为原始状态源并提供必要的状态适配。持续更新相同源；单个阶段训练完成仅 TRAINED，GPU 验证完成才 VALIDATED；本地官方 OOF 不冒称隐藏测试 TESTED。没有该系统的队友服务器保留同等 JSON 状态和日志。不要启动/修改 name-that-llm-jnu 里与本次无关的 v 系列队列。

3. 按 ../11/README.md 准备环境，依次准备固定公开资产（能复用的先校验再复用）：

```bash
cd 11/12-v5
python ../11/scripts/assets.py native
python ../11/scripts/assets.py notebook
python ../11/scripts/assets.py backbone
python ../11/scripts/assets.py stage
CUDA_VISIBLE_DEVICES='' python scripts/v51_data.py --download
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_v51_*.py' -v
python scripts/run_v51.py --dry-run
```

Open Pool 固定 v15 的两份文件 SHA 在计划内，不使用旧 capped9500 流。查阅更新/新来源及 cache/external_v51.json，报告 gemini3.7 清洗后条数。未核验的 10,838 行先隔离：用 scripts/v51_screen_infer.py 的 native8b/q14v5/ll8v5 三个入口真实 GPU 打分；参照 artifacts/screen_*.example.json，从固定公开 Notebook setup 定位、下载/核验原始底座和五折 adapter，填入本地绝对路径 spec。每条打分传实际 --deadline，在共享服务器上由实时租约监护包裹；可复用哈希核验通过的已有分数。按 V51.md 用 v51_screen.py 合并后重新构建筛选版外部数据，再启动正式训练。找不到相应公开资产或本窗口无法完成时明确记录继续隔离，不能用同一模型三个折伪造三票。任何数据更新必须在正式训练前冻结，正式开始后不要覆盖本次源文件。不得使用测试标签，不调用付费 API。

4. 预约服务器先实现实时租约查询命令，把 argv 数组存为 lease_argv.json。它必须核对最新本人预约和实际 GPU 占用，输出 allowed、gpus、带时区 deadline；gpus 包含当前空闲或本启动器已占用的授权卡，排除他人占用的卡。预约未开始或无效时 allowed=false。然后执行：

```bash
python scripts/run_v51.py --smoke --lease-command /ABSOLUTE/PATH/lease_argv.json
python scripts/run_v51.py --lease-command /ABSOLUTE/PATH/lease_argv.json
```

队友明确独占且没有预约系统时可替换为 --gpus 0,1,2,3 --deadline 带时区的实际窗口结束时间；#510 示例结束时间为 2026-11-15T14:00:00+08:00，不能提前到 04:00 前上卡。smoke 不通过不跑正式。布局为三组专家各占一卡、EB 前向占第四卡，动态补位；一个模型不会 torchrun 占四卡。未完成阶段优先续跑，已完成 GPU 打分经哈希核验后复用。预留真实保存/释放时间，时间不够就断点排队，不删断点、不降低训练量来冒称完成。

5. 检查每组 external、五个单折排除与十个双折排除阶段。固定最后 checkpoint，不用排除折指标选择 checkpoint。训练结束立即同卡打分；已训练但未打分时优先补 GPU 打分，不能标成实验已完成。双折排除只服务 SAGE 外折训练，部署只保留五个单折模型。确认所有 scores.npz、scores.json、complete.json 与 checkpoint 哈希、排除折、ID 和类别一致。EB collector 的 topic/五折分块可以恢复，至少覆盖一千个外部面板。

6. 正式完成后启动器自动运行 CPU 门槛；也可显式执行 python scripts/v51_evaluate.py。读取 runs_v51/evidence/decision.json，报告 native-alone sparse1–2 BA、EB 配对 prompt-bootstrap CI、每组专家/原 native 组内准确率、SAGE 家族内准确率 CI、组合 CI、逐次分组指标和 native 数值核验。区分本地 OOF 验证、GPU smoke、隐藏测试、公榜。说明 native 单独验证与完整集成部署的分布差异以及多组件筛选的统计限制。真实训练未完成则只报进度，不能拿合成测试代表提升。

7. 至少一个组件过门槛才构建，只携带通过组件。两个都失败就不构建、不提交，保留负结果；若需要自行生成新数据，只整理方案与预算供用户决定，不自行消耗 API 预算。过门槛后用真实 Kaggle 用户名执行：

```bash
python scripts/build_v51.py --username 实际账号名
python scripts/publish_v51.py
```

本提示词沿用用户的条件授权：通过上述门槛后可以执行一次 python scripts/publish_v51.py --submit；两个都不过绝不提交。若服务器当前会话有更新的提交限制，则遵从最新指令，不重复索要已给出的授权。提交响应不明确就核对 receipt.json 和 Kaggle 历史，不能盲目重试。v5.0 分数以后到达再补作辅助代理校准，不阻塞这些步骤，也不能修改接受门槛。当前 GitHub 代码交付阶段不执行 Kaggle 提交。

最终交付：checkout SHA、实际授权 GPU/运行窗口、数据清洗与筛选记录、CPU 测试与四卡 smoke、完整日志路径、逐阶段状态、真实验证指标与 CI、通过的组件、构建位置、提交收据（若已授权且执行）。对未完成项写明真实原因。
