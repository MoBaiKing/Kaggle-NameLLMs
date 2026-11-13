请在目标服务器（4 张 NVIDIA RTX 5000 Ada）上运行 Kaggle Name That LLM 的 11/12-v3：完成预检、四卡 smoke，再启动正式训练、验证和门槛决策。实际执行并处理可恢复问题；不要修改实验方法，不要用测试集或公榜调参。

仓库：https://github.com/MoBaiKing/Kaggle-NameLLMs
分支：v3-oof-ensemble-ce
方法目录：11/12-v3/

先读本版本 README.md 和 artifacts/experiment_plan.json。新建独立 checkout 或保护现有未提交修改，记录完整 commit SHA；交接时给了指定 commit 就 checkout 并核对。不要运行 11/11、11/12、11/12-v2 的训练入口。

必须保持：seed42；公开折 0–4；每折四个 torchrun rank 协作一个模型，折之间顺序执行；全局 batch 16 条文本；初始化只用公开 native 权重；目标、学习率、epoch、KL、OOF 权重、WiSE α=0.5 和门槛规则按 experiment_plan.json 冻结。

1. 检查 Python、CUDA、nvidia-smi、四卡空闲显存和使用权限。服务器有预约系统时按本人最新有效预约运行，四张卡必须同时可用；不占用他人卡，不改用 CPU 跑 8B 训练或推理。
2. 环境与资产沿用 11/11/README.md（Python 3.10/3.11，PyTorch 2.2.1/CUDA 12.1，11/11/requirements.txt）。在 11/12-v3 下运行 `python ../11/scripts/assets.py all`，不得用同名最新资产替代固定版本。
3. CPU 检查（只用合成小模型/已有 OOF）：

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_objective_v3.py
CUDA_VISIBLE_DEVICES='' python tests/test_proxy_v3.py
CUDA_VISIBLE_DEVICES='' python tests/test_pipeline_v3.py
CUDA_VISIBLE_DEVICES='' python tests/test_decide_v3.py
CUDA_VISIBLE_DEVICES='' python tests/test_notebook_v3.py
python scripts/run.py --gpus 0,1,2,3 --dry-run
python scripts/run.py --gpus 0,1,2,3 --smoke
```

   `cache/oof_targets.json` 里的 payload_sha256 应与 artifacts/verification.json 记录一致；不一致时先排查（数据、资产、numpy 版本），不要直接继续。
4. smoke 通过且窗口允许后，正式运行并保存完整日志与进程信息。有已完成的 v2 权重就加 `--v2-runs`：

```bash
python scripts/run.py --gpus 0,1,2,3 --v2-runs ../12-v2/runs_grpo
```

   退出码 75 表示窗口到期已保存断点：用同一条命令、同一输出目录、同样四卡恢复，不要删除断点重训，不要改参数。
5. 每折检查 done.json、trainables*.safetensors、native_eval.json（parity_ok）和哈希。全部完成后读取 runs_v3/decision.json：报告 decision、各候选的 delta/ci95/standalone_native_delta/per_fold_delta 和敏感性。负结果如实保留；不要用公榜反选。
6. decision 不是 public_only 时，用实际 Kaggle 账号名构建并预览：

```bash
python scripts/build_submission.py --username 实际账号名
python scripts/publish.py
```

   `publish.py --submit` 会消耗提交次数，必须先得到明确同意。

最终交付：checkout SHA、GPU 型号和编号、环境与资产校验、smoke 结果、正式命令及日志位置、逐折状态、decision.json 摘要、Notebook 构建位置。训练还在运行时如实报告进度，不要声称已提分。
