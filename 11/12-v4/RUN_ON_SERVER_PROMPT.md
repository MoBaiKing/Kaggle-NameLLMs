请在目标服务器（4 张 NVIDIA RTX 5000 Ada）上运行 Kaggle Name That LLM 的 11/12-v4（A1 + 公开集成 + A2 + B）：完成预检和四卡 smoke，再启动正式训练、打分和验证。实际执行并处理可恢复问题；不要修改实验方法，不要用测试集或公榜调参。

仓库：https://github.com/MoBaiKing/Kaggle-NameLLMs
分支：v3-oof-ensemble-ce
方法目录：11/12-v4/

先读本版本 README.md 和 artifacts/experiment_plan.json，记录完整 commit SHA。不要运行 11/11、11/12、v2、v3 的训练入口。

必须保持：seed42；四个 torchrun rank 协作；锚定代理 native 占比 0.75；A1/A2/B 的目标、学习率、epoch、权重、门控和特征按 experiment_plan.json 冻结；A2 只用外部 prompt-clean 数据训练。不要提交未改动的公开 Notebook，没有公开兜底。

1. 检查 Python、CUDA、nvidia-smi、四卡空闲显存和权限。有预约系统时按本人最新有效预约运行；不占用他人卡，不用 CPU 跑 1.7B/8B 训练或推理。
2. 环境与资产沿用 11/11/README.md。在 11/12-v4 下：

```bash
python ../11/scripts/assets.py all
python scripts/assets_v4.py
```

3. CPU 检查（只用合成小模型/已有 OOF）：

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_a1_v4.py
CUDA_VISIBLE_DEVICES='' python tests/test_a2_v4.py
CUDA_VISIBLE_DEVICES='' python tests/test_sage_v4.py
CUDA_VISIBLE_DEVICES='' python tests/test_decide_v4.py
python scripts/run.py --dry-run
python scripts/run.py --smoke
```

   `cache/external_v4.json` 和 `cache/oof_targets_v4.json` 的 payload_sha256 应与 experiment_plan.json 记录一致；不一致时先排查，不要直接继续。
4. 正式运行（保存完整日志与进程信息）：

```bash
python scripts/run.py
```

   退出码 75 表示窗口到期并已存断点：用同一条命令恢复，不删断点，不改参数。可以用 `--stages a1`、`--stages a2,b` 分段运行。
5. 检查每折 A1 的 done.json 和 native_eval.json（parity_ok），A2 的 done.json、a2_scores.json（official_scored = 2699）。最后读取 runs_v4/b/decision.json：报告 S1/S2/S3 相对 S0/S1 的 delta、ci95、supports/warning 和 diagnostics。负结果如实保留。
6. 用实际 Kaggle 账号名构建并预览：

```bash
python scripts/build_submission.py --username 实际账号名
python scripts/publish.py
```

   `publish.py --submit` 会消耗提交次数，必须先得到明确同意。

最终交付：checkout SHA、GPU 编号、资产校验、smoke 结果、正式日志位置、A1 逐折与 A2 状态、decision.json 摘要、Notebook 构建位置。训练还在运行时如实报告进度。
