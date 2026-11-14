请在 4 张 NVIDIA RTX 5000 Ada 上运行 Kaggle Name That LLM 的 11/12-v6：拉取代码、准备环境和数据、完成预检和 smoke，再正式训练、评估、过门槛；门槛通过后构建并提交一次 Kaggle。实际执行并处理可恢复问题；不要改方法，不要用测试集或公榜调参。

仓库 https://github.com/MoBaiKing/Kaggle-NameLLMs ，分支 v3-oof-ensemble-ce，方法目录 11/12-v6/。新建独立 checkout，记录完整 commit SHA。先读 11/12-v6/README.md。不要运行其他版本的训练入口。

必须保持：seed 42/43/44 三个种子做汤；四个 torchrun rank 协作；数据规则、超参和门槛按 README 冻结；不提交未改动的公开 Notebook。

1. 环境与资产按 11/11/README.md 准备，然后在 11/12-v6 下：

```bash
python ../11/scripts/assets.py all
mkdir -p ../11/external/open_pool_v15
kaggle datasets download crtproduction/name-that-llm-open-pool -f ntl_open_pool.parquet -p ../11/external/open_pool_v15 --unzip
kaggle datasets download crtproduction/name-that-llm-open-pool -f fidelity_by_reasoning.csv -p ../11/external/open_pool_v15 --unzip
```

   `ntl_open_pool.parquet` 的 SHA256 必须是 5a3dc82288328abedf49b7dd5e866e5c69eb922f9d6008e6fd78cf5153c12332，否则停下报告，不要改代码绕过。

2. CPU 检查：

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_data_v6.py
CUDA_VISIBLE_DEVICES='' python tests/test_train_v6.py
CUDA_VISIBLE_DEVICES='' python tests/test_decide_v6.py
python scripts/run.py --dry-run
python scripts/run.py --smoke
```

   `cache/stream_v6.json` 的 payload_sha256 应为 ce7f0e08fa714bdb9e5aba6a6d087e10d3254465e8acd7e98e7a210c6f7d24bf。smoke 输出在 runs_v6_smoke/，不是正式结果。

3. 正式运行（保存完整日志与进程信息）：

```bash
python scripts/run.py --minutes 600
```

   退出码 75 表示本次时间预算用完并已存断点，用同一条命令继续；不要删断点，不要改参数。显存不足时程序会自动减小 micro batch。

4. 完成后读 runs_v6/decision.json，报告：
   - native_alone 三种分组下的 base_ba/ba/delta/ci95/per_fold_delta
   - 每折的一致性检查
   - decision

   负结果如实报告。

5. 只有 decision 为 v6 时，才用实际 Kaggle 账号名构建、预览并提交一次：

```bash
python scripts/build_submission.py --username 实际账号名
python scripts/publish.py
python scripts/publish.py --submit
```

   decision 为 not_ready 时不提交，报告原因。

最终交付：commit SHA、GPU 编号、资产与数据哈希、smoke 结果、正式日志位置、各阶段耗时、decision.json 摘要；如已提交，给出提交编号和公榜分数。
