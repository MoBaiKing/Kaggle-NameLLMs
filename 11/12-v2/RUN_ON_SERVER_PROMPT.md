请把 Kaggle Name That LLM 的 GRPO v2 迁移到目标服务器，完成四卡预检并启动正式训练。请实际执行并持续处理可恢复问题，不停留在给建议；不要变更实验方法或用测试集调参。

仓库：https://github.com/MoBaiKing/Kaggle-NameLLMs
分支：codex/grpo-v2-enumerated-ensemble
方法目录：11/12-v2/

先读取目标目录适用的 AGENTS.md、本版本 README.md 和 artifacts/experiment_plan.json。新建独立 checkout 或保护现有未提交修改，拉取上述分支，并记录完整 commit SHA；如果交接消息给出指定 commit，则 checkout 该 commit，核对 SHA 后再运行。不要运行 11/11 或 11/12 的旧训练入口。

这版必须保持：seed42；公开折0、1、2、3、4；每折由四个 torchrun rank 协作一个模型，折之间顺序执行；一个全局 batch 为16条文本。原始公开 native 权重作为初始化，不能加载 v1 学生状态。显式投票使用既有 Qwen3-14B、Qwen3-8B、Llama-3.1-8B 成员，3:2:1 概率平均；总预测在原有平衡操作前为完整公开集成45%、显式三模型投票15%、优化 native 40%。保留代码中预先固定的18类枚举目标、学习率、epoch、KL、校准与投票权重。

1. 检查服务器 Python、CUDA、nvidia-smi、四卡空闲显存和使用权限。若服务器使用 gpubook，持续读取本人最新有效预约，四张卡必须同时在授权窗口且可用；预约变化后及时保存/释放并安排恢复，无有效窗口则等待。不得改用 CPU 跑8B训练/推理，不占用他人卡，不修改或取消用户预约。若有 FocusForge，按其 README 先登记稳定任务ID并持续同步状态，不把训练结束写成最终测试完成。

2. 按 11/11/README.md 准备独立训练环境。推荐 Python3.11，安装 PyTorch2.2.1/CUDA12.1 与 11/11/requirements.txt 中的固定依赖；Kaggle CLI 可用独立 Python3.12 环境，并把其命令加入 PATH。先检查已有兼容环境再安装。凭据使用服务器已有配置，不打印或写入仓库；缺少认证或比赛权限时准确说明阻塞原因。

3. 下载/复用并校验固定资产。在仓库 11/12-v2 目录运行 `python ../11/scripts/assets.py all`。不得以同名最新资产替代固定历史版本。复用已有文件前核对哈希、数据顺序、模型 revision 和 tokenizer；v1 tokens 与公开 native reference 只有代码自动核验通过才可共用。保留 v1 参考缓存的符号链接目标；v2 checkpoint、优化器、验证与完成记录独立保存。

4. 先跑以下 CPU 合成/OOF 检查，再做真实四卡 smoke：

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_objective_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_parent_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_distributed_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_pipeline_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_notebook_v2.py
CUDA_VISIBLE_DEVICES='' python tests/test_selection_v2.py
python scripts/run.py --gpus 0,1,2,3 --dry-run
python scripts/run.py --gpus 0,1,2,3 --smoke
```

GPU编号应替换为实际四张获准使用的不同卡，保持四进程协作。CPU命令仅运行小模型/既有OOF分析，不是8B补测。smoke 使用独立 runs_grpo_smoke，不可作为正式结果。预检失败先排查，不跳过技术检查。

5. 四卡 smoke 通过且预约窗口允许后，启动正式命令并保存完整日志与进程信息：

```bash
python scripts/run.py --gpus 0,1,2,3
```

默认每折90分钟是保存/暂停窗口，不是提前结束epoch的条件；必须为后续GPU native验证预留时间。torchrun 工作进程退出75后，启动器可能非零退出：结合日志和 checkpoint 的真实状态区分暂停与失败。恢复时使用同一输出目录、相同四进程数和相同协议参数，不能删断点重头训练。程序会自动缩小 micro batch 处理OOM，不要在同一断点修改启动参数；不能缩短epoch、增加种子或改损失来让作业看起来完成。外部预约管理应在释放前给工作进程足够保存时间。

6. 每折训练完成立即运行入口自带的四卡 native GPU验证；检查 done.json、导出状态、native_eval.json、校准与文件哈希。训练成功但验证失败时如实标注，继续处理可恢复问题。全部预定折通过技术核验后，检查 runs_grpo/decision.json 中 overall_vs_original 和 grpo_vs_same_vote，分别报告总改动和GRPO本身的验证代理差异。它们不是完整Kaggle隐藏测试成绩；负结果也必须保留，不用公榜反选权重或折。

7. 使用用户现有 Kaggle 账号名（不要编造）构建：

```bash
python scripts/build_submission.py --username 实际账号名
python scripts/publish.py
```

第二条是预览；本次迁移任务不包含消耗提交额度的 `--submit`。检查构建的是独立 `ntl-11-12-grpo-v2` Notebook、优化权重包和三个强模型投票路径。原有公开其他成员保留，投票成员缺失、五折未完成、native超时或哈希不符必须明确停止。

最终交付：实际checkout SHA、服务器/四卡型号和GPU编号、环境与资产校验、smoke结果、正式命令及日志/进程位置、逐折真实状态、断点与最终导出路径、两组验证对照、Notebook构建位置。如训练仍在运行，持续监控并报告实际进度，不声称已提分或已完成隐藏测试。
