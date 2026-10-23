# V0.3 模型调校与 RTX 4060 训练说明

当前完整包默认模型为 **ResNet-50＋性别、224 输入、冻结骨干训练回归头**。这是实际完成训练的 CPU 研究候选，后续可以在 RTX 4060 上继续微调。

| 模型 | 选择用验证 MAE（月） | 历史固定留出 MAE（月） |
|---|---:|---:|
| 原始小 CNN，96，20 轮 | 18.7502 | 19.2524 |
| 小 CNN，96，60 轮对照 | 18.3299 | 未运行，不用测试集挑选候选 |
| ResNet-50 冻结骨干／回归头，224 | **12.7541** | **12.9245** |

候选先按验证集选定，再对 1,260 张既有留出图评估。与原模型同一留出数据相比，MAE 下降约 **32.9%**；95% 图片级 bootstrap 区间为 **[12.2627, 13.5453] 月**，3 张有限超范围预测仍计入原始误差。当前 **MAE <6 月目标未通过**。旧留出结果此前已被观察，本次报告明确为探索评估，不能声称新的独立最终／患者级检验。

ResNet 训练在第 56 轮早停，最佳为第 48 轮；ImageNet 骨干保持冻结，BatchNorm 为 eval，无图像增强。没有将此候选描述为 GPU 全量微调的结果。200 张无标签图片已导出：199 条有效、1 条 invalid_output，没有计算 MAE。

## 先运行新版完整包

将新版 ZIP 解压到新目录，进入其中 `bone-atlas`，避免覆盖旧项目的私有上传／数据库。CPU 推理环境：

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -r requirements.lock
.\.venv314\Scripts\python.exe scripts/check_runtime.py --output .\var\runtime-check.json
.\.venv314\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器访问 `http://127.0.0.1:8000`。真实模型和六张公开样例随包提供，质检确认后可评估。新架构同时支持 CPU 加载和原图坐标 Grad-CAM；旧 CNN 保留为回归／回滚版本。`check_runtime.py` 对当前模型执行真实功能检查，旧 CNN 则继续执行冻结数值参考对比；该命令不等于模型精度验收。

## RTX 4060 Laptop 约 8GB 的默认环境

GPU 环境与上面的 CPU 环境分开，避免替换已能运行的解释器。在 `bone-atlas` 目录打开 PowerShell：

```powershell
py -3.14 -m venv .venv314-gpu
.\.venv314-gpu\Scripts\python.exe -m pip install -r requirements-gpu.lock
.\.venv314-gpu\Scripts\python.exe scripts/check_environment.py --device cuda --output .\var\gpu-environment.json
```

默认锁定官方 **torch 2.14.1+cu130 / torchvision 0.29.1+cu130**，已有 Windows x64 / CPython 3.14.0 wheel 解析检查。需要支持 CUDA 13.0 的 NVIDIA 驱动；通常无需另装完整 CUDA Toolkit。首次安装 GPU 依赖及预训练资源需要联网和足够磁盘空间。

预检真正执行 ResNet 的前向、反向和 AdamW 更新，报告实际设备、AMP、显存峰值和错误。只有输出 `status: ready` 且 `resolved_device: cuda:0`（或实际 CUDA 编号）时继续训练。驱动／CUDA 不可用时命令非零退出，不会静默改成 CPU。如果提示驱动不支持，先更新 NVIDIA 官方驱动，再重跑预检；不要通过关闭 TLS／哈希校验解决安装问题。

PyCharm 的项目解释器选择本项目 `.venv314-gpu\Scripts\python.exe`。本次云环境没有 GPU，因此 CUDA 预检和全量微调仍须在你的电脑执行；Linux CPU 测试不能替代这项证据。

## 准备全量原图和固定缓存

项目包含固定数据清单与审计副本，**不含约 10GB 原始图片 ZIP 和训练缓存**。如果已有同一版本原图，放到 `data\rsna-bone-age-v2.zip`。否则执行：

```powershell
.\.venv314-gpu\Scripts\python.exe scripts/download_data.py --output .\data
```

在官方 RSNA v2 原图和随包固定清单就绪后：

```powershell
.\.venv314-gpu\Scripts\python.exe scripts/prepare_training_data.py --data .\data --size 224
```

该过程核对原图 SHA、PNG 解码与 ZIP CRC，并从原图生成 224 灰度内存映射缓存。原图不会由 96 缓存放大替代；清单固定保持 10,091 训练／1,260 验证／1,260 留出。图片 ID、标签、文件名和分区只供数据管理／评估，不进入模型特征。不同数据源需要先运行 `prepare_data.py` 形成相应审计，不能把其他图片伪装成随包源版本。

## 在显卡上继续微调

默认配置在 `configs/train_resnet50_4060.json`：224 输入、batch 8、AMP、有效 batch 8、前 3 轮冻结骨干，再解冻；骨干 LR 3e-5、回归头 LR 3e-4、AdamW、最多 60 轮，早停 patience 10 / min_delta 0.05 月。

推荐用包内已训练研究权重作为起点；加载、记录初始化 SHA 后新建独立实验，不修改活动模型：

```powershell
.\.venv314-gpu\Scripts\python.exe scripts/train.py --config .\configs\train_resnet50_4060.json --data .\data --output .\experiments --weights-path .\var\models\resnet50-frozen-cpu-224.pt
```

如需从官方 ImageNet 骨干及新回归头开始对照，省略 `--weights-path`。每个实验有唯一目录，包含 config、environment、initialization、data-reference、history、checkpoints 和 export；默认只使用训练／验证集，不读取留出集，不自动注册。

显存不足时先关闭占显存的其他程序；需要改实验设置则显式降低物理 batch 并增加梯度累积，例如：

```powershell
.\.venv314-gpu\Scripts\python.exe scripts/train.py --config .\configs\train_resnet50_4060.json --data .\data --output .\experiments --weights-path .\var\models\resnet50-frozen-cpu-224.pt --batch-size 4 --gradient-accumulation-steps 2
```

这是新配置／新运行。已失败运行恢复时须保持配置一致；不能把原 batch 8 的恢复文件悄悄当作 batch 4 的同一个实验。OOM 记录 `failed_oom`，从最后完成轮次恢复；不会在半轮残留梯度上继续。AMP 的缩放溢出会明确记录、跳过该更新并降 scale，持续失败才中断；原始非有限 loss／输出会直接失败。

## 中断和恢复

终端会显示实际 run 目录。把下面 `你的运行目录` 替换为该目录：

```powershell
.\.venv314-gpu\Scripts\python.exe scripts/train.py --data .\data --resume .\experiments\你的运行目录\checkpoints\latest.pt
```

恢复自动读取原配置，核对数据／缓存与配置，恢复模型、优化器、日程、AMP、随机状态、阶段和早停。恢复按完整轮次；中断轮次重新计算。候选在 `export\candidate.json` 和 `export\model.pt`。

## 评估、功能检查与显式注册

所有调参都看验证集。方案锁定后，对已经观察过的历史留出集作探索评估需明确声明：

```powershell
.\.venv314-gpu\Scripts\python.exe scripts/evaluate_model.py --candidate .\experiments\你的运行目录\export --data .\data --split test --acknowledge-exploratory-test --output .\experiments\你的运行目录\holdout-evaluation --device cuda --batch-size 8
.\.venv314-gpu\Scripts\python.exe scripts/check_candidate.py --candidate .\experiments\你的运行目录\export --output .\experiments\你的运行目录\integration-check.json
```

第二步在 CPU 上检查六张公开原图×F/M、重复预测、Grad-CAM 几何映射／透明度和损坏权重拒绝，不把新模型输出与旧 CNN 数字强行比较。独立的最终／外部数据尚缺，历史留出结果始终标探索用途，不会仅因数值低于 6 就显示 G-01 独立验收通过。

确认候选的真实效果和用途后，用带完整证据的命令显式研究注册：

```powershell
.\.venv314-gpu\Scripts\python.exe scripts/register_model.py --candidate .\experiments\你的运行目录\export --evaluation .\experiments\你的运行目录\holdout-evaluation --integration .\experiments\你的运行目录\integration-check.json --storage .\var --research-only --operator Machenike --reason "锁定候选后的研究评估与功能验证通过"
```

注册核对候选、逐样本结果、冻结协议、代码／文件校验与集成证据；缺项或不一致阻止切换。旧权重和结果保留。回滚原始 CNN：

```powershell
.\.venv314-gpu\Scripts\python.exe scripts/register_model.py --rollback rsna-cpu-cnn-20261022T134620Z --storage .\var --operator Machenike --reason "研究比较，回滚旧基线"
```

可以同样回滚到 `resnet50-frozen-cpu-224`。CPU 推理环境与 GPU 环境使用同一个 `var` 时应顺序操作注册，不同时手动改模型文件。注册器提供 Windows／Unix 跨进程锁和原子指针切换；活动模型变更中的任务检测快照冲突。

完整包为控制下载体积，仅存放一份当前 ResNet 权重，并保留旧 CNN 的完整不可变版本。首次切换时，注册器会先为当前模型生成完整快照，再修改活动指针；已在解压包中实际验证 ResNet → CNN → ResNet 双向回滚。切换／训练需要额外磁盘空间。

## 继续提高精度时看什么

先看无增强 train MAE 与 validation MAE 的差距、性别／骨龄分层偏差，再决定后续实验。当前 60 轮小 CNN 的训练误差继续下降而验证停滞，单纯加轮数收益有限。ResNet 全量微调完成后，再单独比较 384 分辨率、经过核对的手部裁剪、损失和采样。不能用删除难例、只报成功样本、反复换测试分区来达标。

0–<24 月训练仅 73、留出仅 9 张，需报告不确定性。原数据无患者 ID／检查日期，患者级隔离未验证；时序、成年身高和闭合任务仍依赖新增随访／结局数据。训练成功、研究精度、独立最终验证及临床使用分别判定。
