# 完整研究项目包

最新需求文档：[PRD V0.4 实验复盘与下一轮调校 · 离线 HTML 包](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_PRD_V0.4_Experiment_Review.zip)。解压后双击 HTML 阅读，包含原要求对照、真实曲线、后续实验与验收。文档为 V0.4，下面的完整应用／模型包为 V0.3。

最新调校权重：[V0.4 模型与训练更新包](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_Model_Tuning_V0.4.zip)。需要下方的 V0.3 完整项目；按 [在线调校报告](../docs/TUNING_V0.4.md) 合并文件并注册。新模型为 CPU 末残差块微调，验证 MAE 12.1334 月、同一历史留出 MAE 12.4175 月；仍未达到 <6 月目标，0–<24 月误差未改善，完整 4060 全骨干训练待本机执行。更新包不包含 `var/`，解压不会替换数据库或切换活动模型。

[下载 V0.3 完整版 BoneAtlas_MVP_0.3_Training.zip](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_MVP_0.3_Training.zip)

大小：100,096,791 字节（约 100.1 MB / 95.5 MiB）。

SHA-256：`a896e394a7fe632d0903388647023cee21809b7ee51208ba5d8e22114bfa3f56`。

包含实际训练的 ResNet-50＋性别研究权重、旧 CNN 回滚版本、中文工作台、六张公开样例和全新待质检数据库，以及固定数据清单、训练／评估证据、Python 3.14 CPU 依赖和 RTX 4060 默认 GPU 训练配置。

历史固定留出 MAE 从 **19.2524 月降至 12.9245 月**（约下降 32.9%），验证 MAE 12.7541 月。当前为 CPU 冻结骨干、训练回归头的真实结果；**尚未达到 <6 月目标**，既有留出仅作探索评估。RTX 4060 全量微调需在本机执行。

解压到新目录，进入 `bone-atlas`，打开 PowerShell：

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -r requirements.lock
.\.venv314\Scripts\python.exe scripts/check_runtime.py
.\.venv314\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器访问 `http://127.0.0.1:8000`。GPU 安装、数据准备、继续微调、恢复、评估和注册步骤见 [V0.3 训练说明](../docs/TRAINING_V0.3.md)。

源码和解压包均为 78 项测试通过、0 跳过；实际浏览器预测／热力图／复核流程和新旧模型双向回滚通过。验证使用 Linux / Python 3.14.8 / PyTorch 2.14.1 CPU。Windows x64 / CPython 3.14.0 依赖解析已检查；原生 Windows 和 RTX 4060 CUDA 实测仍以 CI／本机结果为准。[文件级下载包检查](BoneAtlas_MVP_0.3_Training.check.json)记录体积、SHA 和全新数据库状态。

包中不包含私有上传、原操作数据库、约 10 GB 完整数据集、训练缓存或虚拟环境。当前权重只存一份，首次切换时注册器自动生成当前模型快照；旧模型不可变版本已提供。

保留的历史版本：

- [V0.2 / Python 3.14 完整包](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_MVP_0.2_Python314.zip)：旧 CNN、留出 MAE 19.2524 月。
- [V0.1 / Python 3.12 完整包](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_MVP_0.1.zip)。
- [PRD V0.3 离线 HTML 包](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_PRD_V0.3_GPU_Training.zip)：仅需求文档。

此目录是版本化下载文件，并非 GitHub Releases。若直接下载没有开始，打开 ZIP 文件页面并点击 **Download raw file**。原始数据许可、研究限制见项目文档。
