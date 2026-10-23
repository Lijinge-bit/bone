# Bone Atlas · 骨龄研究工作台

基于 [PRD V0.2](docs/PRD_V0.2.html) 的可运行单时相研究原型：**PNG + 性别 → 骨龄、关注热力图与人工复核记录**。

V0.3 已增加 ResNet-50＋性别、版本化预处理、CUDA/AMP 训练、恢复、独立评估、候选注册／回滚及热力图适配。需求依据为 [PRD V0.3](docs/PRD_V0.3_GPU_TRAINING.html)；实际进展、RTX 4060 Laptop 约 8GB 的安装／训练命令见 [训练说明](docs/TRAINING_V0.3.md)。

最新需求修订见 [PRD V0.4：实验复盘与下一轮调校](docs/PRD_V0.4_EXPERIMENT_REVIEW.html)（[下载离线 HTML 包](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_PRD_V0.4_Experiment_Review.zip)）。文档对照原 Word 和归档实测结果，明确精度差距、GPU 成对实验、独立评估前提与验收清单；当前应用／模型包版本仍为 V0.3。

当前云端活动研究模型已更新为 **ResNet-50 末残差块＋性别／回归头微调、224 输入**：验证 MAE **12.7541 → 12.1334 月**，历史固定留出 MAE **12.9245 → 12.4175 月**。本轮为真实 CPU 单种子、3 轮成对实验；低龄段仍有较大误差，**<6 月目标未通过**，患者独立性未验证。结果、限制、安装新权重和后续训练见 [V0.4 调校报告](docs/TUNING_V0.4.md)。RTX 4060 全骨干 CUDA 对照入口见 [本机实验说明](docs/E1_4060.md)，尚未在云端实机执行。

## 下载完整研究项目包

- [直接下载 V0.3 完整包 BoneAtlas_MVP_0.3_Training.zip](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_MVP_0.3_Training.zip)
- [打开下载文件页面](https://github.com/Lijinge-bit/bone/blob/main/downloads/BoneAtlas_MVP_0.3_Training.zip)：如直接下载未开始，可点击页面上的 **Download raw file**。

V0.3 完整包附带冻结 ResNet 基线、旧 CNN 回滚版本、六张公开样例、全新待质检数据库、固定数据清单、真实训练／评估证据和 GPU 默认配置。解压到新目录后进入 `bone-atlas`，按下文安装依赖并运行。包内不含约 10GB 原图、训练缓存、虚拟环境或私有上传。

本轮新权重和训练工具另见 [V0.4 调校更新包](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_Model_Tuning_V0.4.zip)。它需要 V0.3 完整项目，包含候选元数据、匹配证据和权重下载校验工具；按 [调校报告](docs/TUNING_V0.4.md) 合并文件，自动下载还原权重后显式研究注册。更新包不包含 `var/`，解压不更改现有数据库或活动模型。文档／调校轮次为 V0.4，应用版本仍为 V0.3。

GitHub 的 **Code → Download ZIP** 下载的是源码快照；上面的链接下载的是带模型的研究项目包。项目包采用版本化的 ZIP 作为明确的分发例外，普通 `var/` 运行目录、私有上传与数据库仍不提交到源码仓库。

## 本环境启动

```bash
cd /workspace/bone
/workspace/bone-data/venvs/py314/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

服务默认仅监听本机。启动后通过本机浏览器访问端口 8000；云环境需使用其提供的端口访问能力，聊天中的本机地址不是公开预览链接。

当前环境的依赖、完整 Kaggle ZIP、缓存、已训练权重和六张公开样例已准备。运行中的进程不会随新机器快照自动恢复，需重新启动服务。

## 新环境安装

使用标准 64 位 CPython 3.14。CPU 锁定 **torch 2.14.1+cpu / torchvision 0.29.1+cpu**；默认 GPU 训练单独锁定 **cu130** 构建。Linux 实测为 Python 3.14.8；用户已在 Windows 3.14.0 通过旧模型兼容检查。新 GPU 运行需要在 RTX 4060 上执行真实预检，不能用依赖解析代替。历史兼容记录见 [Python 3.14 升级记录](docs/PYTHON314.md)。

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

依赖通过正常 HTTPS 和包校验安装，不关闭 TLS 或修改校验值。

## 数据与真实模型

使用 [Kaggle RSNA Bone Age / kmader](https://www.kaggle.com/datasets/kmader/rsna-bone-age) 版本 2；公开下载入口在当前环境无需账号。网络需允许 `www.kaggle.com` 和 `storage.googleapis.com`。请核实原始条款及实际使用范围，不把公开下载等同所有用途均已授权。

```bash
# 下载约 9.97 GB 原始 ZIP，保留原图，默认存放 /workspace/bone-data
.venv/bin/python scripts/download_data.py

# 全量匹配、CRC、解码、精确重复与相同 dHash 候选分组，生成固定分区
.venv/bin/python scripts/prepare_data.py

# 从原图生成新的固定 224 缓存
.venv/bin/python scripts/prepare_training_data.py --data /workspace/bone-data --size 224

# 可选 CPU 小模型对照；只按验证选择，不读留出，不替换活动模型
.venv/bin/python scripts/train.py --config configs/train_cnn60_cpu.json --data /workspace/bone-data --output experiments --device cpu

# 仅生成 200 张无标签图的预测，不计算 MAE
.venv/bin/python scripts/predict_unlabeled.py

# 服务运行后导入六张真实样例，不自动接受质量或伪造预测
.venv/bin/python scripts/seed_samples.py
```

原始 ZIP 约 9.97 GB；224 灰度缓存约 0.63 GB，还需模型、优化器检查点及临时工作空间。完整图片未全部解压，原图保留在 ZIP 内，应用只存用户实际导入的图片。

表格检查：训练 12,611 条、测试 200 条；全量 12,811 张图片已匹配，0 解码异常。固定有标签分区：训练 10,091、验证 1,260、留出测试 1,260。没有患者 ID，因此不能宣称患者级隔离已验证。完整指标和限制见 [模型说明](docs/MODEL_CARD.md) 与 [验收记录](docs/ACCEPTANCE.md)。

`prepare_data.py` 保留数据源 ID、真实标签及分区用于评估，但模型只接收图像和性别。文件名、ID、真实骨龄和缺失的实足年龄均不作为预测输入。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_runtime.py
node --check app/static/app.js

# 可选浏览器闭环，需已安装 Playwright 和系统 Chromium
.venv/bin/python scripts/check_browser.py
```

浏览器检查真实执行重复导入、质量复核、模型推理、热力图和意见保存。运行后会留下清楚标识的自动化验收记录。

## 配置与文件

| 配置 | 默认值 | 用途 |
|---|---|---|
| `BONE_STORAGE` | 仓库 `var/` | SQLite、导入原图、解释图、模型与报告 |
| `BONE_DATASET` | `/workspace/bone-data` | 原始 ZIP、清单、审计与训练缓存 |

当 `/workspace/bone-data` 不存在时，数据目录回退至项目内 `data/`。可下载项目包包含原环境的审计报告副本，界面会区分附带报告与本机是否具有完整图片包。

### Windows 启动项目包

下载新版完整包并解压，在包含 `requirements.lock` 和 `app` 的 `bone-atlas` 目录打开 PowerShell。使用新的 `.venv314` 环境，避免沿用其他 Python 版本的旧环境：

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -r requirements.lock
.\.venv314\Scripts\python.exe scripts/check_runtime.py
.\.venv314\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

在本机浏览器访问 `http://127.0.0.1:8000`。项目包附带真实 CPU 研究权重、六张已下载的公开样例及其研究记录，可以在安装依赖后直接查看和评估；不包含约 10 GB 完整图片或虚拟环境。完整数据重新准备时可用脚本参数指定本地目录，例如 `--output ./data`、`--data ./data` 和 `--archive ./data/rsna-bone-age-v2.zip`。

PyCharm 的项目解释器选择 `bone-atlas\.venv314\Scripts\python.exe`。以上命令直接调用该解释器，无需先激活虚拟环境。这组命令安装 CPU 推理环境；V0.3 完整包提供冻结基线，V0.4 调校更新提供新的末残差块微调候选，实际效果和限制以对应报告为准。

```text
app/main.py          FastAPI、版本化记录、任务与复核
app/static/          无构建依赖的中文浏览器工作台
app/imaging.py       PNG 验证、可复核质控提示与等比填充
app/network.py       CNN／ResNet-50＋性别模型适配
app/predictor.py     权重校验、预测与 Grad-CAM
scripts/             下载、审计、训练、预测与浏览器检查
tests/               流程与真实模型集成测试
docs/                PRD、模型卡和验收说明
var/                 运行数据与权重（Git 忽略）
```

数据库与权重属于运行数据，不提交到 Git。新环境若没有文件快照，应按上述流程重新准备；模型不可用时 API 返回明确错误，工作台允许先导入与复核，不生成假结果。

## 访问与后续阶段

当前为本机单用户研究原型。使用远程服务、多人或真实临床数据前，需加入身份认证、机构权限及相应授权流程。已有同源修改检查、参数化 SQL、文件限制、数据版本和权重完整性校验，不将这些基础措施描述为医院级合规认证。

后续重点是提高模型精度（较高分辨率、预训练模型、经验证的手部／ROI 处理）、补充骨骼位置框及医院 DICOM 样本；纵向联合预测另外需要患者身份、检查日期、身高与真实成年／闭合结局。
