# Bone Atlas · 骨龄研究工作台

基于 [PRD V0.2](docs/PRD_V0.2.html) 的可运行单时相研究原型：**PNG + 性别 → 骨龄、关注热力图与人工复核记录**。

当前已有真实 CPU 模型和独立留出测试结果，**MAE 19.2524 个月，尚未达到 <6 个月目标**。本项目不是临床产品，不提供成年身高、闭合时间窗、治疗建议或医院签署报告。

## 下载完整研究项目包

- [直接下载 Python 3.14 完整版 BoneAtlas_MVP_0.2_Python314.zip](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_MVP_0.2_Python314.zip)
- [打开下载文件页面](https://github.com/Lijinge-bit/bone/blob/main/downloads/BoneAtlas_MVP_0.2_Python314.zip)：如直接下载未开始，可点击页面上的 **Download raw file** 下载按钮。

完整包约 7.5 MB，附带真实研究模型权重、六张公开样例、全新待复核样本记录和审计／评估报告。解压后进入 `bone-atlas`，按下文 Windows 或 Python 启动步骤安装依赖并运行。它不包含约 10 GB 原始数据集和虚拟环境。

GitHub 的 **Code → Download ZIP** 下载的是源码快照；上面的链接下载的是带模型的研究项目包。项目包采用版本化的 ZIP 作为明确的分发例外，普通 `var/` 运行目录、私有上传与数据库仍不提交到源码仓库。

## 本环境启动

```bash
cd /workspace/bone
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

服务默认仅监听本机。启动后通过本机浏览器访问端口 8000；云环境需使用其提供的端口访问能力，聊天中的本机地址不是公开预览链接。

当前环境的依赖、完整 Kaggle ZIP、缓存、已训练权重和六张公开样例已准备。运行中的进程不会随新机器快照自动恢复，需重新启动服务。

## 新环境安装

使用标准 64 位 CPython 3.14，运行依赖已升级至 **PyTorch 2.14.1+cpu**。Linux 实测版本为 Python 3.14.8；Windows x64 的 Python 3.14 安装包依赖已验证可解析，原生 Windows 运行需由该平台确认。完整验证结果见 [Python 3.14 升级记录](docs/PYTHON314.md)。

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

# 真实 CPU 小模型；按验证集选最佳轮次，再执行有标签留出测试
.venv/bin/python scripts/train.py --epochs 20 --threads 3

# 仅生成 200 张无标签图的预测，不计算 MAE
.venv/bin/python scripts/predict_unlabeled.py

# 服务运行后导入六张真实样例，不自动接受质量或伪造预测
.venv/bin/python scripts/seed_samples.py
```

原始 ZIP 约 9.97 GB，缓存约 0.21 GB；需要额外工作空间。完整图片未全部解压，原图保留在 ZIP 内，应用只存用户实际导入的图片。

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

PyCharm 的项目解释器选择 `bone-atlas\.venv314\Scripts\python.exe`。以上命令直接调用该解释器，无需先激活虚拟环境。升级的是运行依赖，已有权重没有重训；原有精度限制仍适用。

```text
app/main.py          FastAPI、版本化记录、任务与复核
app/static/          无构建依赖的中文浏览器工作台
app/imaging.py       PNG 验证、可复核质控提示与等比填充
app/network.py       真实 CPU CNN
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
