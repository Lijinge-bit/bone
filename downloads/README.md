# 完整研究项目包

后续训练升级需求文档另见 [PRD V0.3 离线 HTML 包](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_PRD_V0.3_GPU_Training.zip)。其中只含一份可双击阅读的静态 HTML，面向 RTX 4060 Laptop 约 8GB / Python 3.14，覆盖训练、评估、推理／热力图适配及验收。它是后续需求稿，现有应用完整包仍为下方的 0.2 版本。

[下载 Python 3.14 完整版 BoneAtlas_MVP_0.2_Python314.zip](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_MVP_0.2_Python314.zip)

大小：7,517,235 字节（约 7.5 MB）。

SHA-256：`a2409e565c78a573eb468b9ed4aa14960f0679a148dd9d25178d89a5a2a82c40`。

新版使用 Python 3.14 / PyTorch 2.14.1 CPU，包含：

- Python 后端、中文浏览器工作台、固定依赖与 Windows / PyCharm 启动说明；
- 实际训练的 CPU 研究权重与 SHA-256 校验信息；
- 六张公开 RSNA 样例、全新待质检的数据库；
- 数据审计、原始模型评估、3.14 兼容测试和浏览器验证报告。

解压后的根目录为 `bone-atlas`。在该目录打开 PowerShell：

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -r requirements.lock
.\.venv314\Scripts\python.exe scripts/check_runtime.py
.\.venv314\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器访问 `http://127.0.0.1:8000`。实际运行验证使用 Linux / Python 3.14.8；Windows x64 依赖 wheel 解析已通过，原生 Windows 运行以对应 CI 或本机结果为准。

包中不包含私有上传、原操作数据库、约 10 GB 完整数据集或虚拟环境。保留的 [旧版 0.1 包](https://github.com/Lijinge-bit/bone/raw/refs/heads/main/downloads/BoneAtlas_MVP_0.1.zip) 使用原 Python 3.12 / PyTorch 2.7.1 依赖；Python 3.14 请下载上面的新版。

依赖升级没有重训模型，留出 MAE 仍为 **19.2524 个月**，未达到 <6 个月目标，仅用于研究。原始数据许可及实际使用范围见项目文档。

此目录是版本化下载文件，并非 GitHub Releases。若浏览器没有开始下载，打开 ZIP 的文件页面并点击 **Download raw file**。
