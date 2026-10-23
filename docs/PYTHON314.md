# Python 3.14 运行升级

0.2 版将 `torch==2.7.1+cpu` 升级为 `torch==2.14.1+cpu`，支持标准 64 位 CPython 3.14。`requirements.lock` 同时固定运行依赖和 Windows 专用的 colorama；不需要安装 torchvision，也不需要 CUDA。保留原始权重与其 SHA-256，没有重新训练或修改预处理。

## Windows / PyCharm

先下载 README 中的 **BoneAtlas_MVP_0.2_Python314.zip**，解压到新目录，避免覆盖已有私有上传和运行记录。进入其中 `bone-atlas` 目录，确认能看到 `app`、`requirements.lock`、`var/models/active.json`，在 PowerShell 执行：

```powershell
py -3.14 --version
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -r requirements.lock
.\.venv314\Scripts\python.exe scripts/check_runtime.py
.\.venv314\Scripts\python.exe -m unittest discover -s tests -v
.\.venv314\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器访问 `http://127.0.0.1:8000`，公开样例需要先人工确认质量，再点击骨龄评估和热力图。PyCharm 在“设置 → 项目 → Python 解释器”选择当前目录 `.venv314\Scripts\python.exe`。没有激活环境也可以运行上面的命令。

若 `py -3.14 --version` 失败，先用 `py -0p` 查看 Python 安装路径；可将第一条创建环境命令替换成 `& "你的Python314完整路径\python.exe" -m venv .venv314`。不要再使用 `py -3.12`。GitHub 的 Code → Download ZIP 是源码快照，带权重与样例的完整包请使用 README 的专用下载链接。

## 检查方法

`scripts/check_runtime.py` 会真正加载经过 SHA-256 校验的旧权重，检查六张原始公开图片各自的 F/M 输入（12 组）、重复预测、原图尺寸的 RGBA Grad-CAM、透明度与空间关注分布。比较基准保存在 `reports/runtime-reference.json`，来自 Python 3.12.14 / PyTorch 2.7.1 CPU，固定预测容差 0.001 月；不会重新生成基准来迎合新版结果。

完整数据环境可附加 `--cache /路径/cache-96.npz`，重跑固定 1,260 张留出样本，核对图片 ID、性别、标签、缓存校验值及原预测 CSV。此步骤是运行兼容回归验证，没有用留出集调参。`--bundle 文件.zip --storage 临时目录` 可以只准备校验后的公开模型与图片，用当前源码执行验证，不复制操作数据库。

Linux 实测中，12 组预测最大差异为 **0.00002289 月**，12 张 Grad-CAM 的 RGBA 像素与旧版本完全一致；留出集最大预测差异为 **0.00007138 月**。15 项测试无跳过，独立解压包启动的真实浏览器流程也通过。

原模型有 3 张留出图片输出负值（ID 1559、4249、15479）。兼容报告单独记录，原始 MAE 使用未经裁剪的预测计算。应用对超出 0–360 月的结果仍返回评估失败，不展示为有效骨龄；依赖升级没有修复模型本身的这项限制。

200 张无标签图片也已重新导出：198 条有效预测，4455、4464 两条仍标记为 `invalid_output`；新旧输出状态完全一致。无标签数据没有 MAE，报告见 `reports/python314-unlabeled.json`。

## 验证范围

实际 Linux / CPU 的 Python 3.14.8 结果见 [运行兼容报告](reports/python314-runtime.json)，另有测试、项目包和浏览器报告。Windows x64 使用 uv 的目标平台解析验证全部锁定依赖具有兼容 wheel；这是安装依赖检查，不能等同于在 Windows 主机运行成功。GitHub Actions 已配置 Python 3.14 的 Ubuntu / Windows 矩阵，用当前源码加载完整包的公开权重和样例再次验证，并上传各自报告。

训练时的历史报告保留 PyTorch 2.7.1 版本信息，这是原训练环境。新版兼容报告独立记录 PyTorch 2.14.1。模型留出 MAE 仍为 **19.2524 月**，未达到 PRD 的 <6 月目标，仅用于研究流程。
