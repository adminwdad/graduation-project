# 环境搭建记录（阶段 0）

- 日期：2026-11-02
- 硬件：RTX 3050 Ti Laptop（4GB 显存），驱动 596.21，支持 CUDA 13.2
- 系统：Windows 11（22631）
- Python：3.10.7
- 虚拟环境：项目根目录 `.venv`

## 安装命令

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install ultralytics opencv-python fastapi uvicorn sqlalchemy -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv\Scripts\python.exe -m pip install labelImg -i https://pypi.tuna.tsinghua.edu.cn/simple
```

CUDA 版 PyTorch（官方源下载稍慢但稳定）：

```powershell
curl.exe -L -C - -o torch-2.9.1+cu128-cp310-cp310-win_amd64.whl `
  "https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp310-cp310-win_amd64.whl"
curl.exe -L -C - -o torchvision-0.24.1+cu128-cp310-cp310-win_amd64.whl `
  "https://download.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp310-cp310-win_amd64.whl"
.venv\Scripts\python.exe -m pip install .\torch-2.9.1+cu128-*.whl .\torchvision-0.24.1+cu128-*.whl
```

## 踩坑记录

1. **labelImg 安装报 GBK 解码错误**
   `UnicodeDecodeError: 'gbk' codec can't decode byte ...`
   实际包已装上，但 pip 最后读记录文件失败。解决办法：先 `pip install wheel`（让 pip 走 bdist_wheel 而不是 legacy setup.py install），或设 `PYTHONUTF8=1` 后重试。

2. **torch CUDA 版本选择**
   - 清华 PyPI 源自动拉的是 `torch 2.14.1+cpu`，必须换成 CUDA 版。
   - 官方 cu130 通道有 `torch 2.14.1+cu130`，但 Windows + Python 3.10 没有配套的 CUDA 版 `torchvision`（只有 0.29.1），强行组合会导致 torchvision 算子不可用。
   - 最终选稳定配对：**torch 2.9.1+cu128 + torchvision 0.24.1+cu128**（sm_86 完全支持）。

3. **下载源速度实测（2026-11-02，本机）**
   | 源 | 速度 |
   |---|---|
   | download.pytorch.org（官方） | 3.5–4 MB/s ✅ |
   | SJTU 镜像 | 6 MB/s 但中途反复 SSL 报错 (curl 35) |
   | 阿里云 pytorch-wheels | 0.5 MB/s |
   | 华为 / NJU / USTC / 清华 | 无此文件或基本不可用 |

4. **curl 下载 wheel 的 URL 编码**
   文件名里的 `+` 在 URL 中必须写成 `%2B`，否则官方源返回 403。

## 验证结果

`scripts/check_env.py` 输出：

```
Python : 3.10.7 (64bit)
torch  : 2.9.1+cu128
torchvision: 0.24.1+cu128
CUDA 编译版本: 12.8
cuDNN  : 91002
GPU    : NVIDIA GeForce RTX 3050 Ti Laptop GPU
显存   : 4.0 GB
算力   : sm_86
CUDA 可用: True
ultralytics  8.4.198
cv2          5.0.0
fastapi      0.162.0
uvicorn      0.54.0
sqlalchemy   2.0.55
numpy        2.2.6
YOLOv8n 加载成功，类别数: 80
```

## 交付物

- `.venv/` 虚拟环境
- `requirements.txt`（`pip freeze` 生成，54 个包）
- `scripts/check_env.py` 环境自检脚本
- `wheels/` 离线安装包（torch/torchvision，确认无误后可删）