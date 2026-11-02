# 基于改进 YOLOv8 的校园交通违规检测系统

毕业设计项目。完整实施计划见 [PLAN.md](PLAN.md)。

## 目录

| 目录 | 说明 |
|---|---|
| `configs/` | 数据集配置、训练超参 |
| `datasets/` | 数据集（不纳入 git） |
| `docs/` | 环境、数据集、实验记录 |
| `models/` | 改进模型代码（CBAM 等） |
| `rules/` | 违规规则引擎 |
| `backend/` | FastAPI 后端 |
| `frontend/` | Vue3 前端 |
| `scripts/` | 推理、训练、工具脚本 |
| `runs/` | 训练 / 推理产物（不纳入 git） |
| `tests/` | 单元测试 |

## 环境

- Python 3.10.7 + `.venv`
- torch 2.9.1+cu128 / torchvision 0.24.1+cu128
- ultralytics 8.4.198
- 详见 `docs/env-setup.md`、`环境依赖说明.txt`

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\check_env.py
```

## 推理

```powershell
.venv\Scripts\python.exe scripts\detect_video.py --source 视频.mp4
```

支持视频、单张图片或图片目录；结果默认输出到 `runs/detect/<源名称>/`（带框视频/图片 + `detections.json`）。性能数据见 `docs/benchmark.md`。

## 进度

- [x] 阶段 0：环境准备
- [x] 阶段 1：baseline 推理（脚本完成，合成 1080p 冒烟测试通过；待真实校园视频复测）
- [ ] 阶段 2：数据准备
- [ ] 阶段 3：baseline 训练
- [ ] 阶段 4：改进与消融
- [ ] 阶段 5：追踪与规则引擎
- [ ] 阶段 6：前后端
- [ ] 阶段 7：论文与答辩