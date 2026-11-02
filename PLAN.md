# 毕业设计实施计划

**题目**：基于改进 YOLOv8 的校园交通违规检测系统
**硬件**：RTX 3050 Ti Laptop / 4GB 显存，Windows 11
**周期**：按阶段推进，不设硬性周数
**项目目录**：`D:\学习\毕业设计`

---

## 总体策略

分两条线并行推进：

- **算法线**（第 1–9 周）：跑通 → 数据 → baseline → 改进 → 消融实验
- **系统线**（第 9–14 周）：追踪 → 违规规则 → 后端 → 前端大屏

先保证"能跑、能演示、有对比表"，再谈创新点。任何阶段卡住，都有降级方案。

---

## 阶段 0：环境准备（3 天）

### 目标
本机能稳定训练 YOLOv8n，并确认 GPU 可用。

### 具体任务
1. 建项目目录与虚拟环境（Python 3.10，已装）
2. 安装依赖（全部走清华镜像）：
   ```
   pip install ultralytics opencv-python fastapi uvicorn sqlalchemy labelImg -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. 安装 CUDA 版 PyTorch（版本按 3050 Ti 驱动选，用官方 index）
4. 验证：
   - `torch.cuda.is_available()` 为 True
   - `torch.cuda.get_device_name(0)` 显示 RTX 3050 Ti
   - `ultralytics` 能加载 `yolov8n.pt`

### 交付物
- `requirements.txt`
- `scripts/check_env.py`（打印 GPU / CUDA / 依赖版本）
- `docs/env-setup.md`（踩坑记录）

### 验收标准
终端能输出 GPU 型号与显存，无报错。

### 降级方案
GPU 装不上 → 先用 CPU 跑通推理，训练阶段再解决。

---

## 阶段 1：baseline 跑通（第 1–2 周）

### 目标
用官方 `yolov8n.pt` 在校园视频上出检测结果，建立信心和演示素材。

### 具体任务
1. 下载预训练权重（GitHub 直连不稳，按顺序试）
   - 优先 `https://hf-mirror.com/Ultralytics/YOLOv8`（国内可访问）
   - 或用 `ghproxy.net` 代理 GitHub release
   - 最后考虑 modelscope
2. 写 `scripts/detect_video.py`：输入视频 → 输出带框视频 + JSON 结果
3. 在 1080p 校园视频上测 FPS、显存占用，记录进 `docs/benchmark.md`

### 交付物
- 可运行推理脚本
- 1 段带检测框的演示视频
- 4GB 显存下的 FPS/显存基线数据

### 验收标准
视频里人和车被正确框出；FPS ≥ 15（YOLOv8n, imgsz=640）。

---

## 阶段 2：数据准备（第 3–5 周）

### 目标
构建校园交通数据集，至少 1500 张标注图。

### 类别设计（初版 5 类）
`person / ebike / car / helmet / no_helmet`

> helmet 与 person 分开标，规则引擎才能判断"谁没戴头盔"。

### 数据来源
1. **公开数据**（占 50–70%）：VisDrone、UA-DETRAC、SHWD 安全帽、Roboflow 电动车/头盔集
2. **自建数据**（占 30–50%）：校园路口拍摄，覆盖白天/傍晚、不同角度、遮挡、多人骑车
3. 标注工具：LabelImg 或 X-AnyLabeling
4. 按 8:1:1 划分，生成 `data.yaml`

### 交付物
- `datasets/campus/`（images、labels、train/val/test）
- `configs/data.yaml`
- `docs/dataset.md`（来源、数量、类别分布、标注规范）

### 验收标准
`ultralytics` 能正确读取数据集并开始训练；类别分布表能贴进论文。

### 风险与对策
标注是最耗时环节。若时间紧 → 先只做 3 类（person / ebike / car），规则只做"载人"。

---

## 阶段 3：baseline 训练（第 5–6 周）

### 目标
训练出可用的 baseline 模型，得到论文的第一组数字。

### 训练参数（4GB 显存）
```bash
yolo detect train model=yolov8n.pt data=configs/data.yaml \
  imgsz=640 batch=8 epochs=100 amp=True \
  project=runs pretrained=True cache=False workers=2
```
若 OOM → `batch=4` 或 `imgsz=512`。

### 记录指标
mAP@0.5、mAP@0.5:0.95、Precision、Recall、参数量、GFLOPs、FPS、训练时长。

### 交付物
- `runs/detect/baseline/`
- `docs/experiments.md`（baseline 一行数据）
- baseline 演示视频

### 验收标准
mAP@0.5 ≥ 0.70（自建小数据集也可接受 0.6+）。

---

## 阶段 4：改进与消融实验（第 6–9 周）

### 目标
做出 1–2 个可复现的改进，产出消融对比表。

### 改进项（按优先级）
1. **注意力模块**：在 backbone/neck 插入 CBAM 或 SE
   - 代码：新增 `models/cbam.py`，修改 yaml
   - 实验：baseline vs +CBAM
2. **损失函数**：CIoU → WIoU 或 EIoU
   - 实验：收敛曲线对比
3. **轻量化**（可选）：Backbone 换 GhostNet
   - 指标：参数量、GFLOPs、FPS

### 实验矩阵（最少 3 组）
| 编号 | 配置 | 目的 |
|---|---|---|
| E0 | baseline | 对照 |
| E1 | +CBAM | 注意力有效性 |
| E2 | +CBAM +WIoU | 组合效果 |

每组固定随机种子、相同超参、相同数据划分。

### 交付物
- 改进模型代码（模块化，可通过 yaml 开关）
- 消融表（mAP / P / R / 参数量 / FPS）
- 训练曲线图、PR 曲线图
- `docs/experiments.md` 完整记录

### 验收标准
至少一项指标有可解释的提升；即使没提升，也必须有分析（论文里"改进未生效的原因"也是内容）。

---

## 阶段 5：追踪与违规规则引擎（第 9–11 周）

### 目标
从"能检测"升级到"能判违规"。

### 规则定义
| 违规 | 判定逻辑 | 关键参数 |
|---|---|---|
| 载人 | 同一 ebike ID 关联人体框 ≥ 2 | 关联阈值 IoU/中心点距离 |
| 未戴头盔 | person 头部区域无 helmet 框 | 头部区域估计方式 |
| 逆行 | 轨迹方向与参考方向夹角 > 90° | 参考方向、最小轨迹长度 |
| 违停 | 目标静止超过 N 秒且在禁停区 | 静止阈值、禁停区多边形 |

### 工程实现
1. 用 ultralytics 内置 ByteTrack：`model.track(..., persist=True)`
2. `rules/` 目录，每条规则一个类，输入 `Track` 序列，输出 `Violation`
3. 多帧投票去误报：连续 K 帧满足条件才上报
4. 单元测试：构造合成轨迹测试每条规则（不依赖视频）

### 交付物
- `rules/base.py` + 4 个规则实现
- `scripts/run_violation.py`（视频 → 违规事件 JSON + 截图）
- `tests/test_rules.py`（覆盖 4 条规则的正负样例）
- 评测脚本：手工标注 100 段视频片段的违规真值，算精确率/召回率

### 验收标准
4 条规则在测试视频上能正确触发；系统级精确率 ≥ 0.7。

### 说明
4 条规则全做，不降级；先实现载人/头盔两类，再补逆行/违停。

---

## 阶段 6：后端 + 前端（第 11–14 周）

### 目标
做成可现场演示的 Web 系统。

### 后端（FastAPI）
| 接口 | 功能 |
|---|---|
| POST /api/upload | 上传视频/图片，返回任务 ID |
| GET /api/jobs/{id} | 查询处理进度与结果 |
| GET /api/violations | 违规记录分页查询（类型/时间/地点筛选） |
| GET /api/stats | 按类型、按天、按地点聚合统计 |
| GET /media/{path} | 返回带框截图/视频 |

数据库：SQLite（本地演示足够）→ 表 `job`、`violation`、`user`
处理方式：后台线程/任务队列，避免请求阻塞

### 前端（Vue3 + Element Plus + ECharts）
1. 上传页：拖拽上传 + 进度条
2. 结果页：视频播放器 + 违规事件列表（点条目跳转到对应帧截图）
3. 统计大屏：违规类型饼图、每日趋势折线、地点热力/柱状图

### 交付物
- `backend/`（FastAPI 完整可运行）
- `frontend/`（Vue 工程）
- `docker-compose.yml`（可选）
- 演示数据（预置 5~10 条处理好的视频结果）

### 验收标准
浏览器上传一段视频 → 2 分钟内看到检测结果和统计图表。

---

## 阶段 7：论文与答辩（第 14–16 周）

### 论文结构（建议）
1. 绪论（背景、意义、国内外现状）
2. 相关技术（YOLO 系列、注意力机制、多目标跟踪、规则引擎）
3. 系统需求分析与总体设计
4. 数据集构建与预处理
5. 改进模型设计与实验（**核心章节**）
6. 系统实现（追踪、规则、前后端）
7. 系统测试与结果分析
8. 总结与展望

### 必备图表清单
- 系统架构图、数据流图、数据库 ER 图
- 网络结构改进示意图（CBAM 插入位置）
- 训练损失/mAP 曲线、PR 曲线
- 消融对比表、与 baseline 检测效果对比图
- 规则判定流程图
- 系统界面截图、违规统计图

### 交付物
- 论文初稿（Word/LaTeX）
- 答辩 PPT（15~20 页）
- 演示视频（3~5 分钟，含 live demo 录屏）
- 答辩问题清单与回答要点（20 题）

### 验收标准
查重通过、格式符合学院模板、演示视频能离线播放。

---

## 分工：我做 / 你做

| 类型 | 我负责 | 你负责 |
|---|---|---|
| 代码 | 全部脚本、模型改进、后端、前端骨架 | 运行、反馈报错、微调界面 |
| 数据 | 下载脚本、格式转换、划分脚本 | 校园拍摄、标注、确认类别 |
| 实验 | 训练命令、参数调优、结果整理 | 开机器跑训练、上传日志 |
| 论文 | 大纲、实验章节草稿、图表脚本 | 撰写正文、按学院模板排版 |
| 决策 | 给方案与推荐 | 拍板（题目范围、时间、导师要求） |

---

## 已确认的决策（2026-11-02）

1. **项目目录**：`D:\学习\毕业设计`，所有项目文件都放这里
2. **时间线**：不设限，按阶段推进
3. **违规类型**：4 条全做（载人、未戴头盔、逆行、违停）
4. **拍摄条件**：校园电动车场景充足，可自建数据

---

## 里程碑检查点

| 阶段 | 必须完成 | 可演示的东西 |
|---|---|---|
| 阶段 0 | 环境 + GPU 可用 | 环境检查输出 |
| 阶段 1 | baseline 推理 | 一段带框视频 |
| 阶段 2 | 数据集 | data.yaml + 类别统计 |
| 阶段 3 | baseline 训练 | mAP 数字 |
| 阶段 4 | 改进 + 消融表 | 对比表格 |
| 阶段 5 | 规则引擎 | 违规事件 JSON |
| 阶段 6 | Web 系统 | 浏览器完整演示 |
| 阶段 7 | 论文 + PPT | 答辩材料 |