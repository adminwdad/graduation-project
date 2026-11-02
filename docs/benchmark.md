# 推理基线与性能测试（阶段 1）

- 日期：2026-11-02
- 硬件：RTX 3050 Ti Laptop（4 GB 显存）/ Windows 11
- 软件：torch 2.9.1+cu128、ultralytics 8.4.198、opencv-python 5.0.0
- 模型：官方 `yolov8n.pt`（COCO 80 类，未微调）
- 脚本：`scripts/detect_video.py`

## 测试素材

目前还没有真实校园视频，先用 Ultralytics 官方示例图 `bus.jpg` 生成一段 **1920×1080 / 30 fps / 300 帧（10 秒）** 的平移测试视频做冒烟测试：

`datasets/samples/campus_demo_1080p.mp4`（合成素材，仅用于跑通流程）

> 拿到真实校园视频后需要用同样的命令复测，本页数据到时更新。

## 测试命令

```powershell
# FP32
.venv\Scripts\python.exe scripts\detect_video.py `
  --source datasets\samples\campus_demo_1080p.mp4 --imgsz 640

# FP16（ultralytics 8.4.198 用 quantize=16 代替 half）
.venv\Scripts\python.exe scripts\detect_video.py `
  --source datasets\samples\campus_demo_1080p.mp4 --imgsz 640 --half
```

## 结果

| 配置 | 分辨率 | 纯推理 (ms/帧) | 纯推理 FPS | 端到端 FPS | 显存峰值（分配/保留） |
|---|---|---|---|---|---|
| YOLOv8n FP32, CUDA | 1920×1080 → imgsz 640 | 11.73 | **85.3** | 41.2 | 24.7 / 40.0 MB |
| YOLOv8n FP16, CUDA | 1920×1080 → imgsz 640 | 12.37 | **80.9** | 40.3 | 13.1 / 34.0 MB |
| YOLOv8n FP32, CPU（单图） | 810×1080 → imgsz 640 | 41.95 | 23.8 | 23.6 | — |

端到端 FPS 包含视频解码、画框和写盘；纯推理只统计模型前向 + NMS。

## 结论

1. GPU 纯推理 FPS 约 80–85，**远超阶段 1 验收线 15 FPS**，4 GB 显存余量充足。
2. 当前端到端约 41 FPS，瓶颈在 CPU 侧视频解码/编码与逐帧写盘，不在模型本身。
3. `imgsz=640` 下 FP16 没有带来速度提升（甚至略慢），原因是小模型前向时间短、主要开销在预处理；FP16 的价值主要是省显存。
4. 显存统计为 PyTorch 分配器峰值，不含 CUDA context；训练阶段需另用 `nvidia-smi` 监控总占用。
5. CPU 也能跑（单图约 24 FPS），GPU 装不上时的降级方案可用。

## 已知限制 / 下一步

- **待办**：用真实校园 1080p 视频复测，补上 FPS、显存和检测效果截图。
- 合成测试视频里的 bus 被误判为 airplane，属于图像被拉伸/裁剪后的域偏移，不代表真实视频效果。
- 后续阶段 5 会把逐帧检测升级为 `model.track()`，到时再测带跟踪的 FPS。