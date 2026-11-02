"""阶段 1：用官方 YOLOv8 对视频 / 图片做 baseline 检测。

输入可以是视频、单张图片或图片目录；输出带检测框的视频/图片，以及
每帧检测结果的 JSON，并打印推理 FPS、端到端 FPS 和显存峰值。

用法（项目根目录）：
    .venv\\Scripts\\python.exe scripts\\detect_video.py --source 校园视频.mp4
    .venv\\Scripts\\python.exe scripts\\detect_video.py --source 图片.jpg
    .venv\\Scripts\\python.exe scripts\\detect_video.py --source datasets\\samples --imgsz 640
    .venv\\Scripts\\python.exe scripts\\detect_video.py --source 视频.mp4 --max-frames 300 --half

默认输出到 runs/detect/<源名称>/：
    annotated.mp4 / annotated_<图片名>.jpg
    detections.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# 参数与工具
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 baseline 推理：视频/图片 -> 带框结果 + JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="视频、图片或图片目录路径")
    parser.add_argument("--model", default="yolov8n.pt", help="模型权重路径或名称")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率（长边）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument(
        "--classes",
        default="",
        help="只保留指定类别，逗号分隔，可写类别名或 id（如 person,car 或 0,2）",
    )
    parser.add_argument("--device", default=None, help="cuda:0 / cpu；默认自动选择")
    parser.add_argument("--half", action="store_true", help="CUDA 下使用 FP16 推理")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧/图片")
    parser.add_argument("--output", default=None, help="输出目录，默认 runs/detect/<源名称>")
    parser.add_argument("--line-width", type=int, default=2, help="检测框线宽")
    parser.add_argument("--no-save-video", action="store_true", help="不写带框视频/图片")
    parser.add_argument("--no-json", action="store_true", help="不写 detections.json")
    parser.add_argument("--show", action="store_true", help="实时显示检测窗口（按 q 退出）")
    parser.add_argument("--warmup", type=int, default=5, help="正式计时前的预热推理次数")
    return parser.parse_args(argv)


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread 在部分平台不支持中文路径，用 imdecode 兜底。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def resolve_classes(model, spec: str) -> list[int] | None:
    """把类别名/id 字符串解析成 id 列表；空字符串表示不过滤。"""
    if not spec.strip():
        return None
    names = {int(k): str(v) for k, v in model.names.items()}
    name_to_id = {v.lower(): k for k, v in names.items()}
    result: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            cid = int(token)
            if cid not in names:
                raise ValueError(f"类别 id {cid} 不存在，可用范围 0-{max(names)}")
            result.append(cid)
        else:
            cid = name_to_id.get(token.lower())
            if cid is None:
                raise ValueError(
                    f"未知类别名 {token!r}；可用类别：{', '.join(sorted(name_to_id))}"
                )
            result.append(cid)
    return sorted(set(result))


def collect_sources(source: Path) -> tuple[str, list[Path]]:
    """返回 (类型, 文件列表)，类型为 video / image / images。"""
    if source.is_dir():
        files = sorted(
            p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"目录里没有图片：{source}")
        return "images", files
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在：{source}")
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image", [source]
    if suffix in VIDEO_EXTS:
        return "video", [source]
    raise ValueError(f"不支持的输入类型：{source.suffix}（支持视频或图片）")


def cuda_sync(enabled: bool) -> None:
    if enabled:
        import torch

        torch.cuda.synchronize()


def vram_peak_mb(enabled: bool) -> tuple[float, float] | tuple[None, None]:
    """返回 (模型/张量实际分配峰值, 含缓存分配器的保留峰值)，单位 MB。"""
    if not enabled:
        return None, None
    import torch

    return (
        round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    )


def boxes_to_records(result, model) -> list[dict]:
    """把 ultralytics 的 Boxes 转成可 JSON 序列化的列表。"""
    records: list[dict] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return records
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(int)
    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
        records.append(
            {
                "class_id": int(cls_id),
                "class_name": str(model.names[int(cls_id)]),
                "conf": round(float(conf), 4),
                "xyxy": [round(float(x1), 1), round(float(y1), 1),
                         round(float(x2), 1), round(float(y2), 1)],
                "xywh": [round(float(x1), 1), round(float(y1), 1),
                         round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
            }
        )
    return records


# --------------------------------------------------------------------------- #
# 推理主流程
# --------------------------------------------------------------------------- #
def predict_frame(model, frame: np.ndarray, args, class_ids):
    kwargs = dict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=class_ids,
        device=args.device,
        verbose=False,
    )
    # ultralytics >= 8.4.83 用 quantize 取代 half：16 表示 FP16
    if args.half:
        kwargs["quantize"] = 16
    return model.predict(**kwargs)[0]


def run_video(model, source: Path, out_dir: Path, args, class_ids, use_cuda) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps_in <= 0 or fps_in > 240:
        fps_in = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = None
    video_path = None
    if not args.no_save_video:
        video_path = out_dir / "annotated.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频：{video_path}")

    frames_json: list[dict] = []
    class_counter: Counter = Counter()
    infer_times: list[float] = []
    total_detections = 0
    idx = 0
    started = False
    wall_start = time.perf_counter()
    warmed = False

    try:
        ok, frame = cap.read()
        while ok:
            if args.max_frames is not None and idx >= args.max_frames:
                break

            # 预热：只做推理，不计时、不写盘
            if not warmed and args.warmup > 0:
                for _ in range(args.warmup):
                    predict_frame(model, frame, args, class_ids)
                cuda_sync(use_cuda)
                if use_cuda:
                    import torch

                    torch.cuda.reset_peak_memory_stats()
                warmed = True
                wall_start = time.perf_counter()

            started = True
            cuda_sync(use_cuda)
            t0 = time.perf_counter()
            result = predict_frame(model, frame, args, class_ids)
            cuda_sync(use_cuda)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_times.append(infer_ms)

            records = boxes_to_records(result, model)
            total_detections += len(records)
            for rec in records:
                class_counter[rec["class_name"]] += 1

            frames_json.append(
                {
                    "frame": idx,
                    "time_sec": round(idx / fps_in, 3),
                    "infer_ms": round(infer_ms, 2),
                    "detections": records,
                }
            )

            if writer is not None:
                writer.write(result.plot(line_width=args.line_width))

            if args.show:
                cv2.imshow("YOLOv8 detect", result.plot(line_width=args.line_width))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            idx += 1
            ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - wall_start
    processed = idx
    infer_total = sum(infer_times) / 1000.0
    vram_alloc, vram_reserved = vram_peak_mb(use_cuda)
    summary = {
        "source": str(source),
        "source_type": "video",
        "width": width,
        "height": height,
        "fps_in": round(fps_in, 3),
        "frames_in_video": frame_count,
        "frames_processed": processed,
        "elapsed_sec": round(elapsed, 3),
        "infer_total_sec": round(infer_total, 3),
        "infer_ms_per_frame": round(infer_total / processed * 1000, 2) if processed else None,
        "infer_fps": round(processed / infer_total, 1) if infer_total > 0 else None,
        "end_to_end_fps": round(processed / elapsed, 1) if elapsed > 0 else None,
        "peak_vram_mb": vram_alloc,
        "peak_vram_reserved_mb": vram_reserved,
        "total_detections": total_detections,
        "class_counts": dict(sorted(class_counter.items(), key=lambda kv: -kv[1])),
        "output_video": str(video_path) if video_path else None,
    }
    return {"summary": summary, "frames": frames_json}


def run_images(model, sources: list[Path], source_type: str, out_dir: Path, args,
               class_ids, use_cuda) -> dict:
    frames_json: list[dict] = []
    class_counter: Counter = Counter()
    infer_times: list[float] = []
    total_detections = 0
    saved: list[str] = []
    limit = args.max_frames if args.max_frames is not None else len(sources)
    selected = sources[:limit]

    # 预热：用第一张图
    first = imread_unicode(selected[0])
    if first is None:
        raise RuntimeError(f"无法读取图片：{selected[0]}")
    for _ in range(max(0, args.warmup)):
        predict_frame(model, first, args, class_ids)
    cuda_sync(use_cuda)
    if use_cuda:
        import torch

        torch.cuda.reset_peak_memory_stats()

    wall_start = time.perf_counter()
    for idx, path in enumerate(selected):
        frame = first if idx == 0 else imread_unicode(path)
        if frame is None:
            print(f"[WARN] 跳过无法读取的图片：{path}")
            continue

        cuda_sync(use_cuda)
        t0 = time.perf_counter()
        result = predict_frame(model, frame, args, class_ids)
        cuda_sync(use_cuda)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        infer_times.append(infer_ms)

        records = boxes_to_records(result, model)
        total_detections += len(records)
        for rec in records:
            class_counter[rec["class_name"]] += 1

        frames_json.append(
            {
                "frame": idx,
                "image": str(path),
                "infer_ms": round(infer_ms, 2),
                "detections": records,
            }
        )

        if not args.no_save_video:
            annotated = result.plot(line_width=args.line_width)
            out_path = out_dir / f"annotated_{path.stem}.jpg"
            if imwrite_unicode(out_path, annotated):
                saved.append(str(out_path))

        if args.show:
            cv2.imshow("YOLOv8 detect", result.plot(line_width=args.line_width))
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    elapsed = time.perf_counter() - wall_start
    processed = len(frames_json)
    infer_total = sum(infer_times) / 1000.0
    vram_alloc, vram_reserved = vram_peak_mb(use_cuda)
    summary = {
        "source": str(sources[0].parent if source_type == "images" else sources[0]),
        "source_type": source_type,
        "images_processed": processed,
        "elapsed_sec": round(elapsed, 3),
        "infer_total_sec": round(infer_total, 3),
        "infer_ms_per_frame": round(infer_total / processed * 1000, 2) if processed else None,
        "infer_fps": round(processed / infer_total, 1) if infer_total > 0 else None,
        "end_to_end_fps": round(processed / elapsed, 1) if elapsed > 0 else None,
        "peak_vram_mb": vram_alloc,
        "peak_vram_reserved_mb": vram_reserved,
        "total_detections": total_detections,
        "class_counts": dict(sorted(class_counter.items(), key=lambda kv: -kv[1])),
        "output_images": saved,
    }
    return {"summary": summary, "frames": frames_json}


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def print_summary(summary: dict, args) -> None:
    print("=" * 64)
    print(f"源        : {summary['source']}")
    if summary["source_type"] == "video":
        print(f"视频信息  : {summary['width']}x{summary['height']} @ {summary['fps_in']} fps, "
              f"共 {summary['frames_in_video']} 帧")
    print(f"模型      : {args.model}  (imgsz={args.imgsz}, conf={args.conf}, "
          f"iou={args.iou}, device={args.device or 'auto'}, half={args.half})")
    print(f"处理数量  : {summary.get('frames_processed', summary.get('images_processed'))}")
    print(f"纯推理    : {summary['infer_ms_per_frame']} ms/帧 -> {summary['infer_fps']} FPS")
    print(f"端到端    : {summary['end_to_end_fps']} FPS（含解码/画框/写盘）")
    if summary["peak_vram_mb"] is not None:
        print(f"显存峰值  : {summary['peak_vram_mb']} MB 分配 / "
              f"{summary['peak_vram_reserved_mb']} MB 保留")
    print(f"检测目标  : {summary['total_detections']} 个 {summary['class_counts']}")
    if summary.get("output_video"):
        print(f"输出视频  : {summary['output_video']}")
    if summary.get("output_images"):
        print(f"输出图片  : {len(summary['output_images'])} 张 -> {Path(summary['output_images'][0]).parent}")
    print("=" * 64)
    if summary["infer_fps"] is not None and summary["infer_fps"] < 15:
        print("[WARN] 纯推理 FPS < 15，低于阶段 1 验收线；可尝试 --half 或降低 --imgsz。")
    elif summary["infer_fps"] is not None:
        print(f"[OK] 纯推理 FPS {summary['infer_fps']} >= 15，达到阶段 1 验收线。")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        print(f"[ERROR] 源不存在：{source}")
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        print(f"[ERROR] 未安装 ultralytics：{exc}\n请先运行 scripts/check_env.py 检查环境。")
        return 3

    source_type, files = collect_sources(source)
    stem = source.stem if source_type in ("video", "image") else source.name
    out_dir = Path(args.output).resolve() if args.output else Path("runs/detect") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    try:
        class_ids = resolve_classes(model, args.classes)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    use_cuda = False
    try:
        import torch

        use_cuda = torch.cuda.is_available() and str(args.device).lower() != "cpu"
    except ImportError:
        pass
    if args.half and not use_cuda:
        print("[WARN] --half 只在 CUDA 下生效，已忽略。")

    print(f"[INFO] 加载模型 {args.model}，设备 {'CUDA' if use_cuda else 'CPU'}，"
          f"输入 {source}（{source_type}）")
    try:
        if source_type == "video":
            payload = run_video(model, files[0], out_dir, args, class_ids, use_cuda)
        else:
            payload = run_images(model, files, source_type, out_dir, args, class_ids, use_cuda)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 2

    payload["meta"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "classes": args.classes or "all",
        "source": str(source),
        "half": bool(args.half),
        "device": args.device or ("cuda" if use_cuda else "cpu"),
    }

    print_summary(payload["summary"], args)

    if not args.no_json:
        json_path = out_dir / "detections.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[INFO] 检测结果 JSON：{json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())