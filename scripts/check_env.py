"""环境自检：打印 Python / PyTorch / CUDA / 依赖版本，并尝试加载 YOLOv8n。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe scripts\\check_env.py
"""
import platform
import sys


def main() -> int:
    print("=" * 56)
    print("Python :", sys.version.split()[0], f"({platform.architecture()[0]})")
    print("系统   :", platform.platform())

    try:
        import torch
        import torchvision
    except ImportError as e:
        print("[FAIL] 无法导入 torch / torchvision:", e)
        return 1

    print("torch  :", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("CUDA 编译版本:", torch.version.cuda)
    print("cuDNN  :", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU    :", torch.cuda.get_device_name(0))
        print(f"显存   : {props.total_memory / 1024**3:.1f} GB")
        print("算力   :", f"sm_{props.major}{props.minor}")
        print("CUDA 可用: True")
    else:
        print("[WARN] CUDA 不可用，当前只能跑 CPU")

    print("-" * 56)
    for name in ("ultralytics", "cv2", "fastapi", "uvicorn", "sqlalchemy", "numpy"):
        try:
            mod = __import__(name)
            print(f"{name:<12}", getattr(mod, "__version__", "unknown"))
        except Exception as e:  # noqa: BLE001
            print(f"{name:<12} [FAIL] {e}")

    print("-" * 56)
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        print("YOLOv8n 加载成功，类别数:", len(model.names))
    except Exception as e:  # noqa: BLE001
        print("[WARN] YOLOv8n 加载失败（可能是权重未下载）:", e)

    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())