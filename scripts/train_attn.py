# 注意力分支训练脚本：YOLO11n-CBAM，与基线完全隔离
# 输出目录固定为 runs/corn-11n-attn，不会覆盖 runs/corn-baseline3
# 使用方法（在 PyCharm 的 yolo 环境中运行）：
#   python scripts/train_attn.py --data configs/corn-clean.yaml --model configs/yolo11n-cbam.yaml --epochs 100 --batch 8 --device 0
import argparse
import sys
from pathlib import Path

# 把工程根目录加入路径，保证 scripts 下运行时也能 import models
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    """解析参数，默认输出名锁定为 corn-11n-attn，防止误覆盖基线。"""
    p = argparse.ArgumentParser(description="YOLO11n-CBAM注意力分支训练")
    p.add_argument("--data", type=str, default="configs/corn-clean.yaml", help="数据yaml")
    p.add_argument("--model", type=str, default="configs/yolo11n-cbam.yaml", help="CBAM结构yaml")
    p.add_argument("--imgsz", type=int, default=416, help="输入尺寸，与基线一致")
    p.add_argument("--epochs", type=int, default=100, help="训练轮数")
    p.add_argument("--batch", type=int, default=8, help="批次，8GB显存建议8")
    p.add_argument("--device", type=str, default="0", help="设备")
    p.add_argument("--name", type=str, default="corn-11n-attn", help="输出名，请勿改为corn-baseline3")
    p.add_argument("--workers", type=int, default=0, help="Windows建议0")
    p.add_argument("--patience", type=int, default=25, help="早停")
    return p.parse_args()


def main():
    args = parse_args()
    # 安全锁：禁止覆盖基线目录
    if args.name == "corn-baseline3":
        print("[错误] --name 不允许使用 corn-baseline3，请换成 corn-11n-attn。")
        return
    if not Path(args.data).exists():
        print(f"[错误] 找不到数据yaml: {args.data}")
        return
    if not Path(args.model).exists():
        print(f"[错误] 找不到结构yaml: {args.model}")
        return
    try:
        # 注册 CBAM，让 ultralytics 在解析 yaml 时认识 CBAM
        from models.cbam import CBAM
        import ultralytics.nn.modules as ULM
        import ultralytics.nn.tasks as ULT
        setattr(ULM, "CBAM", CBAM)
        # 新版解析用 tasks 模块查表，一并注册
        for mod in (ULT, ULM):
            try:
                setattr(mod, "CBAM", CBAM)
            except Exception:
                pass
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[错误] 缺依赖: {e}")
        return
    print(f"结构: {args.model}，数据: {args.data}，输出: runs/{args.name}")
    model = YOLO(args.model)
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=args.device, project="runs",
        name=args.name, patience=args.patience, workers=args.workers,
    )
    print(f"训练完成: runs/{args.name}，基线 runs/corn-baseline3 未改动。")


if __name__ == "__main__":
    main()
