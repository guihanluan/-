# 注意力公平训练脚本 v2：按层顺序迁移预训练权重，再 100 轮完整训练
# 与 train_corn.py 同策略（epochs 100、patience 50），只改结构，保证公平对比
# 使用方法（在 yolo 环境中运行）：
#   python scripts/train_attn_v2.py --arch configs/yolo11n-cbam-neck.yaml --name corn-11n-cbam-neck --epochs 100 --batch 8 --device 0 --workers 0 --patience 50
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROTECTED_NAMES = (
    "corn-baseline3", "corn-11n-attn", "corn-11n-attn-ft",
    "corn-11n-attn-single", "corn-v8n-416", "corn-v8n-416b",
    "corn-11s-416", "corn-11s-416b",
)


def parse_args():
    """解析参数，默认 100 轮 + patience 50，与基线规则一致。"""
    p = argparse.ArgumentParser(description="注意力公平训练v2")
    p.add_argument("--arch", type=str, required=True, help="注意力结构yaml，如configs/yolo11n-cbam-neck.yaml")
    p.add_argument("--data", type=str, default="configs/corn-clean.yaml", help="数据yaml")
    p.add_argument("--source", type=str, default="yolo11n.pt", help="预训练来源，可用yolo11n.pt或基线best.pt")
    p.add_argument("--name", type=str, required=True, help="输出名，禁止覆盖已有目录")
    p.add_argument("--imgsz", type=int, default=416, help="输入尺寸")
    p.add_argument("--epochs", type=int, default=100, help="轮数")
    p.add_argument("--batch", type=int, default=8, help="批次")
    p.add_argument("--device", type=str, default="0", help="设备")
    p.add_argument("--workers", type=int, default=0, help="Windows建议0")
    p.add_argument("--patience", type=int, default=50, help="早停")
    p.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    p.add_argument("--freeze", type=int, default=None, help="冻结前N层，可选")
    p.add_argument("--weak-aug", action="store_true", help="弱增强：mosaic减半并关闭强扰动")
    return p.parse_args()


def main():
    args = parse_args()
    if args.name in PROTECTED_NAMES:
        print(f"[错误] --name={args.name} 受保护，请换新名，避免覆盖旧实验。")
        return
    for key in ("arch", "data", "source"):
        if not Path(getattr(args, key)).exists():
            print(f"[错误] 找不到文件: {getattr(args, key)}")
            return
    from attn_utils import register_custom_modules, save_as_checkpoint, transfer_by_order
    from ultralytics import YOLO

    register_custom_modules()
    # 1. 按注意力结构建新模型（注意力层随机初始化）
    print(f"结构: {args.arch}，预训练来源: {args.source}")
    dst = YOLO(args.arch)
    # 2. 加载源权重，按层顺序迁移，跳过新增注意力层
    src = YOLO(args.source)
    copied, skipped, mismatched = transfer_by_order(src.model, dst.model)
    print(f"源 {args.source} -> 新结构：复制 {copied} 层，注意力 {skipped} 个保持随机。")
    # 3. 存为检查点再加载，保证 train() 不丢弃已迁移权重
    init_pt = Path("outputs/init_weights") / (Path(args.arch).stem + "-init.pt")
    save_as_checkpoint(dst, init_pt)
    print(f"已存初始检查点: {init_pt}")
    model = YOLO(str(init_pt))
    # 4. 完整训练，与基线同规则
    train_kwargs = dict(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=args.device, project="runs",
        name=args.name, patience=args.patience, workers=args.workers,
        lr0=args.lr0,
    )
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.weak_aug:
        train_kwargs.update(dict(mosaic=0.5, mixup=0.0, copy_paste=0.0, erasing=0.0, scale=0.3, degrees=0.0))
    model.train(**train_kwargs)
    print(f"训练完成: runs/{args.name}，旧实验目录未改动。")


if __name__ == "__main__":
    main()
