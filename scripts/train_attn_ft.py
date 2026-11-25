# 注意力分支微调脚本：加载基线权重，只让CBAM随机初始化，保证公平对比
# 输出：runs/corn-11n-attn-ft，不覆盖 baseline3 与 attn
# 使用方法（在 PyCharm 的 yolo 环境中运行）：
#   python scripts/train_attn_ft.py --data configs/corn-clean.yaml --model configs/yolo11n-cbam.yaml --pretrained runs/corn-baseline3/weights/best.pt --epochs 30 --batch 8 --device 0
import argparse
import sys
from pathlib import Path

# 保证 scripts 下运行也能 import models
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    """解析微调参数，epochs 建议 20-30，lr 建议更小。"""
    p = argparse.ArgumentParser(description="YOLO11n-CBAM公平微调")
    p.add_argument("--data", type=str, default="configs/corn-clean.yaml", help="数据yaml")
    p.add_argument("--model", type=str, default="configs/yolo11n-cbam.yaml", help="CBAM结构yaml")
    p.add_argument("--pretrained", type=str, default="runs/corn-baseline3/weights/best.pt", help="基线best.pt")
    p.add_argument("--imgsz", type=int, default=416, help="输入尺寸，与基线一致")
    p.add_argument("--epochs", type=int, default=30, help="微调轮数，建议20-30")
    p.add_argument("--batch", type=int, default=8, help="批次")
    p.add_argument("--device", type=str, default="0", help="设备")
    p.add_argument("--name", type=str, default="corn-11n-attn-ft", help="输出名，禁止用旧名")
    p.add_argument("--workers", type=int, default=0, help="Windows建议0")
    p.add_argument("--lr0", type=float, default=0.001, help="微调初始学习率，比从零小10倍")
    p.add_argument("--patience", type=int, default=15, help="早停")
    return p.parse_args()


def main():
    args = parse_args()
    # 安全锁：禁止覆盖已有两个目录
    if args.name in ("corn-baseline3", "corn-11n-attn"):
        print("[错误] --name 不允许覆盖已有目录，请用 corn-11n-attn-ft。")
        return
    if not Path(args.data).exists():
        print(f"[错误] 找不到数据yaml: {args.data}")
        return
    if not Path(args.model).exists():
        print(f"[错误] 找不到结构yaml: {args.model}")
        return
    if not Path(args.pretrained).exists():
        print(f"[错误] 找不到基线权重: {args.pretrained}")
        return
    try:
        import torch
        from models.cbam import CBAM
        import ultralytics.nn.modules as ULM
        import ultralytics.nn.tasks as ULT
        setattr(ULM, "CBAM", CBAM)
        for mod in (ULT, ULM):
            try:
                setattr(mod, "CBAM", CBAM)
            except Exception:
                pass
        from ultralytics import YOLO
        from ultralytics.nn.tasks import torch_safe_load
    except ImportError as e:
        print(f"[错误] 缺依赖: {e}")
        return
    print(f"结构: {args.model}，预训练: {args.pretrained}，输出: runs/{args.name}")
    # 1. 按 CBAM 结构建新模型（CBAM 随机初始化）
    model = YOLO(args.model)
    # 2. 加载基线权重，只拷贝形状相同的层，CBAM 层自动跳过保持随机初始化
    ckpt, _ = torch_safe_load(args.pretrained)
    src = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    src_sd = src.float().state_dict() if hasattr(src, "state_dict") else src
    dst_sd = model.model.state_dict()
    matched, skipped = 0, []
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k].copy_(v)
            matched += 1
        else:
            skipped.append(k)
    # 统计 CBAM 是否保持随机初始化
    cbam_keys = [k for k in dst_sd if "cbam" in k.lower() or "channel_attention" in k or "spatial_attention" in k]
    print(f"已从基线迁移 {matched} 层，跳过 {len(skipped)} 层（含新增CBAM）。")
    print(f"CBAM相关层共 {len(cbam_keys)} 个，保持随机初始化参与微调。")
    # 3. 微调：小学习率，短轮数
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=args.device, project="runs",
        name=args.name, patience=args.patience, workers=args.workers,
        lr0=args.lr0,
    )
    print(f"微调完成: runs/{args.name}，基线与从零版均未改动。")


if __name__ == "__main__":
    main()
