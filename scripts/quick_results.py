"""
quick_results.py
================
在断开服务器前快速提取并打印关键训练结果。

用法（在 E:\qiu 目录下运行）：
    python scripts/quick_results.py --exp corn-11n-cbam-neck-4
"""
import argparse
import csv
import sys
from pathlib import Path

EQUIP_ROOT = Path(r"E:\qiu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    return p.parse_args()


def main():
    exp = EQUIP_ROOT / "runs" / "detect" / "runs" / sys.argv[1] if len(sys.argv) > 1 else None
    if not exp:
        exp = EQUIP_ROOT / "runs" / "detect" / "runs" / parse_args().exp

    csv_path = exp / "results.csv"
    weights = exp / "weights"
    plots = exp / "plots"

    print(f"实验目录: {exp}")
    print(f"results.csv: {'✓' if csv_path.exists() else '✗'}")
    print(f"weights/best.pt: {'✓' if (weights/'best.pt').exists() else '✗'}")
    print(f"weights/last.pt: {'✓' if (weights/'last.pt').exists() else '✗'}")
    print(f"plots/confusion_matrix.png: {'✓' if (plots/'confusion_matrix.png').exists() else '✗'}")
    print()

    if not csv_path.exists():
        print("[ERROR] results.csv 不存在")
        return

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        print("[ERROR] CSV 为空")
        return

    last = rows[-1]
    print("=== 整体指标（最后一轮） ===")
    print(f"Epochs completed: {len(rows)}")
    for k in ("metrics/mAP50-50(%)", "metrics/mAP50-95(%)", "metrics/precision(B)", "metrics/recall(B)"):
        v = last.get(k, "-")
        print(f"  {k}: {v}")

    print("\n=== 每类指标 ===")
    print("| Class | P | R | mAP50 | mAP50-95 |")
    print("|---|---|---|---|---|")
    for row in rows:
        cls = row.get("Class", "").strip()
        if cls and cls not in ("all", ""):
            print(f"| {cls} | {row.get('Box(P', '-')} | {row.get('R', '-')} | {row.get('mAP50', '-')} | {row.get('mAP50-95', '-')} |")

    print("\n=== 权重文件列表 ===")
    if weights.exists():
        for f in sorted(weights.iterdir()):
            if f.is_file():
                print(f"  {f.name}: {f.stat().st_size} bytes")
    else:
        print("  目录不存在")

    print("\n=== 绘图文件列表 ===")
    if plots.exists():
        for f in sorted(plots.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(plots)}: {f.stat().st_size} bytes")
    else:
        print("  目录不存在")

    # 建议的归档命令
    print("\n=== 建议归档命令 ===")
    archive_dir = EQUIP_ROOT / "outputs" / "archive" / sys.argv[1] if len(sys.argv) > 1 else parse_args().exp
    print(f"mkdir -p {archive_dir}")
    if (weights/'best.pt').exists():
        print(f"cp {weights/'best.pt'} {archive_dir}/")
    if (weights/'last.pt').exists():
        print(f"cp {weights/'last.pt'} {archive_dir}/")
    print(f"cp {csv_path} {archive_dir}/")
    if (plots/'confusion_matrix.png').exists():
        print(f"cp {plots/'confusion_matrix.png'} {archive_dir}/")
    if (plots/'results.png').exists():
        print(f"cp {plots/'results.png'} {archive_dir}/")
    print(f"cp {exp/'args.yaml'} {archive_dir}/")


if __name__ == "__main__":
    main()
