"""
archive_server_results.py
=========================
在断开服务器前，读取本地实验目录的 results.csv / weights/ / plots/，
输出完整的训练报告并保存到 outputs/archive/<name>/。

用法（在 E:\qiu 目录下运行）：
    python scripts/archive_server_results.py --exp corn-11n-cbam-neck-4 [--baseline corn-baseline3]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ── 全局路径 ──────────────────────────────────────────────────────────────────
EQUIP_ROOT = Path(r"E:\qiu")                         # 服务器根目录，可按需改
ARCHIVE_DIR = EQUIP_ROOT / "outputs" / "archive"     # 归档输出根目录


def parse_args():
    p = argparse.ArgumentParser(description="收集服务器实验结果并归档")
    p.add_argument("--exp", required=True, help="实验目录名，如 corn-11n-cbam-neck-4")
    p.add_argument("--baseline", default=None,
                   help="基线目录名，如 corn-baseline3（可选）")
    return p.parse_args()


def locate(exp_name: str, baseline_name: str | None) -> dict:
    """定位实验和基线的权重 / CSV / 绘图路径。"""
    runs_root = EQUIP_ROOT / "runs" / "detect" / "runs"
    result = {"exp": {}, "baseline": {}}
    for key, name in [("exp", exp_name), ("baseline", baseline_name)]:
        d = runs_root / name
        if not d.exists():
            print(f"[WARN] 目录不存在: {d}", file=sys.stderr)
            continue
        weights = d / "weights"
        result[key] = {
            "best_pt":  weights / "best.pt"  if (weights / "best.pt").exists()  else None,
            "last_pt":  weights / "last.pt"  if (weights / "last.pt").exists()  else None,
            "results_csv": d / "results.csv" if (d / "results.csv").exists()     else None,
            "conf_matrix_png": d / "plots" / "confusion_matrix.png" if (d / "plots" / "confusion_matrix.png").exists() else None,
            "results_png": d / "plots" / "results.png" if (d / "results.png").exists() else None,
            "args_yaml": d / "args.yaml" if (d / "args.yaml").exists() else None,
        }
    return result


def read_csv(path: Path) -> list[dict] | None:
    if path is None or not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_metric(rows: list[dict], key: str) -> float | None:
    """从 results.csv 的行中提取某个指标（取最后一个 epoch 的值）。"""
    if not rows:
        return None
    last = rows[-1]
    val = last.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def extract_per_class(rows: list[dict]) -> list[dict] | None:
    """提取最后一行的 per-class 指标（Box/P/D/R/mAP50/mAP50-95）。"""
    if not rows:
        return None
    last = rows[-1]
    cols = list(last.keys())
    # 常见列名：Class, Images, Instances, Box(P, R, mAP50, mAP50-95), ...
    classes = []
    for row in rows:
        cls_name = row.get("Class", "").strip()
        if cls_name and cls_name != "all" and cls_name != "":
            classes.append({
                "name": cls_name,
                "images":  row.get("Images", ""),
                "instances": row.get("Instances", ""),
                "P":       float(row.get("Box(P",  "0") or 0),
                "R":       float(row.get("R",      "0") or 0),
                "mAP50":   float(row.get("mAP50",  "0") or 0),
                "mAP50_95":float(row.get("mAP50-95","0") or 0),
            })
    return classes if classes else None


def parse_args_yaml(path: Path | None) -> dict:
    """极简解析 args.yaml，提取关键训练参数。"""
    if path is None or not path.exists():
        return {}
    cfg = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^(\w+)\s*:\s*(.+)$", line.strip())
            if m:
                cfg[m.group(1)] = m.group(2).strip().strip('"\'')
    return cfg


def build_report(exp_data: dict, baseline_data: dict | None) -> str:
    """组装 Markdown 报告。"""
    lines = []
    lines.append("# 玉米病害识别 — 实验结果归档")
    lines.append("")
    lines.append("> 生成时间: " + __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")

    # ── 实验配置 ──
    lines.append("## 实验配置")
    lines.append("")
    exp_cfg = parse_args_yaml(exp_data["args_yaml"])
    bl_cfg  = parse_args_yaml(baseline_data["args_yaml"]) if baseline_data else {}
    lines.append("| 参数 | 本实验 | 基线 |")
    lines.append("|---|---|---|")
    for k in ("model", "data", "imgsz", "epochs", "batch", "device", "patience", "lr0", "optimizer"):
        ev = exp_cfg.get(k, "-")
        bv = bl_cfg.get(k, "-")
        lines.append(f"| {k} | {ev} | {bv} |")
    lines.append("")

    # ── 权重文件 ──
    lines.append("## 权重文件")
    lines.append("")
    if exp_data["best_pt"]:
        sz = exp_data["best_pt"].stat().st_size // 1024
        lines.append(f"- `best.pt` ({sz} KB) → `{exp_data['best_pt']}`")
    if exp_data["last_pt"]:
        sz = exp_data["last_pt"].stat().st_size // 1024
        lines.append(f"- `last.pt` ({sz} KB) → `{exp_data['last_pt']}`")
    lines.append("")

    # ── 整体指标对比 ──
    lines.append("## 整体指标对比")
    lines.append("")
    lines.append("| 指标 | 本实验 | 基线 | 提升 |")
    lines.append("|---|---|---|---|")
    exp_rows = read_csv(exp_data["results_csv"])
    bl_rows  = read_csv(baseline_data["results_csv"]) if baseline_data else None
    for metric_key, label in [
        ("metrics/mAP50-50(%)",  "mAP50"),
        ("metrics/mAP50-95(%)", "mAP50-95"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)",    "Recall"),
    ]:
        ev = extract_metric(exp_rows, metric_key)
        bv = extract_metric(bl_rows, metric_key) if bl_rows else None
        if ev is not None:
            diff = f"+{(ev - bv) * 100:.1f} pp" if bv is not None and bv != 0 else "-"
            bv_str = f"{bv * 100:.1f}" if bv is not None else "-"
            lines.append(f"| {label} | **{ev * 100:.1f}** | {bv_str} | {diff} |")
    lines.append("")

    # ── 每类指标 ──
    lines.append("## 每类指标")
    lines.append("")
    exp_cls = extract_per_class(exp_rows)
    bl_cls  = extract_per_class(bl_rows) if bl_rows else None
    if exp_cls:
        lines.append("| 类别 | 实验 P | 实验 R | 实验 mAP50 | 实验 mAP50-95 | 基线 mAP50 | 提升 |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, ec in enumerate(exp_cls):
            bc = bl_cls[i] if bl_cls and i < len(bl_cls) else None
            diff = f"+{(ec['mAP50'] - bc['mAP50']) * 100:.1f}" if bc else "-"
            lines.append(
                f"| {ec['name']} | {ec['P']*100:.1f} | {ec['R']*100:.1f} | "
                f"**{ec['mAP50']*100:.1f}** | {ec['mAP50_95']*100:.1f} | "
                f"{bc['mAP50']*100:.1f if bc else '-'} | {diff} |"
            )
    lines.append("")

    # ── 早停 / 轮次 ──
    lines.append("## 训练轮次")
    lines.append("")
    epochs_done = len(exp_rows) if exp_rows else 0
    epochs_config = exp_cfg.get("epochs", "?")
    lines.append(f"- 配置 epochs: {epochs_config}")
    lines.append(f"- 实际完成: {epochs_done} 轮（早停）")
    lines.append("")

    # ── 绘图文件 ──
    lines.append("## 绘图文件")
    lines.append("")
    for label, p in [("confusion_matrix.png", exp_data.get("conf_matrix_png")),
                      ("results.png",           exp_data.get("results_png"))]:
        if p and p.exists():
            sz = p.stat().st_size // 1024
            lines.append(f"- `{label}` ({sz} KB) → `{p}`")
    lines.append("")

    # ── 归档清单 ──
    lines.append("## 归档清单")
    lines.append("")
    lines.append("将以下内容复制到 `E:\\qiu\\outputs\\archive\\<exp_name>\\`：")
    if exp_data["best_pt"]:
        lines.append(f"- `best.pt`  ({exp_data['best_pt'].stat().st_size} bytes)")
    if exp_data["last_pt"]:
        lines.append(f"- `last.pt`  ({exp_data['last_pt'].stat().st_size} bytes)")
    if exp_data["results_csv"]:
        lines.append(f"- `results.csv`")
    if exp_data["conf_matrix_png"]:
        lines.append(f"- `plots/confusion_matrix.png`")
    if exp_data["results_png"]:
        lines.append(f"- `plots/results.png`")
    if exp_data["args_yaml"]:
        lines.append(f"- `args.yaml`")
    lines.append("- 本报告 `results_summary.md`")
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    found = locate(args.exp, args.baseline)

    if not found["exp"]["results_csv"] and not found["exp"]["best_pt"]:
        print(f"[ERROR] 未找到实验 '{args.exp}' 的有效文件，请检查目录名。", file=sys.stderr)
        sys.exit(1)

    report = build_report(found["exp"], found["baseline"])

    # 写入归档目录
    out_dir = ARCHIVE_DIR / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_summary.md").write_text(report, encoding="utf-8")

    # 复制权重
    if found["exp"]["best_pt"] and (out_dir / "best.pt").exists() is False:
        import shutil
        shutil.copy2(found["exp"]["best_pt"], out_dir / "best.pt")
        print(f"[COPY] best.pt → {out_dir / 'best.pt'}")
    if found["exp"]["last_pt"] and (out_dir / "last.pt").exists() is False:
        import shutil
        shutil.copy2(found["exp"]["last_pt"], out_dir / "last.pt")
        print(f"[COPY] last.pt → {out_dir / 'last.pt'}")

    # 复制结果 CSV
    if found["exp"]["results_csv"]:
        import shutil
        shutil.copy2(found["exp"]["results_csv"], out_dir / "results.csv")
        print(f"[COPY] results.csv → {out_dir / 'results.csv'}")

    # 复制绘图
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for src, dst_name in [
        (found["exp"].get("conf_matrix_png"), "confusion_matrix.png"),
        (found["exp"].get("results_png"),     "results.png"),
    ]:
        if src and src.exists():
            shutil.copy2(src, plots_dir / dst_name)
            print(f"[COPY] {dst_name} → {plots_dir / dst_name}")

    print()
    print("=" * 60)
    print("报告已保存至:")
    print(f"  {out_dir / 'results_summary.md'}")
    print("=" * 60)
    print()
    print(report)


if __name__ == "__main__":
    main()
