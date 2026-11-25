# -*- coding: utf-8 -*-
"""鲁棒性退化测试：模拟田间恶劣条件，测「退化-精度」曲线。

在笔记本上运行（需要 ultralytics + best.pt）：
  python scripts/robustness_test.py
  python scripts/robustness_test.py --split valid --max-images 300
  python scripts/robustness_test.py --tags blur_k5 bright_m30 occlude_20

做法：对同一批验证图施加一种退化，写临时目录，用官方 val 测 mAP50，
      与 clean（无退化）对比得出下降幅度。临时图片测完即删，只保留汇总。

输出：outputs/robustness/summary.json 与 summary.csv、degradation_curve.png
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="鲁棒性退化测试")
    p.add_argument("--model", default="runs/corn-baseline3/weights/best.pt")
    p.add_argument("--data-root", default="datasets/corn-clean")
    p.add_argument("--split", default="valid", choices=["valid", "test", "train"])
    p.add_argument("--names", default="configs/corn-clean.yaml")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--max-images", type=int, default=0, help="只取前 N 张，0=全部")
    p.add_argument("--tags", nargs="+", default=None, help="只跑指定退化项")
    p.add_argument("--work", default="outputs/robustness")
    p.add_argument("--keep-images", action="store_true", help="保留退化后的图片")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------- 各类退化 ----------------
def deg_none(im, rng):
    return im


def deg_blur(k):
    def f(im, rng):
        return cv2.GaussianBlur(im, (k, k), 0)
    return f


def deg_bright(delta):
    """delta 为相对亮度变化，如 +0.30 / -0.30。"""
    def f(im, rng):
        out = im.astype(np.float32) * (1.0 + delta)
        return np.clip(out, 0, 255).astype(np.uint8)
    return f


def deg_contrast(delta):
    """绕均值缩放，delta 如 +0.20 / -0.20。"""
    def f(im, rng):
        mean = im.mean()
        out = (im.astype(np.float32) - mean) * (1.0 + delta) + mean
        return np.clip(out, 0, 255).astype(np.uint8)
    return f


def deg_noise(sigma):
    """加性高斯噪声，sigma 为噪声强度。"""
    def f(im, rng):
        noise = rng.normal(0, sigma, im.shape)
        return np.clip(im.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return f


def deg_occlude(ratio):
    """随机贴遮挡块，**总覆盖面积**严格控制在 ratio，模拟叶片重叠。

    尺寸上限按剩余待填面积推算，避免单块就超过目标（原实现单块可达
    w/2*h/2=25%，导致 10% 目标实际盖到 17%，把结论带偏）。
    """
    def f(im, rng):
        h, w = im.shape[:2]
        out = im.copy()
        target = ratio * h * w
        filled = 0.0
        guard = 0
        while filled < target and guard < 200:
            guard += 1
            remain = target - filled
            # 单块面积不超过剩余量，边长不超过短边/3
            side_cap = min(h, w) // 3
            max_area = min(remain, side_cap * side_cap)
            if max_area < 16:
                break
            s = int(np.sqrt(max_area * rng.uniform(0.4, 1.0)))
            s = max(8, s)
            bw = min(s, side_cap, w)
            bh = min(int(s * rng.uniform(0.8, 1.25)), side_cap, h)
            if bw * bh > remain:
                bh = max(8, int(remain / bw))
            x = int(rng.integers(0, max(1, w - bw)))
            y = int(rng.integers(0, max(1, h - bh)))
            out[y:y + bh, x:x + bw] = 40          # 暗色遮挡物
            filled += bw * bh
        return out
    return f


def deg_jpeg(quality):
    """先压再解，模拟传输压缩。"""
    def f(im, rng):
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return im
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return f


def deg_far(short_side):
    """先降到短边 short_side 再放大回原尺寸，模拟远距离拍摄。"""
    def f(im, rng):
        h, w = im.shape[:2]
        s = short_side / min(h, w)
        small = cv2.resize(im, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return f


DEGRADATIONS = [
    ("clean", "无退化（基准）", deg_none),
    ("blur_k5", "高斯模糊 k=5（轻微抖动）", deg_blur(5)),
    ("blur_k9", "高斯模糊 k=9（明显抖动）", deg_blur(9)),
    ("bright_p30", "亮度 +30%（强光）", deg_bright(0.30)),
    ("bright_m30", "亮度 -30%（阴影）", deg_bright(-0.30)),
    ("contrast_p20", "对比度 +20%（逆光）", deg_contrast(0.20)),
    ("contrast_m20", "对比度 -20%（灰雾）", deg_contrast(-0.20)),
    ("noise_s10", "高斯噪声 σ=10", deg_noise(10)),
    ("noise_s25", "高斯噪声 σ=25", deg_noise(25)),
    ("occlude_10", "遮挡 10%", deg_occlude(0.10)),
    ("occlude_20", "遮挡 20%", deg_occlude(0.20)),
    ("jpeg_q50", "JPEG 压缩 q=50", deg_jpeg(50)),
    ("far_208", "远距离 降采样至 208", deg_far(208)),
    ("far_156", "远距离 降采样至 156", deg_far(156)),
]


def make_variant(tag, fn, src_imgs, src_labels, dst_dir, seed, max_images):
    """把退化后的图片写到 dst_dir/images，标签软链接/复制到 dst_dir/labels。"""
    img_dir = dst_dir / "images"
    lbl_dir = dst_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    files = src_imgs[:max_images] if max_images else src_imgs
    n = 0
    for f in files:
        im = cv2.imread(str(f))
        if im is None:
            continue
        out = fn(im, rng)
        cv2.imwrite(str(img_dir / f.name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lb = src_labels / (f.stem + ".txt")
        if lb.exists():
            shutil.copy2(lb, lbl_dir / lb.name)
        n += 1
    return n


def write_data_yaml(dst_dir, names):
    cfg = {"path": str(dst_dir.resolve()), "train": "images", "val": "images",
           "nc": len(names), "names": names}
    p = dst_dir / "data.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return p


def run_val(yolo, data_yaml, args):
    r = yolo.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch,
                 device=args.device, plots=False, verbose=False, save_json=False)
    out = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
           "precision": float(r.box.mp), "recall": float(r.box.mr)}
    try:
        idx = list(getattr(r.box, "ap_class_index", []))
        ap50 = list(getattr(r.box, "ap50", []))
        out["ap50_per_class"] = {int(i): round(float(v), 4) for i, v in zip(idx, ap50)}
    except Exception:
        out["ap50_per_class"] = {}
    return out


def plot_curve(rows, names, out_png):
    """画退化-精度曲线：每个退化项一根柱，基准画横线。"""
    items = [r for r in rows if r["tag"] != "clean"]
    if not items:
        return False
    base = next((r["mAP50"] for r in rows if r["tag"] == "clean"), None)
    W, H = max(900, 58 * len(items)), 420
    canvas = np.full((H, W, 3), 255, np.uint8)
    left, right, top, bottom = 60, 20, 30, 110
    plot_h = H - top - bottom
    vmax = 1.0 if base is None else max(1.0, base * 1.1)
    # 基准横线
    if base is not None:
        y = int(top + plot_h * (1 - base / vmax))
        cv2.line(canvas, (left, y), (W - right, y), (0, 140, 0), 2)
        cv2.putText(canvas, f"clean {base:.3f}", (left + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 110, 0), 1)
    cv2.line(canvas, (left, top), (left, top + plot_h), (120, 120, 120), 1)
    cv2.line(canvas, (left, top + plot_h), (W - right, top + plot_h), (120, 120, 120), 1)
    for i in range(11):
        v = vmax * i / 10
        y = int(top + plot_h * (1 - i / 10))
        cv2.line(canvas, (left - 4, y), (left, y), (160, 160, 160), 1)
        cv2.putText(canvas, f"{v:.1f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    slot = (W - left - right) / len(items)
    for i, r in enumerate(items):
        x0 = int(left + i * slot + slot * 0.18)
        x1 = int(left + (i + 1) * slot - slot * 0.18)
        y = int(top + plot_h * (1 - r["mAP50"] / vmax))
        color = (60, 120, 220) if base is None or r["mAP50"] >= base * 0.95 else (60, 60, 200)
        cv2.rectangle(canvas, (x0, y), (x1, top + plot_h), color, -1)
        cv2.putText(canvas, f"{r['mAP50']:.3f}", (x0, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)
        cv2.putText(canvas, r["tag"], (x0 - 2, top + plot_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)
        drop = (r["mAP50"] - base) if base is not None else 0.0
        cv2.putText(canvas, f"{drop:+.3f}", (x0, top + plot_h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 40, 40), 1)
    cv2.putText(canvas, "mAP50 under degradation (green line = clean baseline)",
                (left, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.imwrite(str(out_png), canvas)
    return True


def main():
    args = parse_args()
    from ultralytics import YOLO

    with open(args.names, encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]

    src = Path(args.data_root) / args.split
    src_imgs = sorted([p for ext in ("*.jpg", "*.jpeg", "*.png")
                       for p in (src / "images").glob(ext)])
    src_labels = src / "labels"
    if not src_imgs:
        print(f"[错误] {src / 'images'} 下没有图片")
        return 2
    if args.max_images:
        src_imgs = src_imgs[:args.max_images]
    print(f"[信息] 模型 {args.model}")
    print(f"[信息] 评测集 {src}，{len(src_imgs)} 张图，imgsz {args.imgsz}")

    todos = [d for d in DEGRADATIONS if not args.tags or d[0] in args.tags]
    missing = set(args.tags or []) - {d[0] for d in todos}
    if missing:
        print(f"[错误] 未知退化项 {sorted(missing)}；可选 {[d[0] for d in DEGRADATIONS]}")
        return 2
    if not any(d[0] == "clean" for d in todos):
        todos.insert(0, DEGRADATIONS[0])

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(args.model)

    rows, t_start = [], time.time()
    for tag, desc, fn in todos:
        t0 = time.time()
        dst = work / tag
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        n = make_variant(tag, fn, src_imgs, src_labels, dst, args.seed, 0)
        yaml_p = write_data_yaml(dst, names)
        print(f"\n[{tag}] {desc}  ({n} 张，生成 {time.time()-t0:.0f}s) 评测中...")
        try:
            m = run_val(yolo, yaml_p, args)
        except Exception as e:
            print(f"  [失败] {e}")
            if not args.keep_images:
                shutil.rmtree(dst, ignore_errors=True)
            continue
        row = {"tag": tag, "desc": desc, **m}
        rows.append(row)
        print(f"  mAP50 {m['mAP50']:.4f}  mAP50-95 {m['mAP50_95']:.4f}"
              f"  P {m['precision']:.4f}  R {m['recall']:.4f}  ({time.time()-t0:.0f}s)")
        if m.get("ap50_per_class"):
            print(f"  分类型 AP50 {m['ap50_per_class']}")
        if not args.keep_images:
            shutil.rmtree(dst, ignore_errors=True)

    if not rows:
        print("[错误] 没有任何成功结果")
        return 1

    base = next((r["mAP50"] for r in rows if r["tag"] == "clean"), None)
    for r in rows:
        r["mAP50_drop"] = round(r["mAP50"] - base, 4) if base is not None else None
        r["mAP50_drop_pct"] = (round((r["mAP50"] - base) / base * 100, 2)
                               if base else None)

    summary = {"model": args.model, "split": args.split, "imgsz": args.imgsz,
               "max_images": args.max_images or len(src_imgs),
               "images": len(src_imgs), "seed": args.seed,
               "names": names, "elapsed_s": round(time.time() - t_start, 1),
               "results": rows}
    with open(work / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cols = ["tag", "desc", "mAP50", "mAP50_95", "mAP50_drop", "mAP50_drop_pct",
            "precision", "recall"]
    with open(work / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])

    plotted = plot_curve(rows, names, work / "degradation_curve.png")

    print("\n" + "=" * 74)
    print(f"{'退化项':<14}{'mAP50':>8}{'mAP50-95':>10}{'变化':>9}{'变化%':>9}{'P':>8}{'R':>8}")
    print("-" * 74)
    for r in rows:
        d = f"{r['mAP50_drop']:+.4f}" if r["mAP50_drop"] is not None else "-"
        dp = f"{r['mAP50_drop_pct']:+.1f}%" if r["mAP50_drop_pct"] is not None else "-"
        print(f"{r['tag']:<14}{r['mAP50']:>8.4f}{r['mAP50_95']:>10.4f}"
              f"{d:>9}{dp:>9}{r['precision']:>8.4f}{r['recall']:>8.4f}")
    print("=" * 74)
    print(f"结果目录 {work.resolve()}")
    print(f"  summary.json / summary.csv / degradation_curve.png"
          f"{'  （曲线已生成）' if plotted else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
