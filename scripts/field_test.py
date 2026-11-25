# -*- coding: utf-8 -*-
"""小车实拍现场测试：批量采集并统计识别效果，自动生成报告。

两种用法（在车上，与 best.onnx / corn-clean.yaml / patrol_detect.py 同目录）：
  # 1) 摄像头实时采集一轮
  python3 field_test.py --source 0 --frames 100 --label indoor --outdir ./field

  # 2) 用已有图片目录代替摄像头（无摄像头时先测模型，或手机拍完传上来）
  python3 field_test.py --source ./imgs --label photos --outdir ./field

输出（每个 label 一个子目录）：
  detections.jsonl   每帧的检测明细，一行一帧
  summary.json       本轮汇总指标
  contact_sheet.jpg  缩略图拼版，一眼看整体效果
  frames/            若干张标注图（默认最多存 30 张）

用 --compare 可以扫多个 conf 阈值，一次看出该用哪个：
  python3 field_test.py --source 0 --frames 100 --conf 0.15 0.25 0.40
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patrol_detect as pd


def parse_args():
    p = argparse.ArgumentParser(description="小车实拍现场测试")
    p.add_argument("--model", default="best.onnx", help="ONNX 模型路径")
    p.add_argument("--config", default="corn-clean.yaml", help="数据 yaml")
    p.add_argument("--source", default="0", help="摄像头编号或图片目录")
    p.add_argument("--frames", type=int, default=100, help="采集帧数上限")
    p.add_argument("--label", default="run", help="本轮名称，作为输出子目录名")
    p.add_argument("--outdir", default="./field", help="输出根目录")
    p.add_argument("--conf", type=float, nargs="+", default=[0.25], help="置信度阈值，可给多个做对比")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--providers", default="auto", choices=["auto", "cpu", "cuda", "trt"])
    p.add_argument("--save-max", type=int, default=30, help="最多保存多少张标注图")
    p.add_argument("--warmup", type=int, default=3, help="预热帧数，不计入统计")
    return p.parse_args()


def collect_frames(source, frames, warmup):
    """按来源采集帧，yield (frame_index, BGR 图)。来源可以是摄像头编号或图片目录。"""
    if source.isdigit():
        cam = int(source)
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            print(f"[错误] 无法打开摄像头 {cam}")
            print("       ls /dev/video* 看设备号；或改用 --source ./imgs 指定图片目录")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[信息] 摄像头 {cam} 已打开，分辨率 {w}x{h}")
        idx, total = 0, 0
        try:
            while total < frames:
                ok, frame = cap.read()
                if not ok:
                    print("[警告] 读帧失败，提前结束")
                    break
                idx += 1
                if idx <= warmup:
                    continue
                total += 1
                yield total, frame
        finally:
            cap.release()
    else:
        d = Path(source)
        if not d.is_dir():
            print(f"[错误] --source {source} 既不是摄像头编号也不是图片目录")
            return
        files = sorted(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png")))
        if not files:
            print(f"[错误] 目录 {d} 下没有 jpg/png 图片")
            return
        print(f"[信息] 图片目录 {d}，共 {len(files)} 张")
        for i, f in enumerate(files[:frames], 1):
            img = cv2.imread(str(f))
            if img is None:
                print(f"[警告] 读不到 {f.name}，跳过")
                continue
            yield i, img


def summarize(dets_per_frame, costs, names, conf_thr):
    """统计一轮结果：有检出的帧占比、各类计数、置信度分布、耗时。"""
    n_frame = len(dets_per_frame)
    hit_frames = sum(1 for d in dets_per_frame if d)
    per_cls = {n: 0 for n in names}
    confs = []
    for d in dets_per_frame:
        for x in d:
            per_cls[names[int(x["cls"])]] += 1
            confs.append(x["conf"])
    total = sum(per_cls.values())
    return {
        "conf_threshold": round(conf_thr, 3),
        "iou_threshold": None,
        "frames": n_frame,
        "frames_with_detection": hit_frames,
        "frame_hit_rate": round(hit_frames / n_frame, 4) if n_frame else 0.0,
        "total_detections": total,
        "detections_per_frame": round(total / n_frame, 3) if n_frame else 0.0,
        "per_class": per_cls,
        "empty_frames": n_frame - hit_frames,
        "conf_min": round(min(confs), 4) if confs else None,
        "conf_mean": round(float(np.mean(confs)), 4) if confs else None,
        "conf_max": round(max(confs), 4) if confs else None,
        "infer_ms_mean": round(float(np.mean(costs)), 1) if costs else None,
        "infer_ms_p95": round(float(np.percentile(costs, 95)), 1) if costs else None,
    }


def contact_sheet(frames_for_sheet, cols=5, thumb=200):
    """把若干帧缩略图拼成一张，快速看整体效果。"""
    if not frames_for_sheet:
        return None
    rows = (len(frames_for_sheet) + cols - 1) // cols
    sheet = np.full((rows * thumb, cols * thumb, 3), 30, np.uint8)
    for i, f in enumerate(frames_for_sheet):
        r, c = divmod(i, cols)
        small = cv2.resize(f, (thumb, thumb), interpolation=cv2.INTER_AREA)
        sheet[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = small
    return sheet


def main():
    args = parse_args()
    nc, names = pd.load_config(args.config)
    print(f"[信息] 类别数 {nc}，类别名 {names}")
    sess = pd.make_session(args.model, args.providers)
    inp = sess.get_inputs()[0].name
    print(f"[信息] 输入 {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"[信息] 输出 {[(o.name, o.shape) for o in sess.get_outputs()]}")

    root = Path(args.outdir) / args.label
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 所有 conf 共用同一批帧，避免不同阈值之间输入不一致
    print(f"\n[信息] 开始采集 {args.frames} 帧（预热 {args.warmup} 帧不计入）...")
    raw = []
    for i, frame in collect_frames(args.source, args.frames, args.warmup):
        blob, meta = pd.preprocess(frame, args.imgsz)
        t0 = time.perf_counter()
        out = sess.run(None, {inp: blob})
        cost = (time.perf_counter() - t0) * 1000
        raw.append((i, frame, out[0], meta, cost))
        if i % 20 == 0:
            print(f"  已处理 {i}/{args.frames} 帧，最近一帧 {cost:.1f} ms")

    if not raw:
        print("[错误] 没有采集到任何帧")
        return 2
    print(f"[信息] 采集完成，共 {len(raw)} 帧")

    results = {}
    sheet_frames = []
    for conf_thr in args.conf:
        dets_frames, costs = [], []
        for i, frame, out, meta, cost in raw:
            h, w = frame.shape[:2]
            r = pd.postprocess(out, meta, h, w, conf_thr, args.iou)
            dets_frames.append([{"cls": int(x[5]), "conf": round(x[4], 4),
                                 "x": int(x[0]), "y": int(x[1]),
                                 "w": int(x[2] - x[0]), "h": int(x[3] - x[1])} for x in r])
            costs.append(cost)
        s = summarize(dets_frames, costs, names, conf_thr)
        s["iou_threshold"] = args.iou
        results[f"conf_{conf_thr}"] = s

        # 明细逐帧落盘
        with open(root / f"detections_conf{conf_thr}.jsonl", "w", encoding="utf-8") as f:
            for (i, frame, out, meta, cost), dl in zip(raw, dets_frames):
                h, w = frame.shape[:2]
                f.write(json.dumps({"frame": i, "size": [w, h], "infer_ms": round(cost, 1),
                                    "detections": dl}, ensure_ascii=False) + "\n")

        # 只对第一个阈值存图，避免重复
        if conf_thr == args.conf[0]:
            for k, ((i, frame, out, meta, cost), dl) in enumerate(zip(raw, dets_frames)):
                r = [[x["x"], x["y"], x["x"] + x["w"], x["y"] + x["h"], x["conf"], x["cls"]] for x in dl]
                vis = pd.draw(frame.copy(), r, names)
                cv2.putText(vis, f"#{i} {len(dl)} det  {cost:.0f}ms", (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if k < args.save_max:
                    cv2.imwrite(str(frames_dir / f"frame_{i:05d}.jpg"), vis)
                if k % max(1, len(raw) // 25) == 0:
                    sheet_frames.append(vis)

    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "source": args.source, "imgsz": args.imgsz,
                   "model": str(args.model), "results": results}, f,
                  ensure_ascii=False, indent=2)

    sheet = contact_sheet(sheet_frames)
    if sheet is not None:
        cv2.imwrite(str(root / "contact_sheet.jpg"), sheet)

    print("\n" + "=" * 66)
    print(f"本轮 {args.label} 汇总（{len(raw)} 帧）")
    print("=" * 66)
    for key, s in results.items():
        print(f"\n[conf {s['conf_threshold']}]")
        print(f"  有检出帧   {s['frames_with_detection']}/{s['frames']}"
              f"  ({s['frame_hit_rate']*100:.1f}%)   空帧 {s['empty_frames']}")
        print(f"  检出总数   {s['total_detections']}  ({s['detections_per_frame']} 个/帧)")
        print(f"  分类计数   {s['per_class']}")
        if s["conf_mean"] is not None:
            print(f"  置信度     min {s['conf_min']}  mean {s['conf_mean']}  max {s['conf_max']}")
        print(f"  推理耗时   均值 {s['infer_ms_mean']} ms  P95 {s['infer_ms_p95']} ms"
              f"  ≈ {1000/s['infer_ms_mean']:.1f} FPS" if s["infer_ms_mean"] else "")
    print(f"\n结果目录 {root.resolve()}")
    print("  summary.json / detections_conf*.jsonl / contact_sheet.jpg / frames/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
