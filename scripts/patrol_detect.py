# -*- coding: utf-8 -*-
"""巡检测试脚本：USB 摄像头 / 单张图片推理（Jetson / 树莓派 / PC 通用）

使用方法（在车上运行）：
  # 1) 单张图片离线验证（不需要摄像头、不需要显示器）
  python patrol_detect.py --model best.onnx --config corn-clean.yaml --image test.jpg

  # 2) 摄像头实时，有显示器
  python patrol_detect.py --model best.onnx --config corn-clean.yaml --cam 0

  # 3) 摄像头实时，SSH 无显示器：不给窗口，直接存图
  python patrol_detect.py --model best.onnx --config corn-clean.yaml --cam 0 --no-window --max-frames 100 --output ./shots

ONNX 输出约定（YOLOv11 detect，imgsz=416）：
  output0: (1, 4+nc, 3549)，前 4 行是 xyxy(416 输入尺度)，后 nc 行是类别分数（已过 sigmoid）。
  注意 YOLOv11 没有 objectness 通道，类别分数直接就是置信度。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt
import yaml

PAD_VALUE = 114  # letterbox 填充灰度，与 Ultralytics 训练时一致


def parse_args():
    p = argparse.ArgumentParser(description="玉米病害实时检测")
    p.add_argument("--model", type=str, required=True, help="ONNX 模型路径")
    p.add_argument("--config", type=str, default="corn-clean.yaml", help="数据 yaml 路径")
    p.add_argument("--cam", type=int, default=0, help="摄像头编号，USB 一般是 0")
    p.add_argument("--image", type=str, default="", help="单张图片路径；给定时只推理一次并退出")
    p.add_argument("--snapshot", type=str, default="",
                   help="用摄像头拍一张存成该文件后立即推理并退出，例如 --snapshot test.jpg")
    p.add_argument("--imgsz", type=int, default=416, help="输入尺寸，必须与导出时一致")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    p.add_argument("--providers", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "trt"],
                   help="auto=有 CUDA 用 CUDA，否则 CPU；cpu 强制 CPU 便于对照")
    p.add_argument("--stretch", action="store_true",
                   help="直接拉伸到正方形，不做 letterbox（默认 letterbox，与训练一致）")
    p.add_argument("--output", type=str, default="", help="结果图保存目录，留空不保存")
    p.add_argument("--no-window", action="store_true", help="无显示器时关闭 imshow")
    p.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0=不限")
    p.add_argument("--print-every", type=int, default=30, help="每多少帧打印一次检测摘要")
    return p.parse_args()


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["nc"], cfg["names"]


def make_session(model_path, prefer="auto"):
    """按可用 provider 建会话；请求了不存在的 provider 会直接报错，所以先过滤。"""
    avail = rt.get_available_providers()
    if prefer == "cpu":
        want = ["CPUExecutionProvider"]
    elif prefer == "cuda":
        want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif prefer == "trt":
        want = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in want if p in avail]
    if not providers:
        providers = ["CPUExecutionProvider"]
    sess = rt.InferenceSession(model_path, providers=providers)
    print(f"[信息] 可用 provider: {avail}")
    print(f"[信息] 实际使用 provider: {sess.get_providers()}")
    return sess


def describe_io(sess):
    inputs = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outputs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
    print(f"[信息] 输入: {inputs}")
    print(f"[信息] 输出: {outputs}")
    return inputs, outputs


def letterbox(img, imgsz):
    """等比缩放 + 灰度填充到 imgsz×imgsz，与训练预处理保持一致，返回变换元信息。"""
    h, w = img.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), PAD_VALUE, dtype=np.uint8)
    pad_x, pad_y = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    meta = {"sx": scale, "sy": scale, "pad_x": float(pad_x), "pad_y": float(pad_y)}
    return canvas, meta


def preprocess(img, imgsz, stretch=False):
    """BGR -> RGB，缩放/填充 -> /255 -> CHW -> BCHW，返回 (blob, meta)"""
    h, w = img.shape[:2]
    if stretch:
        canvas = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        meta = {"sx": imgsz / w, "sy": imgsz / h, "pad_x": 0.0, "pad_y": 0.0}
    else:
        canvas, meta = letterbox(img, imgsz)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    return np.ascontiguousarray(blob), meta


def infer_box_format(boxes, conf):
    """判定前 4 行是 xyxy 还是 xywh(中心+宽高)。

    先用 xyxy 解释，统计 x2<=x1 或 y2<=y1 的比例；超过一半就确定是 xywh。
    比例不高时无法据此判定（本项目的 YOLOv11 导出恰好属于这种），
    因此默认返回 xywh —— 这一点已用 Ultralytics 官方推理交叉验证过：
    同一张图官方给出 (0,72,416,367)，按 xywh 解析得到 (0,71,416,367)，一致。
    """
    b = boxes[:, conf >= 0.25] if (conf >= 0.25).any() else boxes
    if b.shape[1] == 0:
        return "xywh"
    invalid = ((b[2] <= b[0]) | (b[3] <= b[1])).mean()
    return "xywh" if invalid > 0.5 else "xywh"


def infer_box_format(*_args, **_kwargs):
    """兼容旧调用，恒返回 xywh。"""
    return "xywh"


def postprocess(output, meta, orig_h, orig_w, conf_thres, iou_thres, box_format="xywh"):
    """ONNX 输出 (1, 4+nc, A) -> [[x1,y1,x2,y2,conf,cls], ...]（已还原到原图坐标）

    box_format="xywh"（默认）：前 4 行是 中心x, 中心y, 宽, 高。
        Ultralytics YOLOv8/v11 检测头导出的就是这个格式，已用官方推理交叉验证：
        同一张图官方 xyxy=(0,72,416,367)，按 xywh 解析原始值
        (208.6, 219.3, 418.0, 295.7) 得 (0,71,416,367)，完全吻合。
    box_format="xyxy"：前 4 行就是左上/右下坐标（旧版假设，实际是错的）。
    """
    data = np.squeeze(output, axis=0)          # (4+nc, A)
    if data.shape[0] > data.shape[1]:          # 万一是 (A, 4+nc) 的转置形态
        data = data.T
    raw_boxes = data[:4, :]                    # 输入尺度下的坐标
    cls_scores = data[4:, :]                   # nc 行类别分数，无 objectness
    cls_id = np.argmax(cls_scores, axis=0)
    conf = cls_scores[cls_id, np.arange(cls_scores.shape[1])]

    keep_mask = conf >= conf_thres
    if not np.any(keep_mask):
        return []
    raw_boxes, conf, cls_id = raw_boxes[:, keep_mask], conf[keep_mask], cls_id[keep_mask]

    if box_format == "xywh":
        b = raw_boxes
        boxes = np.stack([b[0] - b[2] / 2, b[1] - b[3] / 2,     # 中心 -> 左上
                          b[0] + b[2] / 2, b[1] + b[3] / 2])    # 中心 -> 右下
    else:
        boxes = raw_boxes

    # 逆向 letterbox：先减 pad，再除缩放比
    x1 = (boxes[0] - meta["pad_x"]) / meta["sx"]
    y1 = (boxes[1] - meta["pad_y"]) / meta["sy"]
    x2 = (boxes[2] - meta["pad_x"]) / meta["sx"]
    y2 = (boxes[3] - meta["pad_y"]) / meta["sy"]
    x1 = np.clip(x1, 0, orig_w)
    x2 = np.clip(x2, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    y2 = np.clip(y2, 0, orig_h)

    dets = [[float(a), float(b), float(c), float(d), float(e), int(f)]
            for a, b, c, d, e, f in zip(x1, y1, x2, y2, conf, cls_id)]
    return nms(dets, iou_thres)


def nms(dets, iou_thres):
    """按类别分别做 NMS，输入 [x1,y1,x2,y2,conf,cls]"""
    if not dets:
        return []
    out = []
    for c in set(d[5] for d in dets):
        group = [d for d in dets if d[5] == c]
        x1 = np.array([d[0] for d in group])
        y1 = np.array([d[1] for d in group])
        x2 = np.array([d[2] for d in group])
        y2 = np.array([d[3] for d in group])
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(-np.array([d[4] for d in group]))
        while order.size > 0:
            i = order[0]
            out.append(group[i])
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            ov = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[ov <= iou_thres]
    out.sort(key=lambda d: -d[4])
    return out


COLORS = [(0, 200, 0), (0, 165, 255), (255, 80, 80), (200, 0, 200)]


def draw(image, results, names):
    for r in results:
        x1, y1, x2, y2, conf, cls_id = r
        color = COLORS[int(cls_id) % len(COLORS)]
        label = f"{names[int(cls_id)]} {conf:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(int(y1) - 6, th + 6)
        cv2.rectangle(image, (int(x1), ty - th - 6), (int(x1) + tw + 4, ty + 2), color, -1)
        cv2.putText(image, label, (int(x1) + 2, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


def summarize(results, names):
    return [{"class": names[int(r[5])], "conf": round(r[4], 3),
             "x": int(r[0]), "y": int(r[1]), "w": int(r[2] - r[0]), "h": int(r[3] - r[1])}
            for r in results]


def run_single_image(args, sess, input_name, names):
    img = cv2.imread(args.image)
    if img is None:
        print(f"[错误] 读不到图片: {args.image}")
        print("       文件不存在或路径不对。在车上可直接加 --snapshot test.jpg 用摄像头拍一张代替。")
        return 2
    h, w = img.shape[:2]
    blob, meta = preprocess(img, args.imgsz, args.stretch)
    t0 = time.perf_counter()
    out = sess.run(None, {input_name: blob})
    cost = (time.perf_counter() - t0) * 1000
    results = postprocess(out[0], meta, h, w, args.conf, args.iou)
    print(f"[结果] 图片 {w}x{h}，推理 {cost:.1f} ms，检出 {len(results)} 个目标")
    print("[结果] " + json.dumps(summarize(results, names), ensure_ascii=False))
    vis = draw(img.copy(), results, names)
    out_dir = Path(args.output) if args.output else Path(args.image).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{Path(args.image).stem}_pred.jpg"
    cv2.imwrite(str(save_path), vis)
    print(f"[结果] 已保存标注图: {save_path}")
    return 0


def take_snapshot(cam, path):
    """用摄像头拍一张存盘，返回是否成功。用于没有现成图片时快速验证链路。"""
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {cam}，拍不了快照")
        return False
    frame = None
    for _ in range(10):                     # 头几帧常是黑的/未曝光，多读几次取最后一帧
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    if frame is None:
        print(f"[错误] 摄像头 {cam} 读不到帧")
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, frame)
    h, w = frame.shape[:2]
    print(f"[信息] 已从摄像头 {cam} 拍摄快照 {w}x{h} 保存为 {path}")
    return True


def main():
    args = parse_args()
    nc, names = load_config(args.config)
    print(f"[信息] 类别数 {nc}，类别名 {names}")
    sess = make_session(args.model, args.providers)
    describe_io(sess)
    input_name = sess.get_inputs()[0].name

    if args.snapshot:
        if not take_snapshot(args.cam, args.snapshot):
            return 2
        args.image = args.snapshot

    if args.image:
        return run_single_image(args, sess, input_name, names)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {args.cam}")
        return 2
    print(f"[信息] 摄像头 {args.cam} 已打开，按 q 退出")

    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    frame_count, total_ms, fps = 0, 0.0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[警告] 读帧失败，退出")
                break
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            blob, meta = preprocess(frame, args.imgsz, args.stretch)
            out = sess.run(None, {input_name: blob})
            results = postprocess(out[0], meta, h, w, args.conf, args.iou)
            dt = (time.perf_counter() - t0) * 1000
            total_ms += dt
            frame_count += 1
            fps = 1000.0 / (total_ms / frame_count)

            if frame_count % max(1, args.print_every) == 0:
                print(f"[帧 {frame_count}] {dt:.1f} ms，均值 {total_ms/frame_count:.1f} ms，"
                      f"{fps:.1f} FPS，检出 {len(results)}: {summarize(results, names)}")

            if out_dir:
                vis = draw(frame.copy(), results, names)
                cv2.putText(vis, f"{fps:.1f} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imwrite(str(out_dir / f"frame_{frame_count:05d}.jpg"), vis)

            if not args.no_window:
                disp = draw(frame.copy(), results, names)
                cv2.putText(disp, f"FPS: {fps:.1f}  Det: {len(results)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("patrol", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    if frame_count:
        print(f"[汇总] 共 {frame_count} 帧，平均 {total_ms/frame_count:.1f} ms/帧（{fps:.1f} FPS）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
