# -*- coding: utf-8 -*-
"""小车实时画面识别（Jetson / 树莓派 / PC 通用）

设计要点（都是为了实车能用）：
  1) 采集线程与推理线程分离，摄像头不会因为推理慢而积压旧帧 —— 永远识别最新画面；
  2) 显示与识别分离，跑不动时自动丢帧保低延迟，而不是越跑越卡；
  3) 没有显示器（SSH）时自动降级为无窗口模式，按间隔存快照；
  4) 提供 --resolution 降低采集分辨率，这是 Jetson 上提速最有效的一招；
  5) 热键：q 退出、s 存图、+/- 调置信度、d 切换显示开关。

常用命令：
  # 有显示器（noMachine / 本地桌面）
  python3 realtime.py --model best.onnx --config corn-clean.yaml --cam 0

  # SSH 无显示器，每 20 帧存一张图
  python3 realtime.py --model best.onnx --config corn-clean.yaml --cam 0 --no-window --save-every 20

  # Jetson 想更流畅：降分辨率 + 放宽 NMS
  python3 realtime.py --model best.onnx --config corn-clean.yaml --cam 0 --resolution 416x416

  # 直接录成视频便于回看
  python3 realtime.py --model best.onnx --config corn-clean.yaml --cam 0 --record demo.mp4
"""
import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patrol_detect as pd

CN_NAMES = {
    "Corn Rust": "玉米锈病",
    "Grey Leaf Spot": "灰斑病",
    "Leaf Blight": "叶枯病",
}

try:                                    # 中文标签需要 PIL，没有就退回英文
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def parse_args():
    p = argparse.ArgumentParser(description="小车实时画面识别")
    p.add_argument("--model", default="best.onnx")
    p.add_argument("--config", default="corn-clean.yaml")
    p.add_argument("--cam", type=int, default=0, help="摄像头编号，USB 一般 0")
    p.add_argument("--resolution", default="", help="采集分辨率，如 640x480 或 416x416")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--providers", default="auto", choices=["auto", "cpu", "cuda", "trt"])
    p.add_argument("--no-window", action="store_true", help="强制无窗口模式")
    p.add_argument("--cn", action="store_true", help="用中文类别名显示")
    p.add_argument("--font", default="", help="中文字体文件路径，留空自动查找")
    p.add_argument("--save-every", type=int, default=0, help="每 N 帧存一张标注图，0=不存")
    p.add_argument("--outdir", default="./live", help="存图目录")
    p.add_argument("--record", default="", help="录制为 mp4 文件")
    p.add_argument("--max-fps", type=float, default=0, help="限制处理帧率，0=不限")
    p.add_argument("--duration", type=float, default=0, help="运行多少秒后自动退出，0=不限")
    return p.parse_args()


class CameraStream:
    """后台线程持续读帧，只保留最新一帧，避免推理慢时画面滞后。"""

    def __init__(self, cam, resolution=""):
        self.cap = cv2.VideoCapture(cam)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {cam}")
        if resolution:
            w, h = resolution.lower().split("x")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # 关键：内核只缓存最新帧
        # 部分摄像头/驱动在切换模式后会吐几帧旧的或很慢的帧，先丢一批
        for _ in range(5):
            self.cap.read()
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = False
        self._ok_reads = 0
        self._fail_reads = 0
        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._fail_reads += 1
                if self._fail_reads > 50:
                    break
                time.sleep(0.01)
                continue
            self._ok_reads += 1
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self):
        """取最新帧及其序号；没新帧时返回 None，让调用方跳过而不是重复处理。"""
        with self._lock:
            if self._frame is None:
                return None, self._seq
            return self._frame.copy(), self._seq

    def release(self):
        self._stop = True
        if self._t.is_alive():
            self._t.join(timeout=1.0)
        self.cap.release()


CJK_FONT_CANDIDATES = [
    # Linux（Jetson / Ubuntu 常见路径）
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    # Windows（本机调试用）
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]
_FONT_PATH = None


def find_cjk_font(explicit=""):
    """找一个能渲染中文的字体文件；找不到返回 None（调用方应退回英文）。"""
    global _FONT_PATH
    if explicit and Path(explicit).exists():
        _FONT_PATH = explicit
        return _FONT_PATH
    if _FONT_PATH and Path(_FONT_PATH).exists():
        return _FONT_PATH
    for p in CJK_FONT_CANDIDATES:
        if Path(p).exists():
            _FONT_PATH = p
            return p
    return None


def put_text_cn(img, text, org, color=(255, 255, 255), size=18):
    """用 PIL 画中文；无可用中文字体时返回 None，由调用方退回英文标签。"""
    fp = find_cjk_font()
    if not _HAS_PIL or fp is None:
        return None
    try:
        font = ImageFont.truetype(fp, size)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text(org, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def draw_result(frame, results, names, use_cn, fps, infer_ms, show, conf_thr):
    """画框 + 左上角状态栏。results 为 postprocess 输出。"""
    vis = frame.copy()
    for r in results:
        x1, y1, x2, y2, conf, cls_id = r
        name = names[int(cls_id)]
        label = CN_NAMES.get(name, name) if use_cn else name
        label = f"{label} {conf:.2f}"
        color = (0, 200, 0) if conf >= 0.5 else (0, 165, 255)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(int(y1) - 6, th + 6)
        cv2.rectangle(vis, (int(x1), ty - th - 6), (int(x1) + tw + 6, ty + 2), color, -1)
        drawn = put_text_cn(vis, label, (int(x1) + 3, ty - th - 4), (255, 255, 255), 18) if use_cn else None
        if drawn is not None:
            vis = drawn
        else:
            cv2.putText(vis, f"{name} {conf:.2f}", (int(x1) + 3, ty - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    bar = f"FPS {fps:4.1f} | {infer_ms:5.1f}ms | Det {len(results)} | conf {conf_thr:.2f}"
    if show:
        bar += " | q:quit s:save +/-:conf d:hide"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, bar, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return vis


def has_display():
    """判断是否有可用图形显示环境。"""
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main():
    args = parse_args()
    nc, names = pd.load_config(args.config)
    use_cn = args.cn
    if use_cn:
        fp = find_cjk_font(args.font)
        if not _HAS_PIL:
            print("[警告] 未安装 PIL，中文标签不可用，改用英文（pip3 install pillow 可启用）")
            use_cn = False
        elif fp is None:
            print("[警告] 没找到中文字体，改用英文标签")
            print("       Jetson 上可装：sudo apt install fonts-wqy-zenhei")
            print("       或指定字体：--font /usr/share/fonts/truetype/xxx.ttf")
            use_cn = False
        else:
            print(f"[信息] 中文字体 {fp}")
    print(f"[信息] 类别 {names}")

    sess = pd.make_session(args.model, args.providers)
    inp = sess.get_inputs()[0].name
    print(f"[信息] 输入 {[(i.name, i.shape) for i in sess.get_inputs()]}")

    try:
        cam = CameraStream(args.cam, args.resolution)
    except RuntimeError as e:
        print(f"[错误] {e}")
        print("       ls /dev/video* 查看设备号，试 --cam 1 或 chmod 666 /dev/video0")
        return 2
    print(f"[信息] 摄像头 {args.cam} 打开，采集分辨率 {cam.actual_w}x{cam.actual_h}")

    show = not args.no_window
    if show and not has_display():
        print("[信息] 未检测到图形界面（无 DISPLAY），自动切换无窗口模式")
        print("       如需窗口：用 noMachine/本地桌面登录，或加 --save-every 20 存图查看")
        show = False

    outdir = Path(args.outdir)
    if args.save_every or (args.record and not show):
        outdir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 15.0,
                                 (cam.actual_w, cam.actual_h))

    # 预热，避免首帧耗时污染统计
    dummy = np.zeros((1, 3, args.imgsz, args.imgsz), np.float32)
    for _ in range(3):
        sess.run(None, {inp: dummy})

    conf_thr = args.conf
    processed, saved, last_seq = 0, 0, -1
    t_start = time.time()
    times = deque(maxlen=30)          # 用滑动窗口算 FPS，比总均值更能反映当前状态
    infer_times = deque(maxlen=30)
    fps, infer_ms = 0.0, 0.0
    last_print = time.time()

    print("[信息] 开始识别，Ctrl+C 或 q 退出" if show else "[信息] 开始识别（无窗口），Ctrl+C 退出")
    try:
        while True:
            frame, seq = cam.read()
            if frame is None or seq == last_seq:
                time.sleep(0.003)
                continue
            last_seq = seq

            t0 = time.perf_counter()
            blob, meta = pd.preprocess(frame, args.imgsz)
            out = sess.run(None, {inp: blob})
            h, w = frame.shape[:2]
            results = pd.postprocess(out[0], meta, h, w, conf_thr, args.iou)
            dt = (time.perf_counter() - t0) * 1000
            processed += 1
            infer_times.append(dt)
            now = time.perf_counter()
            times.append(now)
            if len(times) > 1:
                fps = (len(times) - 1) / (times[-1] - times[0])
            infer_ms = float(np.mean(infer_times))

            vis = draw_result(frame, results, names, use_cn, fps, infer_ms, show, conf_thr)

            if args.save_every and processed % args.save_every == 0:
                saved += 1
                cv2.imwrite(str(outdir / f"live_{saved:05d}.jpg"), vis)
            if writer is not None:
                writer.write(vis)

            if show:
                cv2.imshow("Corn Patrol - Realtime", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    saved += 1
                    path = outdir / f"shot_{saved:05d}.jpg"
                    outdir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), vis)
                    print(f"[存图] {path}")
                if key in (ord("+"), ord("=")):
                    conf_thr = min(0.95, conf_thr + 0.05)
                    print(f"[参数] conf -> {conf_thr:.2f}")
                if key == ord("-"):
                    conf_thr = max(0.05, conf_thr - 0.05)
                    print(f"[参数] conf -> {conf_thr:.2f}")

            if time.time() - last_print >= 5:
                last_print = time.time()
                cls_cnt = {}
                for r in results:
                    cls_cnt[names[int(r[5])]] = cls_cnt.get(names[int(r[5])], 0) + 1
                print(f"[{processed:5d} 帧] {fps:4.1f} FPS  {infer_ms:5.1f} ms  "
                      f"本帧检出 {len(results)} {cls_cnt if cls_cnt else ''}")

            if args.max_fps > 0:
                time.sleep(max(0.0, 1.0 / args.max_fps - (time.perf_counter() - t0)))
            if args.duration > 0 and time.time() - t_start >= args.duration:
                print(f"[信息] 已达 {args.duration}s，自动退出")
                break
    except KeyboardInterrupt:
        print("\n[信息] 收到 Ctrl+C，退出")
    finally:
        cam.release()
        if writer is not None:
            writer.release()
            print(f"[信息] 视频已保存 {args.record}")
        if show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print("\n" + "=" * 52)
    print(f"共处理 {processed} 帧，用时 {elapsed:.1f}s，平均 {processed/max(elapsed,1e-6):.1f} FPS")
    if infer_times:
        print(f"单帧耗时 均值 {np.mean(infer_times):.1f} ms  "
              f"P95 {np.percentile(infer_times, 95):.1f} ms")
    if saved:
        print(f"已保存 {saved} 张图到 {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
