# -*- coding: utf-8 -*-
"""实时画面诊断：判断"0 检出"是镜头没内容，还是确实没检测到病斑。

看三组数字就能定位问题：
  亮度均值  —— 太暗(<30)说明镜头被挡/没曝光；太亮(>240)说明过曝对着灯
  标准差    —— 太小(<8)说明对着纯色墙面/白纸，画面没有纹理
  最高分    —— 类别分数最大值。接近 0 说明模型认为"这里没有叶子"；
               有 0.3~0.9 却 0 检出，说明有目标但被阈值卡掉了

用法（车上，与 best.onnx 同目录）：
  python3 live_check.py --model best.onnx --config corn-clean.yaml --cam 0
  python3 live_check.py --model best.onnx --config corn-clean.yaml --cam 0 --frames 30
  python3 live_check.py --model best.onnx --config corn-clean.yaml --image ./imgs/xxx.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patrol_detect as pd


def parse_args():
    p = argparse.ArgumentParser(description="画面诊断：亮度/纹理/最高类别分")
    p.add_argument("--model", default="best.onnx")
    p.add_argument("--config", default="corn-clean.yaml")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--frames", type=int, default=12, help="采样帧数")
    p.add_argument("--image", default="", help="不用摄像头，只诊断一张图")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--providers", default="auto", choices=["auto", "cpu", "cuda", "trt"])
    p.add_argument("--outdir", default="./diag")
    p.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.10, 0.05],
                   help="同时扫几个阈值，看是否有被卡掉的目标")
    return p.parse_args()


def scene_stats(img):
    """画面亮度与纹理强度。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def judge(mean_v, std_v, conf_max, n_det, conf_lo):
    """给出人可读的结论。"""
    tips = []
    if mean_v < 30:
        tips.append("画面过暗：镜头被遮挡或没打开（检查镜头盖/曝光）")
    elif mean_v > 240:
        tips.append("画面过亮：可能正对光源，过曝")
    if std_v < 8:
        tips.append("画面几乎没有纹理：正对纯色墙面/白纸/桌面，模型看不到叶片")
    if conf_max < 0.05:
        tips.append("模型判定此处无叶片：需要把镜头对准玉米叶片")
    elif n_det == 0 and conf_max >= conf_lo:
        tips.append(f"有目标但被阈值卡掉：最高分 {conf_max:.3f}，试 --conf 0.10 或更低")
    if n_det > 0:
        tips.append(f"检出正常：{n_det} 个目标，最高分 {conf_max:.3f}")
    return tips


def main():
    args = parse_args()
    nc, names = pd.load_config(args.config)
    sess = pd.make_session(args.model, args.providers)
    inp = sess.get_inputs()[0].name
    print(f"[信息] 类别 {names}")

    imgs = []
    if args.image:
        im = cv2.imread(args.image)
        if im is None:
            print(f"[错误] 读不到 {args.image}")
            return 2
        imgs = [(Path(args.image).name[:24], im)]
    else:
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            print(f"[错误] 无法打开摄像头 {args.cam}；ls /dev/video* 查设备号")
            return 2
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            cap.read()
        print(f"[信息] 摄像头 {args.cam} "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}，"
              f"采样 {args.frames} 帧")
        for i in range(args.frames):
            ok, f = cap.read()
            if ok and f is not None:
                imgs.append((f"frame{i+1}", f))
        cap.release()

    if not imgs:
        print("[错误] 一帧都没读到")
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conf_lo = min(args.conf)

    print(f"\n{'帧':<9}{'分辨率':>10}{'亮度':>7}{'纹理':>7}{'最高分类分':>11}"
          + "".join(f"{'检出@' + str(c):>11}" for c in args.conf))
    print("-" * (44 + 11 * len(args.conf)))

    stats = []
    for tag, img in imgs:
        h, w = img.shape[:2]
        blob, meta = pd.preprocess(img, args.imgsz)
        out = sess.run(None, {inp: blob})
        arr = np.asarray(out[0])
        conf_max = float(arr[0, 4:, :].max())
        counts = [len(pd.postprocess(out[0], meta, h, w, c, 0.45)) for c in args.conf]
        mean_v, std_v = scene_stats(img)
        stats.append((mean_v, std_v, conf_max, counts[0]))
        line = (f"{tag:<9}{f'{w}x{h}':>10}{mean_v:>7.1f}{std_v:>7.1f}{conf_max:>11.4f}"
                + "".join(f"{c:>11d}" for c in counts))
        print(line)
        cv2.putText(img, f"{tag} bright={mean_v:.0f} tex={std_v:.0f} max={conf_max:.3f}",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(outdir / f"diag_{tag}.jpg"), img)
    means = np.array([s[0] for s in stats])
    stds = np.array([s[1] for s in stats])
    confs = np.array([s[2] for s in stats])
    dets = np.array([s[3] for s in stats])

    print("-" * (44 + 11 * len(args.conf)))
    print(f"{'均值':<9}{'':>10}{means.mean():>7.1f}{stds.mean():>7.1f}{confs.mean():>11.4f}"
          f"{dets.mean():>11.2f}")

    print("\n诊断结论：")
    tips = judge(float(means.mean()), float(stds.mean()), float(confs.max()),
                 int(dets.sum()), conf_lo)
    if not tips:
        print("  未发现明显问题")
    for t in tips:
        print(f"  - {t}")

    print(f"\n采样图已存到 {outdir.resolve()}/diag_*.jpg")
    if float(stds.mean()) < 8:
        print("  这些图应该是纯色画面，打开看一眼确认镜头对着什么")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
