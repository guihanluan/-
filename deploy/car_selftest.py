# -*- coding: utf-8 -*-
"""车上自检脚本：一次性检查文件完整性、依赖、模型 IO、推理链路。

用法（本文件与模型 / corn-clean.yaml 同目录）：
  python3 car_selftest.py                                  # 自动找可用模型
  python3 car_selftest.py --model best_ir9.onnx            # 指定模型
  python3 car_selftest.py --fix-ir                         # 就地把 IR10 模型降为 IR9
  python3 car_selftest.py --image /path/to/field.jpg       # 追加一张真实图测试
  python3 car_selftest.py --skip-camera                    # 不测摄像头

退出码：0 全部通过，1 有失败项。

关于 IR 版本：
  新版 PyTorch 导出的 ONNX 会把 ir_version 声明成 10，而 Python 3.8 只能装
  onnxruntime 1.16.x，其上限是 IR 9，于是报
  "Unsupported model IR version: 10, max supported IR version: 9"。
  本模型的 opset 是 18（ORT 1.16 支持到 19），且不含 FP8/INT4 等 IR10 专属类型，
  所以把 ir_version 这一个字节从 10 改成 9 是安全的，计算结果逐位不变。
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

YAML_FILE = ("corn-clean.yaml", "999350433E4580C71BF8B5178E59FD61D2EB3F9B9E3E619344CB94D45229DBE9", 145)
MODEL_FILE = ("best_ir9.onnx", "A55F0D4809D206BEE00D9FBDE07ECCD96B04102F075683088805CB243EF47196", 10439951)
LEGACY_MODEL = ("best.onnx", "5A91A13E23DFDA29E81E6527F0426386EC9FA2BAF2169B0B31406E6D2CB05A33", 10439951)

# 同一个网络结构有 IR9 / IR10 两种编码，内容等价，哈希不同，两者都算合法
KNOWN_MODEL_HASHES = {
    MODEL_FILE[1].upper(): "IR9（ORT 1.16 可直接加载）",
    LEGACY_MODEL[1].upper(): "IR10（需 --fix-ir 或降版）",
}

PY_DEPS = [("numpy", "numpy"), ("cv2", "opencv-python"), ("onnxruntime", "onnxruntime"), ("yaml", "pyyaml")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_ir_version(path):
    """直接读 protobuf 头部第 1 个字段（field 1 = ir_version，varint）。
    protobuf 编码：0x08 <varint>，小于 128 时 varint 就是单字节。"""
    with open(path, "rb") as f:
        head = f.read(4)
    if len(head) < 2 or head[0] != 0x08:
        return None
    return head[1]


class Checker:
    def __init__(self):
        self.failed = 0

    def ok(self, msg):
        print(f"  [OK]   {msg}")

    def bad(self, msg):
        self.failed += 1
        print(f"  [FAIL] {msg}")

    def info(self, msg):
        print(f"  [--]   {msg}")

    def head(self, title):
        print(f"\n=== {title} ===")


def fix_ir(root: Path, src_name: str, dst_name: str = "best_ir9.onnx"):
    """把 ir_version 从 10 改成 9：只动 protobuf 头部那一个字节，其余完全不变。"""
    src = root / src_name
    dst = root / dst_name
    data = bytearray(src.read_bytes())
    if data[0] != 0x08:
        print(f"[错误] {src_name} 头部不是预期格式，放弃修补")
        return False
    old = data[1]
    if old == 9:
        print(f"[信息] {src_name} 已经是 IR 9，无需修补")
        return True
    if old != 10:
        print(f"[错误] ir_version = {old}，本脚本只处理 10 -> 9")
        return False
    data[1] = 9
    dst.write_bytes(bytes(data))
    after = bytearray(src.read_bytes())
    if sum(1 for i in range(len(after)) if after[i] != data[i]) != 1:
        print("[错误] 检查发现字节变动不止 1 处，放弃")
        return False
    print(f"[完成] 已生成 {dst_name}（{len(data)} 字节，ir_version 10 -> 9，仅改动 1 字节）")
    print(f"       SHA256 {sha256(dst)}")
    return True


def check_files(c: Checker, root: Path, model_name: str, strict_hash: bool):
    c.head("1. 文件存在性与完整性")
    entries = [YAML_FILE]
    expected = MODEL_FILE if model_name == MODEL_FILE[0] else LEGACY_MODEL
    entries.append(expected)

    for name, want_hash, want_size in entries:
        p = root / name
        if not p.exists():
            # best_ir9.onnx 不存在时，退一步看 best.onnx 在不在，给出明确指引
            alt = LEGACY_MODEL[0] if name == MODEL_FILE[0] else None
            hint = f"；{alt} 存在则改用 --model {alt} 或加 --fix-ir 自动生成" if alt and (root / alt).exists() else ""
            c.bad(f"缺少文件 {name}（路径 {p}）{hint}")
            continue
        size = p.stat().st_size
        if size != want_size:
            c.bad(f"{name} 大小 {size} != 期望 {want_size}，拷贝被截断或不是同一版本")
            continue
        c.ok(f"{name} 存在，大小 {size} 字节")
        if strict_hash:
            got = sha256(p)
            if got == want_hash:
                c.ok(f"{name} SHA256 匹配")
            elif name.endswith(".onnx") and got in KNOWN_MODEL_HASHES:
                c.ok(f"{name} SHA256 匹配（{KNOWN_MODEL_HASHES[got]}）")
            else:
                c.bad(f"{name} SHA256 不匹配\n         期望 {want_hash}\n         实际 {got}")

    ir = read_ir_version(root / model_name) if (root / model_name).exists() else None
    if ir is not None:
        if ir <= 9:
            c.ok(f"{model_name} ir_version = {ir}（onnxruntime 1.16 上限 9）")
        else:
            c.bad(f"{model_name} ir_version = {ir} > 9，ORT 1.16 无法加载；"
                  f"执行 python3 car_selftest.py --fix-ir 生成 best_ir9.onnx")


def check_deps(c: Checker):
    c.head("2. Python 依赖")
    for mod, pkg in PY_DEPS:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            c.ok(f"{mod} {ver}")
        except Exception as e:
            c.bad(f"import {mod} 失败：{e}（pip3 install {pkg}）")
    print(f"  [--]   Python {sys.version.split()[0]} / {sys.platform}")


def check_model(c: Checker, root: Path, model_name: str, imgsz: int):
    c.head("3. 模型加载与输入输出")
    import onnxruntime as rt

    avail = rt.get_available_providers()
    c.info(f"onnxruntime {rt.__version__}，可用 provider: {avail}")
    want = [p for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            if p in avail] or ["CPUExecutionProvider"]
    t0 = time.perf_counter()
    try:
        sess = rt.InferenceSession(str(root / model_name), providers=want)
    except Exception as e:
        last = str(e).splitlines()[-1] if str(e).splitlines() else str(e)
        c.bad(f"加载 {model_name} 失败：{last}")
        if "IR version" in str(e):
            c.info("这是版本不匹配，不是文件损坏；执行 --fix-ir 即可解决")
        return None
    c.ok(f"加载耗时 {(time.perf_counter()-t0)*1000:.0f} ms，使用 {sess.get_providers()}")

    ins, outs = sess.get_inputs(), sess.get_outputs()
    c.info(f"输入 {[(i.name, i.shape, i.type) for i in ins]}")
    c.info(f"输出 {[(o.name, o.shape, o.type) for o in outs]}")

    shape = ins[0].shape
    if len(shape) != 4 or shape[1] != 3:
        c.bad(f"输入应为 (N,3,H,W)，实际 {shape}")
    elif isinstance(shape[2], int) and shape[2] != imgsz:
        c.bad(f"输入高 {shape[2]} != --imgsz {imgsz}，导出尺寸与运行尺寸不一致")
    else:
        c.ok(f"输入尺寸与 --imgsz {imgsz} 一致")

    oshape = outs[0].shape
    if len(oshape) == 3 and isinstance(oshape[1], int):
        nc = oshape[1] - 4
        c.info(f"输出第二维 = 4 + {nc}（应为 4 + 3 = 7，没有 objectness 通道）")
        if nc != 3:
            c.bad(f"类别数 {nc} != 3，模型或 yaml 不匹配")
        if isinstance(oshape[2], int):
            c.ok(f"锚点数 {oshape[2]}（416 输入预期 3549）")
    return sess


def check_config(c: Checker, root: Path):
    c.head("4. corn-clean.yaml 与模型一致性")
    import yaml
    p = root / YAML_FILE[0]
    if not p.exists():
        c.bad(f"缺少 {YAML_FILE[0]}")
        return None
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    nc, names = cfg.get("nc"), cfg.get("names")
    c.info(f"nc={nc}, names={names}")
    if nc != 3 or not isinstance(names, list) or len(names) != 3:
        c.bad("nc 或 names 不是 3 类，与模型输出不匹配")
        return None
    c.ok("3 类类别名读取正常")
    return names


def check_inference(c: Checker, sess, names, imgsz: int, extra_image: str):
    c.head("5. 推理链路（onnxruntime 原生，不依赖 ultralytics）")
    import numpy as np

    inp = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    sess.run(None, {inp: dummy})          # 首次含图优化/显存分配，不计入耗时
    t0 = time.perf_counter()
    out = sess.run(None, {inp: dummy})
    c.ok(f"空图推理成功，{(time.perf_counter()-t0)*1000:.1f} ms（预热后）")

    arr = np.asarray(out[0])
    if arr.shape[1] != 7:
        c.bad(f"输出第二维 {arr.shape[1]} != 7")
    else:
        c.ok(f"输出形状 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        c.bad("输出含 NaN/Inf")
        return
    cls = arr[0, 4:, :]
    c.info(f"类别分数范围 [{cls.min():.4f}, {cls.max():.4f}]（空图本应接近 0，>0 说明权重真实加载）")
    if cls.max() == 0:
        c.bad("所有类别分数为 0，权重可能没加载")

    if extra_image:
        try:
            import cv2
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import patrol_detect as pd
            img = cv2.imread(extra_image)
            if img is None:
                c.bad(f"读不到 --image {extra_image}")
            else:
                h, w = img.shape[:2]
                blob, meta = pd.preprocess(img, imgsz)
                out = sess.run(None, {inp: blob})
                res = pd.postprocess(out[0], meta, h, w, 0.25, 0.45)
                c.ok(f"真实图 {Path(extra_image).name} {w}x{h}，检出 {len(res)} 个目标")
                print("         " + json.dumps(pd.summarize(res, names), ensure_ascii=False))
        except Exception as e:
            c.bad(f"真实图推理失败：{e}")


def check_camera(c: Checker, cam: int):
    c.head("6. 摄像头")
    try:
        import cv2
    except Exception as e:
        c.bad(f"cv2 不可用，跳过摄像头检查：{e}")
        return
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        c.bad(f"无法打开 /dev/video{cam}（ls /dev/video* 看设备号；USB 摄像头常需 --cam 1）")
        return
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        c.bad(f"摄像头 {cam} 打开但读不到帧")
    else:
        h, w = frame.shape[:2]
        c.ok(f"摄像头 {cam} 可用，分辨率 {w}x{h}，通道 {frame.shape[2]}")


def pick_model(root: Path, given: str):
    if given:
        return given
    for cand in (MODEL_FILE[0], LEGACY_MODEL[0]):
        if (root / cand).exists():
            return cand
    return MODEL_FILE[0]


def main():
    p = argparse.ArgumentParser(description="车上部署自检")
    p.add_argument("--dir", default=".", help="部署目录，默认当前目录")
    p.add_argument("--model", default="", help="模型文件名，默认自动找 best_ir9.onnx 或 best.onnx")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--image", default="", help="可选：额外跑一张真实拍摄图片")
    p.add_argument("--cam", type=int, default=0, help="摄像头编号；给 -1 跳过")
    p.add_argument("--skip-hash", action="store_true", help="跳过 SHA256（Jetson 上较慢）")
    p.add_argument("--skip-camera", action="store_true")
    p.add_argument("--fix-ir", action="store_true",
                   help="把 best.onnx 的 ir_version 从 10 降到 9，生成本地 best_ir9.onnx")
    a = p.parse_args()

    root = Path(a.dir).resolve()
    print(f"自检目录: {root}")

    if a.fix_ir:
        print("\n=== 0. 修正 IR 版本 ===")
        src = LEGACY_MODEL[0] if (root / LEGACY_MODEL[0]).exists() else MODEL_FILE[0]
        if not fix_ir(root, src):
            return 1
        a.model = MODEL_FILE[0]

    model_name = pick_model(root, a.model)
    if model_name == LEGACY_MODEL[0] and (root / MODEL_FILE[0]).exists():
        print(f"[提示] 同时存在 {MODEL_FILE[0]}，建议用 --model {MODEL_FILE[0]}")
    print(f"使用模型: {model_name}")

    c = Checker()
    check_files(c, root, model_name, not a.skip_hash)
    check_deps(c)
    sess = check_model(c, root, model_name, a.imgsz)
    names = check_config(c, root)
    if sess is not None and names:
        check_inference(c, sess, names, a.imgsz, a.image)
    if not a.skip_camera and a.cam >= 0:
        check_camera(c, a.cam)

    print("\n" + "=" * 46)
    if c.failed == 0:
        print(f"全部检查通过，可以开始跑 patrol_detect.py --model {model_name}")
        return 0
    print(f"有 {c.failed} 项失败，请按上面 [FAIL] 逐项处理")
    return 1


if __name__ == "__main__":
    sys.exit(main())
