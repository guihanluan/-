# best.pt -> ONNX：固定输入、batch=1；TensorRT engine 仍需在 Jetson 本机生成。
# PyCharm 参数示例：--weights runs/corn-baseline3/weights/best.pt --imgsz 416
import argparse
import importlib.util
from time import perf_counter

from common import load_yolo, normalize_device, positive_int, project_path, run_main


def parse_args():
    p = argparse.ArgumentParser(description="导出 ONNX 供 Jetson 转换 TensorRT")
    p.add_argument("--weights", required=True, help="本地 best.pt 路径")
    p.add_argument("--imgsz", type=positive_int, default=416, help="输入尺寸")
    p.add_argument("--device", default="cpu", help="默认 cpu；可用 0 / cuda:0")
    p.add_argument("--opset", type=positive_int, default=None, help="按 Jetson TensorRT 兼容性指定；默认由库选择")
    p.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True, help="简化 ONNX 图，需要 onnxslim")
    return p.parse_args()


def main():
    args = parse_args()
    weights = project_path(args.weights)
    if not weights.is_file():
        raise ValueError(f"找不到模型权重: {weights}")
    if weights.suffix.lower() != ".pt":
        raise ValueError("--weights 应指向训练生成的 .pt 文件")
    dependencies = ("onnx", "onnxslim") if args.simplify else ("onnx",)
    for module in dependencies:
        if importlib.util.find_spec(module) is None:
            hint = "；若不需要图简化，可加 --no-simplify" if module == "onnxslim" else ""
            raise ImportError(f"当前解释器缺少 {module}，请在 PyCharm 所选环境手动补充{hint}")
    YOLO = load_yolo()
    device = normalize_device(args.device)
    if "," in device:
        raise ValueError("导出请指定一块 GPU 或 cpu")
    model = YOLO(str(weights))
    options = {"format": "onnx", "imgsz": args.imgsz, "device": device,
               "batch": 1, "dynamic": False, "half": False,
               "simplify": args.simplify, "nms": False}
    if args.opset is not None:
        options["opset"] = args.opset
    started = perf_counter()
    output = model.export(**options)
    print(f"ONNX 已导出: {output}；耗时 {perf_counter() - started:.2f} 秒")
    print("若产生同名 .onnx.data 文件，也需要一起复制到 Jetson")
    print("在 Jetson 本机生成 FP16 TensorRT engine；这里导出的 ONNX 为 FP32")


if __name__ == "__main__":
    run_main(main)
