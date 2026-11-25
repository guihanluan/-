# 玉米病害验证入口；可调批次、加载进程和 GPU 半精度。
# PyCharm 参数示例：--weights runs/corn-baseline3/weights/best.pt --batch 16 --half
import argparse
import multiprocessing
from time import perf_counter

from common import (load_yolo, nonnegative_int, normalize_device, positive_int,
                    project_path, resolved_dataset_yaml, run_main)


def parse_args():
    p = argparse.ArgumentParser(description="玉米病害模型验证")
    p.add_argument("--weights", required=True, help="本地模型权重路径")
    p.add_argument("--data", default="configs/corn-clean.yaml", help="数据配置")
    p.add_argument("--imgsz", type=positive_int, default=416, help="输入尺寸")
    p.add_argument("--device", default="0", help="0 / cuda:0 / cpu")
    p.add_argument("--batch", type=positive_int, default=16, help="验证批次，显存不足调小")
    p.add_argument("--workers", type=nonnegative_int, default=0, help="数据加载进程数；可尝试 2")
    p.add_argument("--half", action="store_true", help="GPU FP16 验证；可能存在轻微数值差异")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="生成曲线和混淆矩阵")
    p.add_argument("--split", choices=("val", "test"), default="val", help="默认验证集；玉米 test 缺叶枯病")
    p.add_argument("--project", default="runs", help="验证输出目录")
    p.add_argument("--name", default="corn-val", help="验证输出名称")
    return p.parse_args()


def main():
    args = parse_args()
    weights = project_path(args.weights)
    if not weights.is_file():
        raise ValueError(f"找不到模型文件: {weights}")
    data = resolved_dataset_yaml(args.data, required_splits=(args.split,))
    YOLO = load_yolo()
    device = normalize_device(args.device)
    if args.half and device == "cpu":
        raise ValueError("--half 请配合 GPU 使用；CPU 验证去掉 --half")
    if args.split == "test":
        print("注意：当前玉米 test 缺叶枯病，不能完整评估三类病害")
    model = YOLO(str(weights))
    started = perf_counter()
    model.val(
        data=str(data), imgsz=args.imgsz, device=device, batch=args.batch,
        workers=args.workers, half=args.half, plots=args.plots, split=args.split,
        project=str(project_path(args.project)), name=args.name,
    )
    print(f"验证完成，耗时 {perf_counter() - started:.2f} 秒")
    print("请反馈控制台 all / 各类别 P、R、mAP50、mAP50-95")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_main(main)
