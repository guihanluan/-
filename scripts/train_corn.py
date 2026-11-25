# 玉米病害训练入口：safe 保留原基线，fast 提供可选提速配置。
# PyCharm 加速参数示例：--profile fast --name corn-fast
# 保留原训练设置的可选提速：--profile safe --decode-cache --name corn-safe-cache
# 发生 Windows 1455/内存不足时：--profile fast --workers 0 --batch 8
import argparse
import multiprocessing
from time import perf_counter

from common import (batch_size, load_yolo, nonnegative_int, normalize_device,
                    positive_int, project_path, resolved_dataset_yaml, run_main)


PROFILES = {
    "safe": {"batch": 8, "workers": 0, "deterministic": True, "plots": True},
    "fast": {"batch": 16, "workers": 2, "deterministic": False, "plots": False},
}


def parse_args():
    p = argparse.ArgumentParser(description="玉米病害 YOLO11n 训练")
    p.add_argument("--profile", choices=tuple(PROFILES), default="safe", help="safe 稳定；fast 可选提速")
    p.add_argument("--data", default="configs/corn-clean.yaml", help="训练数据配置")
    p.add_argument("--model", default="yolo11n.pt", help="本地预训练权重或模型 YAML")
    p.add_argument("--imgsz", type=positive_int, default=416, help="输入尺寸")
    p.add_argument("--epochs", type=positive_int, default=100, help="训练轮数")
    p.add_argument("--batch", type=batch_size, default=None, help="覆盖模式批次；-1 为 GPU 自动选择")
    p.add_argument("--device", default="0", help="0 / cuda:0 / cpu")
    p.add_argument("--project", default="runs", help="训练输出目录")
    p.add_argument("--name", default=None, help="训练名称；默认 corn-baseline 或 corn-fast")
    p.add_argument("--patience", type=nonnegative_int, default=25, help="早停轮数；0 关闭")
    p.add_argument("--workers", type=nonnegative_int, default=None, help="覆盖加载进程数；Windows 内存不足用 0")
    p.add_argument("--cache", choices=("none", "ram"), default="none", help="默认不缓存；ram 会占用更多内存")
    p.add_argument("--decode-cache", action="store_true", help="独立磁盘无损解码缓存；保留 batch/增强/数值设置")
    p.add_argument("--decode-cache-dir", default="outputs/image_cache", help="独立缓存目录，不写入数据集")
    p.add_argument("--decode-cache-workers", type=positive_int, default=4, help="仅缓存准备阶段的 I/O 线程数")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="混合精度，原基线已开启")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None, help="覆盖可复现性配置")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None, help="覆盖训练绘图配置")
    p.add_argument("--seed", type=nonnegative_int, default=0, help="随机种子")
    return p.parse_args()


def main():
    args = parse_args()
    settings = dict(PROFILES[args.profile])
    for key in settings:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    name = args.name or ("corn-fast" if args.profile == "fast" else "corn-baseline")
    weights = project_path(args.model)
    if not weights.is_file():
        raise ValueError(f"找不到本地模型文件: {weights}；请设置 --model")
    # 先检查数据，再加载大体积依赖；不依赖 Ultralytics 全局 datasets_dir。
    data = resolved_dataset_yaml(args.data)
    YOLO = load_yolo()
    device = normalize_device(args.device)
    if device == "cpu" and settings["batch"] == -1:
        raise ValueError("CPU 不支持自动 batch，请设置 --batch 8 等正整数")
    if "," in device and settings["batch"] == -1:
        raise ValueError("多 GPU 训练请指定正整数 batch，不使用 -1")
    if args.decode_cache and (args.cache != "none" or "," in device):
        raise ValueError("独立解码缓存目前支持单设备及 --cache none")
    project = project_path(args.project)
    print(f"模式: {args.profile}；数据配置: {data}")
    print(f"batch={settings['batch']}，workers={settings['workers']}，"
          f"cache={args.cache}，amp={args.amp}，"
          f"deterministic={settings['deterministic']}，plots={settings['plots']}")
    if args.profile == "fast":
        print("fast 模式允许非确定性算法，训练数值可能与原基线不同；显存/内存不足可降到 batch=8、workers=0")
    if args.cache == "ram":
        print("RAM 缓存会增加内存占用；内存不足时使用 --cache none")
    model = YOLO(str(weights))
    train_options = {}
    if args.decode_cache:
        from image_cache import decoded_cache_trainer
        train_options["trainer"] = decoded_cache_trainer(args.decode_cache_dir, args.decode_cache_workers)
    started = perf_counter()
    try:
        model.train(
            data=str(data), imgsz=args.imgsz, epochs=args.epochs, device=device,
            project=str(project), name=name, patience=args.patience,
            cache=False if args.cache == "none" else "ram",
            amp=args.amp, seed=args.seed, **settings, **train_options,
        )
    except RuntimeError as error:
        if any(word in str(error).lower() for word in ("out of memory", "1455", "paging file")):
            raise SystemExit("[错误] 显存或内存不足，请使用 --workers 0 --batch 8 --cache none；"
                             "仍不足时再调小 batch") from error
        raise
    # Ultralytics 自动追加目录编号；以 trainer 的实际目录为准。
    print(f"训练结果: {model.trainer.save_dir}；用时 {(perf_counter() - started) / 60:.1f} 分钟")
    print("请反馈每轮耗时、GPU 显存占用，以及 Precision / Recall / mAP50 / mAP50-95")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_main(main)
