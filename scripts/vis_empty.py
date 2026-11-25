# 集中复制空标签图片；区分缺失标签，避免把两种问题混在一起。
# PyCharm 参数示例：--root "Corn Diseases.v2-no-tile.yolov11" --split train --workers 4
import argparse
from time import perf_counter

from common import (copy_if_changed, image_files, label_files, label_has_content,
                    parallel_map, positive_int, project_path, run_main, separate_output)


def parse_args():
    p = argparse.ArgumentParser(description="集中检查空标签图片")
    p.add_argument("--root", required=True, help="数据集根目录")
    p.add_argument("--split", choices=("train", "valid", "test"), default="train")
    p.add_argument("--out", default="outputs/empty_check", help="集中图片输出目录")
    p.add_argument("--workers", type=positive_int, default=4, help="I/O 线程数；1 为串行")
    return p.parse_args()


def main():
    args = parse_args()
    started = perf_counter()
    root = project_path(args.root)
    out_dir = project_path(args.out) / args.split
    separate_output(root, out_dir)
    images = image_files(root / args.split / "images")
    labels = label_files(root / args.split / "labels")

    def classify(image):
        label = labels.get(image.stem)
        return image, label is None, label is not None and not label_has_content(label)

    empty, missing = [], []
    for image, is_missing, is_empty in parallel_map(classify, images, args.workers):
        if is_missing:
            missing.append(image.name)
        elif is_empty:
            empty.append(image)
    # 旧复查目录不能混入已经不再为空的图片。
    if out_dir.exists():
        stale = {path.name for path in image_files(out_dir)} - {image.name for image in empty}
        if stale:
            raise ValueError(f"复查目录有 {len(stale)} 张旧图片，请指定新的 --out")
    out_dir.mkdir(parents=True, exist_ok=True)

    def copy_one(image):
        return copy_if_changed(image, out_dir / image.name)

    copied = sum(parallel_map(copy_one, empty, args.workers))
    print(f"[{args.split}] 空标签 {len(empty)} 张，缺失标签 {len(missing)} 张")
    for name in missing[:5]:
        print(f"  缺失标签: {name}")
    print(f"复制/更新 {copied} 张，跳过未变化 {len(empty) - copied} 张 -> {out_dir}")
    print(f"耗时 {perf_counter() - started:.2f} 秒；请人工确认空标签是否漏标")


if __name__ == "__main__":
    run_main(main)
