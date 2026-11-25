# 复制非空标签样本到新目录；原始数据不动。
# 空标签是否漏标需人工确认；本脚本仅按“标签非空”筛选，不判断图片是否有病。
# PyCharm 参数示例：--root "Corn Diseases.v2-no-tile.yolov11" --out datasets/corn-clean --workers 4
import argparse
import json
import os
from time import perf_counter

from common import (CORN_NAMES, copy_if_changed, image_files, label_files,
                    label_has_content, parallel_map, positive_int,
                    project_path, run_main, separate_output)


def parse_args():
    p = argparse.ArgumentParser(description="复制非空标签样本，生成玉米训练数据集")
    p.add_argument("--root", required=True, help="原始数据集根目录")
    p.add_argument("--out", default="datasets/corn-clean", help="独立输出目录")
    p.add_argument("--yaml", default="configs/corn-clean.yaml", help="训练配置输出路径")
    p.add_argument("--workers", type=positive_int, default=4, help="I/O 线程数；1 为串行")
    return p.parse_args()


def plan_split(root, out, split, workers):
    """先规划完整输出，发现旧目录残留时停止，避免把已剔除样本继续用于训练。"""
    images = image_files(root / split / "images")
    labels = label_files(root / split / "labels")
    if len({image.stem for image in images}) != len(images):
        raise ValueError(f"{split} 存在同名不同扩展名图片")
    missing = [image.name for image in images if image.stem not in labels]
    if missing:
        raise ValueError(f"{split} 有 {len(missing)} 张图片缺标签，例如 {missing[0]}；请先检查数据集")
    jobs = [(image, labels[image.stem]) for image in images]

    def select(job):
        image, label = job
        return image, label, label_has_content(label)

    kept, removed = [], []
    for image, label, keep in parallel_map(select, jobs, workers):
        if keep:
            kept.append((image, label))
        else:
            removed.append(image.name)
    destinations = (out / split / "images", out / split / "labels")
    expected = ({image.name for image, _ in kept}, {label.name for _, label in kept})
    for directory, names in zip(destinations, expected):
        separate_output(root, directory)
        if directory.exists():
            stale = {path.name for path in directory.iterdir() if path.is_file()
                     and path.suffix.lower() != ".npy"} - names
            if stale:
                raise ValueError(f"输出目录有旧样本 {next(iter(sorted(stale)))}: {directory}；请换一个 --out 目录")
    return kept, removed


def copy_pair(job):
    image, label, image_dir, label_dir = job
    copied_image = copy_if_changed(image, image_dir / image.name)
    copied_label = copy_if_changed(label, label_dir / label.name)
    if copied_image:
        # 仅清除输出副本的旧解码缓存，避免更新图片后仍加载旧 .npy。
        (image_dir / image.name).with_suffix(".npy").unlink(missing_ok=True)
    return int(copied_image) + int(copied_label)


def main():
    args = parse_args()
    started = perf_counter()
    root, out, yaml_path = map(project_path, (args.root, args.out, args.yaml))
    if not root.is_dir():
        raise ValueError(f"原始数据集不存在: {root}")
    separate_output(root, out)
    separate_output(root, yaml_path)
    # 所有划分先通过规划检查，之后才创建目标目录和复制文件。
    plans = {}
    for split in ("train", "valid", "test"):
        if (root / split).is_dir():
            plans[split] = plan_split(root, out, split, args.workers)
    if "train" not in plans or "valid" not in plans:
        raise ValueError("原始数据必须包含 train 和 valid 划分")
    if any(not plans[split][0] for split in ("train", "valid")):
        raise ValueError("train/valid 过滤后没有非空标签样本")
    for split, (kept, removed) in plans.items():
        image_dir, label_dir = out / split / "images", out / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        jobs = ((image, label, image_dir, label_dir) for image, label in kept)
        copied = sum(parallel_map(copy_pair, jobs, args.workers))
        if copied:
            # YOLO 标签缓存不一定感知同尺寸内容变化；更新副本时使其失效。
            (out / split / "labels.cache").unlink(missing_ok=True)
        print(f"[{split}] 保留 {len(kept)}，剔除空标签 {len(removed)}，"
              f"复制/更新文件 {copied}，跳过未变化文件 {len(kept) * 2 - copied}")
    # 配置保留相对路径；训练/验证入口会生成绝对路径运行副本。
    try:
        relative_root = os.path.relpath(out, yaml_path.parent).replace("\\", "/")
    except ValueError:
        # Windows 跨盘符无法表示相对路径，使用运行时计算出的绝对路径。
        relative_root = out.as_posix()
    yaml_text = f"path: {json.dumps(relative_root, ensure_ascii=False)}\n"
    for split in plans:
        yaml_text += f"{'val' if split == 'valid' else split}: {split}/images\n"
    yaml_text += f"nc: {len(CORN_NAMES)}\nnames: {json.dumps(CORN_NAMES)}\n"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_text, encoding="utf-8")
    # 保存真实剔除名单，便于之后复查；不删除原始图片或标签。
    report = {split: {"kept": len(kept), "removed": removed}
              for split, (kept, removed) in plans.items()}
    (out / "clean_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练配置: {yaml_path}；耗时 {perf_counter() - started:.2f} 秒")


if __name__ == "__main__":
    run_main(main)
