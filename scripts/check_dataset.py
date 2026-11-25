# 数据集检查：并发读取标签，统计类别、空标签和图片/标签对应问题。
# PyCharm 参数示例：--root "Corn Diseases.v2-no-tile.yolov11" --workers 4
import argparse
import json
from time import perf_counter

from common import (image_files, label_files, parallel_map, parse_label,
                    positive_int, project_path, run_main, separate_output)


def parse_args():
    p = argparse.ArgumentParser(description="检查 YOLO 检测数据集")
    p.add_argument("--root", required=True, help="包含 train/valid/test 的数据集根目录")
    p.add_argument("--out", default="outputs/dataset_report.json", help="报告输出路径")
    p.add_argument("--nc", type=positive_int, default=3, help="类别数量")
    p.add_argument("--workers", type=positive_int, default=4, help="标签读取线程数；1 为串行")
    return p.parse_args()


def inspect_label(item):
    """单个标签只读一次；返回结果交给主线程汇总，避免共享计数竞态。"""
    path, nc = item
    counts, bad = [0] * nc, []
    has_content = False
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            has_content = True
            try:
                category, _ = parse_label(line, nc)
            except ValueError as error:
                bad.append(f"{path.name}:{number}: {line.strip()}（{error}）")
            else:
                counts[category] += 1
    return counts, bad, not has_content


def check_split(split_dir, nc, workers=4):
    stat = {"images": 0, "labels": 0, "rows": 0, "empty": 0,
            "bad_rows": [], "class_counts": [0] * nc,
            "missing_label": [], "missing_image": []}
    image_dir, label_dir = split_dir / "images", split_dir / "labels"
    if not image_dir.is_dir():
        return {"error": f"缺少图片目录: {image_dir}"}
    images, labels = image_files(image_dir), label_files(label_dir)
    stat["images"], stat["labels"] = len(images), len(labels)
    image_stems = {image.stem for image in images}
    if len(image_stems) != len(images):
        raise ValueError(f"存在同名不同扩展名图片，无法唯一匹配标签: {image_dir}")
    stat["missing_label"] = [image.name for image in images if image.stem not in labels]
    stat["missing_image"] = [path.name for stem, path in labels.items() if stem not in image_stems]
    jobs = ((labels[image.stem], nc) for image in images if image.stem in labels)
    for counts, bad, empty in parallel_map(inspect_label, jobs, workers):
        stat["empty"] += int(empty)
        stat["rows"] += sum(counts)
        stat["bad_rows"].extend(bad)
        for category, count in enumerate(counts):
            stat["class_counts"][category] += count
    return stat


def main():
    args = parse_args()
    started = perf_counter()
    root, out = project_path(args.root), project_path(args.out)
    if not root.is_dir():
        raise ValueError(f"数据集目录不存在: {root}")
    separate_output(root, out)
    report = {split: check_split(root / split, args.nc, args.workers)
              for split in ("train", "valid", "test")}
    for split, stat in report.items():
        if "error" in stat:
            print(f"[{split}] {stat['error']}")
            continue
        print(f"[{split}] images={stat['images']}, labels={stat['labels']}, "
              f"rows={stat['rows']}, empty={stat['empty']}, class={stat['class_counts']}, "
              f"missing_label={len(stat['missing_label'])}, "
              f"missing_image={len(stat['missing_image'])}, bad={len(stat['bad_rows'])}")
        for bad in stat["bad_rows"][:5]:
            print(f"  异常行: {bad}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已保存: {out}；耗时 {perf_counter() - started:.2f} 秒")


if __name__ == "__main__":
    run_main(main)
