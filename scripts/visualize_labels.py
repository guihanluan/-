# 使用 Pillow 画框；支持并发批量导出，不依赖 cv2。
# PyCharm 参数示例：--root "Corn Diseases.v2-no-tile.yolov11" --split train --num 30 --workers 4
import argparse
import random
from time import perf_counter

from common import (CORN_NAMES, image_files, parallel_map, parse_label,
                    positive_int, project_path, run_main, separate_output)


COLORS = [(255, 0, 0), (0, 180, 0), (0, 0, 255)]


def parse_args():
    p = argparse.ArgumentParser(description="YOLO 标注可视化抽查（Pillow）")
    p.add_argument("--root", required=True, help="数据集根目录")
    p.add_argument("--split", choices=("train", "valid", "test"), default="train")
    p.add_argument("--num", type=positive_int, default=30, help="抽查数量")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--out", default="outputs/vis", help="可视化输出目录")
    p.add_argument("--workers", type=positive_int, default=4, help="导出线程数；1 为串行")
    return p.parse_args()


def draw_one(job):
    """每个任务只处理自己的图片；用上下文管理器及时关闭文件句柄。"""
    from PIL import Image, ImageDraw

    image_path, label_path, out_path = job
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        try:
            width, height = image.size
            draw = ImageDraw.Draw(image)
            count, bad, missing = 0, 0, False
            try:
                with label_path.open(encoding="utf-8-sig") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            category, (xc, yc, bw, bh) = parse_label(line, len(CORN_NAMES))
                        except ValueError:
                            bad += 1
                            continue
                        x1, y1 = max(0, (xc - bw / 2) * width), max(0, (yc - bh / 2) * height)
                        x2, y2 = min(width, (xc + bw / 2) * width), min(height, (yc + bh / 2) * height)
                        color = COLORS[category]
                        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                        draw.text((x1 + 2, max(0, y1 - 12)), CORN_NAMES[category], fill=color)
                        count += 1
            except FileNotFoundError:
                missing = True
            if out_path.is_symlink() or (out_path.exists() and out_path.stat().st_nlink > 1):
                raise ValueError("输出文件不能指向原始图片")
            image.save(out_path)
        finally:
            image.close()
        warnings = f"，异常标注 {bad} 行" if bad else ""
        warnings += "，缺失标签" if missing else ""
        return True, f"{image_path.name}: {count} 个框{warnings} -> {out_path}"
    except (OSError, ValueError) as error:
        return False, f"[错误] {image_path.name}: {error}"


def main():
    args = parse_args()
    # 先检查依赖，避免工作线程里重复输出同一个缺失依赖错误。
    import PIL  # noqa: F401

    started = perf_counter()
    root = project_path(args.root)
    image_dir, label_dir = root / args.split / "images", root / args.split / "labels"
    out_dir = project_path(args.out) / args.split
    separate_output(root, out_dir)
    images = image_files(image_dir)
    if not images:
        raise ValueError(f"图片目录为空: {image_dir}")
    picked = random.Random(args.seed).sample(images, min(args.num, len(images)))
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = ((image, label_dir / f"{image.stem}.txt", out_dir / image.name) for image in picked)
    success = 0
    for ok, message in parallel_map(draw_one, jobs, args.workers):
        success += int(ok)
        print(message)
    print(f"成功 {success}/{len(picked)} 张；耗时 {perf_counter() - started:.2f} 秒")
    print("请人工确认框的位置、漏标和错标；空标签不等于健康叶片")
    if success != len(picked):
        raise SystemExit(1)


if __name__ == "__main__":
    run_main(main)
