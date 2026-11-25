"""独立的无损解码缓存：只换图片读取方式，沿用原数据增强与训练逻辑。"""
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from time import perf_counter

from common import parallel_map, project_path, separate_output


def _regular_output(path):
    """缓存文件不能通过链接覆盖其他文件。"""
    if path.is_symlink():
        raise ValueError(f"缓存文件不能是符号链接: {path}")
    if path.exists():
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
            raise ValueError(f"缓存文件必须是独立的普通文件: {path}")


def prepare_dataset_cache(dataset, root, workers=4):
    """逐像素核对缓存；源内容变化或缓存损坏时重建，不依赖尺寸/时间戳。"""
    import numpy as np
    from ultralytics.utils.patches import imread

    started = perf_counter()
    root = project_path(root)
    if dataset.cache is not None:
        raise ValueError("独立解码缓存请配合 --cache none，避免改变原有缓存/增强行为")
    if not dataset.im_files:
        raise ValueError("无法缓存空数据集")
    sources = [Path(value).resolve() for value in dataset.im_files]
    if len(set(sources)) != len(sources):
        raise ValueError("数据配置包含重复图片，请先检查 train/val 路径")
    # 同一图片目录只检查一次，避免上万次重复解析相同路径。
    for directory in {source.parent for source in sources}:
        separate_output(directory, root)
    root.mkdir(parents=True, exist_ok=True)
    # 固定样本估算空间，不使用 random，保持训练随机数序列。
    sample = imread(str(sources[0]), flags=dataset.cv2_flag)
    if sample is None:
        raise ValueError(f"无法读取图片: {sources[0]}")
    estimate = int((sample.nbytes + 256) * len(sources) * 1.5)
    if shutil.disk_usage(root).free < estimate:
        raise ValueError(f"解码缓存空间不足，建议至少保留 {estimate / 2**30:.1f} GB")

    def prepare(source):
        # 使用与当前 Ultralytics load_image 完全相同的解码函数和颜色标志。
        image = imread(str(source), flags=dataset.cv2_flag)
        if image is None:
            raise ValueError(f"无法读取图片: {source}")
        original_cache = source.with_suffix(".npy")
        if original_cache.exists():
            # 原框架优先读取图片旁的 npy；不悄悄替换与原图不一致的旧缓存。
            existing = np.load(original_cache, allow_pickle=False)
            if not isinstance(existing, np.ndarray):
                existing.close()
                raise ValueError(f"图片旁的旧缓存不是像素数组，请检查: {original_cache}")
            if existing.dtype != image.dtype or not np.array_equal(existing, image):
                raise ValueError(f"图片旁的旧缓存与原图不一致，请检查: {original_cache}")
        key = hashlib.sha256(f"{source}|{dataset.cv2_flag}".encode("utf-8")).hexdigest()
        target = root / f"{key}.npy"
        _regular_output(target)
        try:
            cached = np.load(target, allow_pickle=False)
            valid = (isinstance(cached, np.ndarray) and cached.dtype == image.dtype
                     and np.array_equal(cached, image))
            if not isinstance(cached, np.ndarray):
                cached.close()
        except (OSError, ValueError, EOFError):
            valid = False
        if not valid:
            temporary = target.with_suffix(".tmp")
            _regular_output(temporary)
            try:
                with temporary.open("wb") as handle:
                    np.save(handle, image, allow_pickle=False)
                # 写入后再次核对，成功后才替换缓存。
                restored = np.load(temporary, allow_pickle=False)
                if restored.dtype != image.dtype or not np.array_equal(restored, image):
                    raise ValueError(f"缓存写入校验失败: {target}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return target, not valid, target.stat().st_size

    results = list(parallel_map(prepare, sources, workers))
    # 不改 load_image、buffer、transforms 或抽样顺序，只指定独立的 npy 路径。
    dataset.npy_files = [result[0] for result in results]
    report = {
        "images": len(results), "rebuilt": sum(result[1] for result in results),
        "reused": sum(not result[1] for result in results),
        "bytes": sum(result[2] for result in results),
        "seconds": perf_counter() - started, "root": str(root),
        "pixel_equality_checked": True,
    }
    return report


def decoded_cache_trainer(root, workers=4):
    """返回项目内训练器；不修改安装环境，也不改变模型数值配置。"""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class DecodedCacheTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            report = prepare_dataset_cache(dataset, root, workers)
            print(f"[{mode}] 无损缓存: {report['images']} 张，"
                  f"新建/更新 {report['rebuilt']}，复用 {report['reused']}，"
                  f"逐像素核对通过，用时 {report['seconds']:.1f} 秒")
            reports = getattr(self, "decoded_cache_reports", {})
            reports[mode] = report
            self.decoded_cache_reports = reports
            (self.save_dir / "image_cache_report.json").write_text(
                json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
            return dataset

    return DecodedCacheTrainer
