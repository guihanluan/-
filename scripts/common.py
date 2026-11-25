"""脚本共享工具：项目路径、文件扫描、标签解析和少量 I/O 并发。"""
import argparse
import hashlib
import math
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CORN_NAMES = ["Corn Rust", "Grey Leaf Spot", "Leaf Blight"]


def project_path(value):
    """相对参数统一从项目根目录解析，兼容 PyCharm 的不同工作目录。"""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("不能小于 0")
    return number


def batch_size(value):
    number = int(value)
    if number != -1 and number < 1:
        raise argparse.ArgumentTypeError("batch 必须是正整数，或 -1（GPU 自动选择）")
    return number


def scan_files(directory, suffixes):
    """一次扫描筛选文件，避免为图片逐个重复查询文件属性。"""
    with os.scandir(directory) as entries:
        paths = [Path(entry.path) for entry in entries
                 if Path(entry.name).suffix.lower() in suffixes and entry.is_file()]
    return sorted(paths)


def image_files(directory):
    return scan_files(directory, IMAGE_SUFFIXES)


def label_files(directory):
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in scan_files(directory, {".txt"})}


def label_has_content(path):
    """读到第一条非空行就返回；不把整个标签读入列表。"""
    with path.open(encoding="utf-8-sig") as handle:
        return any(line.strip() for line in handle)


def parse_label(line, nc):
    """校验类别、有限坐标、框大小和边界；拒绝把 1.5 截断成类别 1。"""
    fields = line.split()
    if len(fields) != 5:
        raise ValueError("应有 5 个字段")
    category = float(fields[0])
    if not math.isfinite(category) or not category.is_integer() or not 0 <= category < nc:
        raise ValueError("类别编号不是有效整数")
    coords = tuple(map(float, fields[1:]))
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in coords):
        raise ValueError("坐标必须为 0~1 的有限数值")
    xc, yc, width, height = coords
    if width <= 0 or height <= 0:
        raise ValueError("框的宽高必须大于 0")
    # 容许归一化小数舍入造成的微小误差。
    tolerance = 1e-6
    if (xc - width / 2 < -tolerance or yc - height / 2 < -tolerance
            or xc + width / 2 > 1 + tolerance or yc + height / 2 > 1 + tolerance):
        raise ValueError("标注框超出图片边界")
    return int(category), coords


def parallel_map(function, items, workers):
    """有序消费线程结果，保持报告可复现；workers=1 可退回串行。"""
    if workers <= 1:
        yield from map(function, items)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(function, items)


def separate_output(source, output):
    """拒绝与原始数据目录重叠的输出路径（含符号链接解析）。"""
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(f"输出路径不能与原始数据目录重叠: {output}")


def copy_if_changed(source, destination):
    """保留时间戳；尺寸和修改时间相同时省掉重复复制。"""
    source_stat = source.stat()
    try:
        destination_stat = destination.stat(follow_symlinks=False)
    except FileNotFoundError:
        destination_stat = None
    if destination_stat is not None:
        # 使用已有 stat 结果比较文件身份，省掉 samefile 的两次属性查询。
        if (stat.S_ISLNK(destination_stat.st_mode) or destination_stat.st_nlink > 1
                or (source_stat.st_dev, source_stat.st_ino)
                == (destination_stat.st_dev, destination_stat.st_ino)):
            raise ValueError(f"目标文件不能是符号链接、硬链接或原文件: {destination}")
        if (source_stat.st_size == destination_stat.st_size
                and source_stat.st_mtime_ns == destination_stat.st_mtime_ns):
            return False
    shutil.copy2(source, destination)
    return True


def load_yolo():
    """延迟加载模型依赖，并关闭 Ultralytics 的自动安装行为。"""
    try:
        from ultralytics import YOLO
        from ultralytics.utils import checks
    except ImportError as error:
        raise ImportError("当前 PyCharm 解释器缺少 ultralytics 或其依赖，请检查所选环境") from error
    checks.AUTOINSTALL = False
    return YOLO


def normalize_device(value):
    """统一 cuda 别名，并在训练前给出设备错误。"""
    import torch

    device = value.strip().lower()
    device = "0" if device == "cuda" else device.removeprefix("cuda:")
    if device == "cpu":
        return device
    indexes = device.split(",")
    if not all(index.isdigit() for index in indexes):
        raise ValueError("device 请填写 cpu、0、cuda:0 或 0,1")
    if not torch.cuda.is_available():
        raise ValueError("当前解释器无法使用 CUDA；请检查 GPU 环境，或设置 --device cpu")
    if len(set(indexes)) != len(indexes):
        raise ValueError("device 中的 GPU 编号不能重复")
    if any(int(index) >= torch.cuda.device_count() for index in indexes):
        raise ValueError(f"GPU 编号越界，可用设备数: {torch.cuda.device_count()}")
    return device


def resolved_dataset_yaml(value, required_splits=("train", "val")):
    """生成运行用绝对路径副本，不修改用户的可迁移 YAML 或全局设置。"""
    import yaml

    source = project_path(value)
    with source.open(encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"数据配置必须是 YAML 字典: {source}")
    names = data.get("names")
    if not isinstance(names, (list, dict)) or not names:
        raise ValueError(f"数据配置必须包含非空 names 列表或字典: {source}")
    if "nc" in data and data["nc"] != len(names):
        raise ValueError(f"nc 与 names 数量不一致: {source}")
    data["nc"] = len(names)
    if data.get("path") is not None and not isinstance(data["path"], str):
        raise ValueError(f"path 必须是路径字符串: {source}")
    root = Path(data.get("path") or ".").expanduser()
    root = (root if root.is_absolute() else source.parent / root).resolve()
    if not root.is_dir():
        raise ValueError(f"数据集根目录不存在: {root}（path 相对于 YAML 所在目录）")
    data["path"] = root.as_posix()
    # 不执行数据配置中可能存在的下载脚本。
    data.pop("download", None)
    for split in ("train", "val", "test"):
        if not data.get(split):
            if split in required_splits:
                raise ValueError(f"数据配置缺少 {split}: {source}")
            continue
        values = data[split] if isinstance(data[split], list) else [data[split]]
        if not values or not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"{split} 必须是路径字符串或非空字符串列表: {source}")
        resolved = []
        for item in values:
            path = (root / item).resolve()
            if split in required_splits and not path.exists():
                raise ValueError(f"{split} 路径不存在: {path}")
            resolved.append(path.as_posix())
        data[split] = resolved if isinstance(data[split], list) else resolved[0]
    target = PROJECT_ROOT / "outputs" / "resolved_configs"
    separate_output(root, target)
    target.mkdir(parents=True, exist_ok=True)
    # 不同来源配置使用不同文件名，防止重名覆盖。
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    output = target / f"{source.stem}-{digest}.yaml"
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    if not output.exists() or output.read_text(encoding="utf-8") != text:
        output.write_text(text, encoding="utf-8")
    return output


def run_main(function):
    """常见输入/文件错误用中文说明，同时返回非零退出码。"""
    try:
        function()
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(f"[错误] {error}") from error
