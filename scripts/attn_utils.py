# 注意力实验公共工具：模块注册 + 按层顺序迁移权重 + 存为可训练检查点
# 关键背景：YOLO(yaml) 构建后 self.ckpt 为 None，train() 会按 cfg 重建模型，
# 之前直接 state_dict 拷贝的权重会被丢弃，之前“微调”实际是从零训练。
# 正确做法：按层顺序迁移 -> torch.save 存 .pt -> YOLO(pt) 再训练。
import sys
from copy import deepcopy
from pathlib import Path

# 把工程根目录加入路径，保证 scripts 下运行也能 import models
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

CUSTOM_MODULE_NAMES = ("CBAM", "SE", "ECA")


def register_custom_modules():
    """注册 CBAM/SE/ECA，让 ultralytics 解析 yaml 时认识这些层。"""
    from models.cbam import CBAM  # noqa: F401
    from models.eca import ECA  # noqa: F401
    from models.se import SE  # noqa: F401

    import ultralytics.nn.modules as ULM
    import ultralytics.nn.tasks as ULT

    for mod in (ULM, ULT):
        for name, cls in (("CBAM", CBAM), ("SE", SE), ("ECA", ECA)):
            try:
                setattr(mod, name, cls)
            except Exception:
                pass
    return {"CBAM": CBAM, "SE": SE, "ECA": ECA}


def _type_name(m):
    """取 Sequential 中每层的模块类型名。"""
    return type(m).__name__


def transfer_by_order(src_model, dst_model, verbose=True):
    """按层顺序把源模型权重复制到目标模型，跳过目标中新增的自定义注意力层。

    为什么不用 state_dict 按名拷贝：插入新层后层序号整体后移，
    按名拷贝会导致 Neck/Head 大量错位，只剩前几层能对上。
    """
    src_layers = list(src_model.model)
    dst_layers = list(dst_model.model)
    i = j = 0
    copied, skipped, mismatched = 0, 0, []
    while i < len(src_layers) and j < len(dst_layers):
        dst_name = _type_name(dst_layers[j])
        if dst_name in CUSTOM_MODULE_NAMES:
            # 目标新增层：保持随机初始化，只计数
            j += 1
            skipped += 1
            continue
        src_name = _type_name(src_layers[i])
        if src_name == dst_name:
            src_sd = src_layers[i].state_dict()
            if src_sd:
                try:
                    dst_layers[j].load_state_dict(src_sd, strict=True)
                    copied += 1
                except RuntimeError as e:
                    mismatched.append((j, dst_name, str(e)[:100]))
            i += 1
            j += 1
        else:
            # 源层在目标中不存在，跳过源层继续对齐
            mismatched.append((i, f"src:{src_name} vs dst:{dst_name}", "skip-src"))
            i += 1
    if verbose:
        print(f"按层迁移：复制 {copied} 层，跳过新增注意力层 {skipped} 个，对齐异常 {len(mismatched)} 处。")
        for item in mismatched[:10]:
            print(f"  - {item}")
    return copied, skipped, mismatched


def save_as_checkpoint(model, path):
    """把已迁移权重的模型存为 .pt，使 YOLO(pt).train() 能正确加载全部权重。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": deepcopy(model.model).float()}, str(path))
    return path
