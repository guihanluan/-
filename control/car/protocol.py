"""小车端 JSON 协议（Python 参考实现，与手机端 CarProtocol.kt 对应）。

只负责「消息的构造与解析」，不涉及具体 BLE / 电机 / 摄像头驱动，
方便在小车（如树莓派 / Jetson，用 best.pt 做识别）上复用。
"""

import json
from dataclasses import dataclass

# ---- BLE GATT 通道 UUID（与手机端保持一致）----
SERVICE_UUID = "0000ff00-0000-1000-8000-00805f9b34fb"
CHAR_CMD_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"
CHAR_NOTIFY_UUID = "0000ff02-0000-1000-8000-00805f9b34fb"
CHAR_IMAGE_UUID = "0000ff03-0000-1000-8000-00805f9b34fb"
REQUEST_MTU = 128

# ---- 消息类型 ----
T_MOVE, T_STOP, T_ESTOP = "mv", "stp", "est"
T_MODE, T_CAPTURE, T_PING = "md", "cap", "pg"
T_STATUS, T_DETECT, T_IMAGE = "sta", "dt", "img"
T_ACK, T_ERROR, T_PONG = "ak", "er", "pong"


def _msg(t: str, **fields) -> bytes:
    """构造一条消息并编码为 UTF-8 字节（BLE 写入用）。separators 去掉多余空格省字节。"""
    obj = {"t": t, **fields}
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


# ---- 下行（手机 → 车）指令解析 ----
def parse_command(raw: bytes):
    """解析手机发来的指令，返回 (type, dict)；解析失败返回 (None, None)。"""
    try:
        obj = json.loads(raw.decode("utf-8"))
        return obj.get("t"), obj
    except Exception:
        return None, None


# ---- 上行（车 → 手机）消息构造 ----
def move(speed: float, angle: float) -> bytes:
    return _msg(T_MOVE, s=speed, a=angle)

def stop() -> bytes:
    return _msg(T_STOP)

def e_stop() -> bytes:
    return _msg(T_ESTOP)

def set_mode(mode: str) -> bytes:
    return _msg(T_MODE, m=mode)

def capture() -> bytes:
    return _msg(T_CAPTURE, q=1)

def ping() -> bytes:
    return _msg(T_PING)

def status(battery: int, speed: float, angle: float, mode: str) -> bytes:
    return _msg(T_STATUS, b=battery, s=speed, a=angle, m=mode)

def detect(cls: str, conf: float, x: int, y: int, ts: int) -> bytes:
    return _msg(T_DETECT, c=cls, p=conf, x=x, y=y, ts=ts)

def image_head(seq: int, w: int, h: int, fmt: str, size: int) -> bytes:
    """图像头：随后把 JPEG 字节按 MTU 切片从图像通道发出。"""
    return _msg(T_IMAGE, seq=seq, w=w, h=h, fmt=fmt, sz=size)

def ack(orig_type: str, ok: bool) -> bytes:
    return _msg(T_ACK, t=orig_type, r=1 if ok else 0)

def error(code: int, msg: str) -> bytes:
    return _msg(T_ERROR, c=code, m=msg)

def pong() -> bytes:
    return _msg(T_PONG)
