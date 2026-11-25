"""小车端 BLE 传输与运动安全（参考骨架）。

包含两部分：
  1) MoveWatchdog：运动看门狗，防手机断连后小车自己乱跑；
  2) 图片分包发送：把 JPEG 按 MTU 切片从图像通道发出。

真正把消息写到蓝牙由具体 BLE 栈完成（Linux 可用 bleak 做 peripheral，
或让 ESP32 做 BLE 桥、本机走串口）。这里只给出与协议相关的逻辑。
"""

import time

from protocol import (  # 导入同目录的 protocol.py
    T_MOVE, T_STOP, T_ESTOP, T_MODE, T_CAPTURE, T_PING,
    T_STATUS, T_DETECT, T_IMAGE, T_ACK, T_ERROR, T_PONG,
    parse_command, move, stop, e_stop, set_mode, capture, ping,
    status, detect, image_head, ack, error, pong,
    REQUEST_MTU,
)


class MoveWatchdog:
    """看门狗：收到 mv 后启动 500ms 定时器，超时没新指令就自动停车。est 立即断电机。"""

    TIMEOUT_S = 0.5

    def __init__(self, on_timeout):
        self._last = 0.0
        self._active = False
        self._on_timeout = on_timeout  # 触发时回调（应执行停车/断电机）

    def feed(self):
        """每收到一条 mv 调一次，刷新计时。"""
        self._last = time.monotonic()
        self._active = True

    def clear(self):
        """收到 stp/est 时调一次，停止看门狗。"""
        self._active = False

    def tick(self):
        """在主循环里周期性调用；超时则触发停车。"""
        if self._active and (time.monotonic() - self._last) > self.TIMEOUT_S:
            self._active = False
            self._on_timeout()


def chunk_image(jpeg: bytes, mtu: int = REQUEST_MTU):
    """把一张 JPEG 按 (MTU-3) 切片，yield 每一包二进制（供图像通道 NOTIFY）。"""
    max_payload = max(20, mtu - 3)  # ATT 头占 3 字节
    for i in range(0, len(jpeg), max_payload):
        yield jpeg[i:i + max_payload]


class CarNode:
    """小车节点示例：把协议解析、看门狗、识别回调串起来（传输层用回调注入）。"""

    def __init__(self, send_notify, send_image):
        self._send_notify = send_notify   # 发 JSON 文本（文字通道）
        self._send_image = send_image     # 发二进制分包（图像通道）
        self._wd = MoveWatchdog(self._safe_stop)
        self._seq = 0

    def _safe_stop(self):
        # 看门狗超时：自动停车（实际项目里这里还应断电机驱动）
        self._send_notify(stop())

    def on_command(self, raw: bytes):
        """手机在指令通道发来的数据，交给这里处理。"""
        t, obj = parse_command(raw)
        if t is None:
            return
        if t == T_MOVE:
            s, a = obj["s"], obj["a"]
            # TODO: 把 s/a 映射成左右轮速度，驱动小车
            self._wd.feed()               # 刷新看门狗
        elif t == T_STOP:
            self._wd.clear()
            self._send_notify(stop())     # TODO: 实际停车
        elif t == T_ESTOP:
            self._wd.clear()
            self._send_notify(e_stop())   # TODO: 立即断电机
        elif t == T_MODE:
            self._send_notify(ack(T_MODE, True))   # TODO: 切换 manual/auto
        elif t == T_CAPTURE:
            self._send_capture()          # 抓拍并回传
        elif t == T_PING:
            self._send_notify(pong())

    def on_detect(self, cls, conf, x, y, ts):
        """识别回调：车端跑 best.pt 得到结果后调用，回传文字。"""
        self._send_notify(detect(cls, conf, x, y, ts))

    def _send_capture(self):
        """抓拍 + 识别画框 + 压缩成小 JPEG + 发头 + 发二进制。"""
        # TODO: 调用摄像头抓帧、用模型推理、画框、cv2.imencode 成 JPEG
        jpeg = b""  # 占位：真实 JPEG 字节
        self._seq += 1
        self._send_notify(image_head(self._seq, w=320, h=240, fmt="jpg", size=len(jpeg)))
        for chunk in chunk_image(jpeg):
            self._send_image(chunk)

    def loop(self):
        """主循环（示例）。"""
        while True:
            self._wd.tick()
            time.sleep(0.05)
