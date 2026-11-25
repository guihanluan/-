package com.pestcar.control

import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCharacteristic
import java.io.ByteArrayOutputStream

/**
 * 手机端 BLE 控制核心（骨架）。
 *
 * 职责：
 *  1) 把摇杆/按钮转成协议消息，写入指令通道；
 *  2) 处理小车在文字/图像通道发来的数据；
 *  3) 在图像通道按 seq 把分包字节拼成完整 JPEG。
 *
 * 说明：本类只管「协议与通道」，真正的连接/权限/线程由调用方（Activity/ViewModel）负责。
 */

class ImageAssembler(
    private val onImageComplete: (seq: Int, bytes: ByteArray) -> Unit
) {
    private var expectSeq = -1
    private var expectSize = 0
    private var received = 0
    private val buffer = ByteArrayOutputStream()

    /** 收到图像头（img JSON）时调用，开始接收一张图 */
    fun begin(head: CarProtocol.ImageHead) {
        expectSeq = head.seq
        expectSize = head.size
        received = 0
        buffer.reset()
    }

    /** 收到图像通道（FF03）的一包二进制时调用 */
    fun append(chunk: ByteArray) {
        if (expectSeq < 0) return            // 还没收到头，丢弃
        buffer.write(chunk)
        received += chunk.size
        if (received >= expectSize) {        // 收满，回调完整图片并复位
            val full = buffer.toByteArray()
            val seq = expectSeq
            expectSeq = -1
            expectSize = 0
            received = 0
            buffer.reset()
            onImageComplete(seq, full)
        }
    }
}

class BleCarController(private val gatt: BluetoothGatt) {

    /** 收到的解析结果通过回调暴露给 UI */
    var onStatus: ((CarProtocol.Status) -> Unit)? = null
    var onDetect: ((CarProtocol.Detect) -> Unit)? = null
    var onImage:  ((seq: Int, bytes: ByteArray) -> Unit)? = null
    var onError:  ((code: Int, msg: String) -> Unit)? = null

    private val imageAssembler = ImageAssembler { seq, bytes ->
        onImage?.invoke(seq, bytes)   // bytes 为完整 JPEG，可 BitmapFactory.decodeByteArray 显示
    }

    /** 向指令通道写入一条协议消息（无响应写，延迟最低） */
    fun send(cmd: String) {
        val ch = gatt.getService(CarProtocol.SERVICE_UUID)
            .getCharacteristic(CarProtocol.CHAR_CMD_UUID)
        ch.writeType = BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE
        ch.value = cmd.toByteArray()
        gatt.writeCharacteristic(ch)
    }

    /** 便捷控制方法：直接调用协议构造器 */
    fun move(speed: Float, angle: Float) = send(CarProtocol.move(speed, angle))
    fun stop() = send(CarProtocol.stop())
    fun eStop() = send(CarProtocol.eStop())
    fun setMode(mode: String) = send(CarProtocol.setMode(mode))
    fun capture() = send(CarProtocol.capture())
    fun ping() = send(CarProtocol.ping())

    /**
     * 处理小车发来的数据；由 BluetoothGattCallback.onCharacteristicChanged 调用。
     * @param uuid   触发该数据的特征值 UUID
     * @param value  原始字节
     */
    fun onCharacteristicChanged(uuid: String, value: ByteArray) {
        when (uuid) {
            // 文字通道：JSON 消息
            CarProtocol.CHAR_NOTIFY_UUID -> {
                val parsed = CarProtocol.parse(String(value)) ?: return
                val (type, obj) = parsed
                when (type) {
                    CarProtocol.T_STATUS -> onStatus?.invoke(CarProtocol.parseStatus(obj))
                    CarProtocol.T_DETECT -> onDetect?.invoke(CarProtocol.parseDetect(obj))
                    CarProtocol.T_IMAGE  -> imageAssembler.begin(CarProtocol.parseImageHead(obj))
                    CarProtocol.T_ERROR  -> onError?.invoke(obj.getInt("c"), obj.getString("m"))
                    // T_ACK / T_PONG 可按需处理（如更新连接状态）
                }
            }
            // 图像通道：二进制分包
            CarProtocol.CHAR_IMAGE_UUID -> imageAssembler.append(value)
        }
    }

    // 提示：连接建立后调用 gatt.requestMtu(CarProtocol.REQUEST_MTU) 提升图传速度。
}
