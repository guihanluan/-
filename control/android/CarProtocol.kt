package com.pestcar.control

import android.util.Base64
import org.json.JSONObject
import java.io.ByteArrayOutputStream

/**
 * 小车运动控制 JSON 协议定义（手机端 / Android）
 *
 * 协议分层：
 *  - 应用层：本协议定义的 JSON 文本消息
 *  - 传输层：BLE GATT（见下方 UUID 常量）
 *
 * 通信方向：双向
 *  - 下行（手机 → 小车）：控制指令
 *  - 上行（小车 → 手机）：状态回报 / 识别结果 / 抓拍图（二进制，走独立图像通道）
 *
 * 与 control/protocol.json、control/car/protocol.py 三处保持字段一致。
 */
object CarProtocol {

    // ============ BLE GATT 通道定义 ============
    /** 服务 UUID */
    const val SERVICE_UUID = "0000ff00-0000-1000-8000-00805f9b34fb"

    /** 指令通道：手机 → 小车（WRITE_NO_RESPONSE，连续发摇杆指令不等待回执，延迟最低） */
    const val CHAR_CMD_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"

    /** 文字通道：小车 → 手机（NOTIFY），传 JSON 文本 */
    const val CHAR_NOTIFY_UUID = "0000ff02-0000-1000-8000-00805f9b34fb"

    /** 图像通道：小车 → 手机（NOTIFY），传图片二进制字节流 */
    const val CHAR_IMAGE_UUID = "0000ff03-0000-1000-8000-00805f9b34fb"

    /** 连接后建议协商的 MTU，越大图传越快（默认仅 23 字节） */
    const val REQUEST_MTU = 128

    // ============ 消息类型（统一用 t 字段表示） ============
    const val T_MOVE   = "mv"    // 摇杆控制
    const val T_STOP   = "stp"   // 停车
    const val T_ESTOP  = "est"   // 急停（最高优先级）
    const val T_MODE   = "md"    // 切换模式 manual/auto
    const val T_CAPTURE = "cap"  // 请求抓拍识别图
    const val T_PING   = "pg"    // 心跳

    const val T_STATUS = "sta"   // 状态回报
    const val T_DETECT = "dt"    // 识别结果
    const val T_IMAGE  = "img"   // 图像头（随后在图像通道发二进制）
    const val T_ACK    = "ak"    // 指令回执
    const val T_ERROR  = "er"    // 错误
    const val T_PONG   = "pong"  // 心跳回复

    // ============ 下行（手机 → 车）消息构造 ============

    /** 摇杆控制：speed ∈ [-1,1]，angle ∈ [-45,45] 度，正=右转 */
    fun move(speed: Float, angle: Float): String =
        build(T_MOVE) { put("s", speed); put("a", angle) }

    /** 停车 */
    fun stop(): String = build(T_STOP)

    /** 急停：立即切断电机，优先级最高 */
    fun eStop(): String = build(T_ESTOP)

    /** 切换模式：manual=手动，auto=自动巡田 */
    fun setMode(mode: String): String = build(T_MODE) { put("m", mode) }

    /** 请求抓拍一张带识别框的图（q=1 表示请求） */
    fun capture(): String = build(T_CAPTURE) { put("q", 1) }

    /** 心跳保活 */
    fun ping(): String = build(T_PING)

    // ============ 上行（车 → 手机）数据模型与解析 ============

    /** 状态回报 */
    data class Status(val battery: Int, val speed: Float, val angle: Float, val mode: String)

    /** 识别结果 */
    data class Detect(val cls: String, val conf: Float, val x: Int, val y: Int, val ts: Long)

    /** 图像头：随后在图像通道连续收到 sz 字节 JPEG */
    data class ImageHead(val seq: Int, val w: Int, val h: Int, val fmt: String, val size: Int)

    /** 指令回执：ref = 被确认的原指令类型（mv/stp/md/cap/pg），ok = 车端是否接收成功 */
    data class Ack(val ref: String, val ok: Boolean)

    /**
     * 把车端发来的 JSON 文本解析为 (类型, JSONObject)；
     * 解析失败（非 JSON / 缺 t 字段）返回 null。
     */
    fun parse(jsonText: String): Pair<String, JSONObject>? = runCatching {
        val obj = JSONObject(jsonText)
        obj.getString("t") to obj
    }.getOrNull()

    fun parseStatus(o: JSONObject) = Status(
        battery = o.getInt("b"),
        speed = o.getDouble("s").toFloat(),
        angle = o.getDouble("a").toFloat(),
        mode = o.getString("m")
    )

    fun parseDetect(o: JSONObject) = Detect(
        cls = o.getString("c"),
        conf = o.getDouble("p").toFloat(),
        x = o.getInt("x"),
        y = o.getInt("y"),
        ts = o.getLong("ts")
    )

    fun parseImageHead(o: JSONObject) = ImageHead(
        seq = o.getInt("seq"),
        w = o.getInt("w"),
        h = o.getInt("h"),
        fmt = o.getString("fmt"),
        size = o.getInt("sz")
    )

    /** r 用数字 1/0 表示，这里转成 Boolean，避免端上两种写法的分歧 */
    fun parseAck(o: JSONObject) = Ack(
        ref = o.getString("ref"),
        ok = o.optInt("r", 0) == 1
    )

    // ============ 内部工具 ============

    /** 构造一条消息：固定带 t 字段，再追加业务字段 */
    private inline fun build(type: String, block: JSONObject.() -> Unit): String =
        JSONObject().apply { put("t", type); block() }.toString()
}
