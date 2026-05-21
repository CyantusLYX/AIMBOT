package com.cyantus.aimbot.eye.network

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.min

object PacketProtocol {
    val MAGIC = byteArrayOf('G'.code.toByte(), 'B'.code.toByte(), 'R'.code.toByte(), '1'.code.toByte())
    const val VERSION: Byte = 1
    const val TYPE_VIDEO_FRAGMENT: Byte = 1
    const val TYPE_IMU_SAMPLE: Byte = 2
    const val COMMON_HEADER_BYTES = 24
    const val VIDEO_HEADER_BYTES = 84
    const val IMU_HEADER_BYTES = 68
    const val MAX_DATAGRAM_BYTES = 1460
    const val VIDEO_PAYLOAD_BYTES = MAX_DATAGRAM_BYTES - VIDEO_HEADER_BYTES
}

data class ImuSample(
    val sequence: Int,
    val deviceTimeNs: Long,
    val roll: Float,
    val pitch: Float,
    val yaw: Float,
    val gyroX: Float,
    val gyroY: Float,
    val gyroZ: Float,
    val accelX: Float,
    val accelY: Float,
    val accelZ: Float,
    val sourceFlags: Short,
)

object PacketWriter {
    fun videoFragmentCount(jpegLength: Int): Int {
        require(jpegLength >= 0) { "jpegLength must be non-negative" }
        if (jpegLength == 0) {
            return 1
        }
        return ceil(jpegLength.toDouble() / PacketProtocol.VIDEO_PAYLOAD_BYTES.toDouble()).toInt()
    }

    fun buildVideoFragment(
        sequence: Int,
        frameId: Int,
        fragmentIndex: Int,
        fragmentCount: Int,
        jpegBytes: ByteArray,
        payloadOffset: Int,
        width: Int,
        height: Int,
        jpegQuality: Int,
        imuSample: ImuSample,
    ): ByteArray {
        require(fragmentIndex in 0 until fragmentCount) { "fragmentIndex out of range" }
        require(payloadOffset in 0..jpegBytes.size) { "payloadOffset out of range" }
        val payloadLength = min(PacketProtocol.VIDEO_PAYLOAD_BYTES, jpegBytes.size - payloadOffset)
        val buffer = ByteBuffer
            .allocate(PacketProtocol.VIDEO_HEADER_BYTES + payloadLength)
            .order(ByteOrder.LITTLE_ENDIAN)

        writeCommonHeader(
            buffer = buffer,
            packetType = PacketProtocol.TYPE_VIDEO_FRAGMENT,
            headerLength = PacketProtocol.VIDEO_HEADER_BYTES,
            sequence = sequence,
            deviceTimeNs = System.nanoTime(),
        )
        buffer.putInt(frameId)
        buffer.putShort(fragmentIndex.toShort())
        buffer.putShort(fragmentCount.toShort())
        buffer.putInt(jpegBytes.size)
        buffer.putShort(width.toShort())
        buffer.putShort(height.toShort())
        buffer.put(jpegQuality.coerceIn(1, 100).toByte())
        buffer.put(byteArrayOf(0, 0, 0))
        buffer.putInt(imuSample.sequence)
        putImuFloats(buffer, imuSample)
        buffer.put(jpegBytes, payloadOffset, payloadLength)
        return buffer.array()
    }

    fun buildImuPacket(sequence: Int, sample: ImuSample): ByteArray {
        val buffer = ByteBuffer
            .allocate(PacketProtocol.IMU_HEADER_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)

        writeCommonHeader(
            buffer = buffer,
            packetType = PacketProtocol.TYPE_IMU_SAMPLE,
            headerLength = PacketProtocol.IMU_HEADER_BYTES,
            sequence = sequence,
            deviceTimeNs = sample.deviceTimeNs,
        )
        buffer.putInt(sample.sequence)
        putImuFloats(buffer, sample)
        buffer.putShort(sample.sourceFlags)
        buffer.putShort(0)
        return buffer.array()
    }

    private fun writeCommonHeader(
        buffer: ByteBuffer,
        packetType: Byte,
        headerLength: Int,
        sequence: Int,
        deviceTimeNs: Long,
    ) {
        buffer.put(PacketProtocol.MAGIC)
        buffer.put(PacketProtocol.VERSION)
        buffer.put(packetType)
        buffer.putShort(headerLength.toShort())
        buffer.putShort(0)
        buffer.putShort(0)
        buffer.putInt(sequence)
        buffer.putLong(deviceTimeNs)
    }

    private fun putImuFloats(buffer: ByteBuffer, sample: ImuSample) {
        buffer.putFloat(sample.roll)
        buffer.putFloat(sample.pitch)
        buffer.putFloat(sample.yaw)
        buffer.putFloat(sample.gyroX)
        buffer.putFloat(sample.gyroY)
        buffer.putFloat(sample.gyroZ)
        buffer.putFloat(sample.accelX)
        buffer.putFloat(sample.accelY)
        buffer.putFloat(sample.accelZ)
    }
}
