package com.cyantus.aimbot.eye.network

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PacketProtocolTest {
    private val imu = ImuSample(
        sequence = 7,
        deviceTimeNs = 1234L,
        roll = 1f,
        pitch = 2f,
        yaw = 3f,
        gyroX = 4f,
        gyroY = 5f,
        gyroZ = 6f,
        accelX = 7f,
        accelY = 8f,
        accelZ = 9f,
        sourceFlags = 7,
    )

    @Test
    fun imuPacketHasExpectedHeaderSizeAndOffsets() {
        val packet = PacketWriter.buildImuPacket(sequence = 99, sample = imu)
        val buffer = ByteBuffer.wrap(packet).order(ByteOrder.LITTLE_ENDIAN)

        assertEquals(PacketProtocol.IMU_HEADER_BYTES, packet.size)
        assertArrayEquals(PacketProtocol.MAGIC, packet.copyOfRange(0, 4))
        assertEquals(PacketProtocol.VERSION.toInt(), buffer.get(4).toInt())
        assertEquals(PacketProtocol.TYPE_IMU_SAMPLE.toInt(), buffer.get(5).toInt())
        assertEquals(PacketProtocol.IMU_HEADER_BYTES, buffer.getShort(6).toInt())
        assertEquals(99, buffer.getInt(12))
        assertEquals(1234L, buffer.getLong(16))
        assertEquals(7, buffer.getInt(24))
        assertEquals(1f, buffer.getFloat(28), 0.0f)
        assertEquals(7, buffer.getShort(64).toInt())
    }

    @Test
    fun videoPacketHasExpectedHeaderSizeAndOffsets() {
        val jpeg = ByteArray(200) { it.toByte() }
        val packet = PacketWriter.buildVideoFragment(
            sequence = 3,
            frameId = 42,
            fragmentIndex = 0,
            fragmentCount = 1,
            jpegBytes = jpeg,
            payloadOffset = 0,
            width = 640,
            height = 480,
            jpegQuality = 70,
            imuSample = imu,
        )
        val buffer = ByteBuffer.wrap(packet).order(ByteOrder.LITTLE_ENDIAN)

        assertEquals(PacketProtocol.VIDEO_HEADER_BYTES + jpeg.size, packet.size)
        assertArrayEquals(PacketProtocol.MAGIC, packet.copyOfRange(0, 4))
        assertEquals(PacketProtocol.TYPE_VIDEO_FRAGMENT.toInt(), buffer.get(5).toInt())
        assertEquals(PacketProtocol.VIDEO_HEADER_BYTES, buffer.getShort(6).toInt())
        assertEquals(42, buffer.getInt(24))
        assertEquals(0, buffer.getShort(28).toInt())
        assertEquals(1, buffer.getShort(30).toInt())
        assertEquals(200, buffer.getInt(32))
        assertEquals(640, buffer.getShort(36).toInt())
        assertEquals(480, buffer.getShort(38).toInt())
        assertEquals(70, buffer.get(40).toInt())
        assertEquals(7, buffer.getInt(44))
        assertEquals(1f, buffer.getFloat(48), 0.0f)
        assertEquals(jpeg.first(), packet[PacketProtocol.VIDEO_HEADER_BYTES])
    }

    @Test
    fun fragmentationUsesMtuBoundedPayloads() {
        val jpeg = ByteArray(PacketProtocol.VIDEO_PAYLOAD_BYTES * 2 + 9) { 1 }
        val count = PacketWriter.videoFragmentCount(jpeg.size)

        assertEquals(3, count)

        var offset = 0
        for (index in 0 until count) {
            val packet = PacketWriter.buildVideoFragment(
                sequence = index,
                frameId = 1,
                fragmentIndex = index,
                fragmentCount = count,
                jpegBytes = jpeg,
                payloadOffset = offset,
                width = 640,
                height = 480,
                jpegQuality = 70,
                imuSample = imu,
            )
            val expectedPayload = if (index < 2) PacketProtocol.VIDEO_PAYLOAD_BYTES else 9
            assertEquals(PacketProtocol.VIDEO_HEADER_BYTES + expectedPayload, packet.size)
            assert(packet.size <= PacketProtocol.MAX_DATAGRAM_BYTES)
            offset += PacketProtocol.VIDEO_PAYLOAD_BYTES
        }
    }
}
