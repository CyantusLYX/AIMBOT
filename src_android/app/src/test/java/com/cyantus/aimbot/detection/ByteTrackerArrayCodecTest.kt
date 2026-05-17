package com.cyantus.aimbot.detection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class ByteTrackerArrayCodecTest {
    @Test
    fun invalidInputStrideThrows() {
        val exception = assertThrows(IllegalArgumentException::class.java) {
            ByteTrackerArrayCodec.requireStride(
                values = floatArrayOf(1f, 2f),
                stride = ByteTrackerArrayCodec.INPUT_STRIDE,
                name = "rawDetections"
            )
        }

        assertTrue(exception.message!!.contains("multiple of 6"))
    }

    @Test
    fun parseTrackedDetectionAtUsesOutputStride() {
        val trackedDetection = ByteTrackerArrayCodec.parseTrackedDetectionAt(
            trackedDetections = floatArrayOf(
                42f,
                10f,
                20f,
                30f,
                40f,
                0.9f
            ),
            index = 0
        )

        assertEquals(42, trackedDetection.trackingId)
        assertEquals(10f, trackedDetection.x, FLOAT_TOLERANCE)
        assertEquals(20f, trackedDetection.y, FLOAT_TOLERANCE)
        assertEquals(30f, trackedDetection.width, FLOAT_TOLERANCE)
        assertEquals(40f, trackedDetection.height, FLOAT_TOLERANCE)
        assertEquals(0.9f, trackedDetection.confidence, FLOAT_TOLERANCE)
    }

    private companion object {
        const val FLOAT_TOLERANCE = 0.0001f
    }
}
