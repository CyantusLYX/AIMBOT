package com.cyantus.aimbot.detection

import android.graphics.RectF

data class RawDetection(
    val boundingBox: RectF,
    val label: String,
    val confidence: Float,
    val classId: Int
)

class ByteTracker {
    init {
        ByteTrackNativeLibrary.ensureLoaded()
    }

    fun track(detections: List<RawDetection>): List<DetectionResult> {
        if (detections.isEmpty()) {
            return emptyList()
        }

        val rawDetections = ByteTrackerArrayCodec.flattenDetections(detections)
        val trackedDetections = nativeUpdate(rawDetections)
        return ByteTrackerArrayCodec.parseTrackedDetections(
            trackedDetections = trackedDetections,
            labels = detections.map { detection -> detection.label }
        )
    }

    fun trackFlat(rawDetections: FloatArray): FloatArray {
        ByteTrackerArrayCodec.requireStride(
            values = rawDetections,
            stride = ByteTrackerArrayCodec.INPUT_STRIDE,
            name = "rawDetections"
        )
        if (rawDetections.isEmpty()) {
            return FloatArray(0)
        }

        return nativeUpdate(rawDetections)
    }

    private external fun nativeUpdate(rawDetections: FloatArray): FloatArray
}

internal object ByteTrackerArrayCodec {
    const val INPUT_STRIDE = 6
    const val OUTPUT_STRIDE = 6

    fun flattenDetections(detections: List<RawDetection>): FloatArray {
        val rawDetections = FloatArray(detections.size * INPUT_STRIDE)
        detections.forEachIndexed { index, detection ->
            val offset = index * INPUT_STRIDE
            val box = detection.boundingBox
            rawDetections[offset] = box.left
            rawDetections[offset + 1] = box.top
            rawDetections[offset + 2] = box.width()
            rawDetections[offset + 3] = box.height()
            rawDetections[offset + 4] = detection.confidence
            rawDetections[offset + 5] = detection.classId.toFloat()
        }

        return rawDetections
    }

    fun parseTrackedDetections(
        trackedDetections: FloatArray,
        labels: List<String>
    ): List<DetectionResult> {
        requireStride(
            values = trackedDetections,
            stride = OUTPUT_STRIDE,
            name = "trackedDetections"
        )

        return List(trackedDetections.size / OUTPUT_STRIDE) { index ->
            val trackedDetection = parseTrackedDetectionAt(trackedDetections, index)

            DetectionResult(
                trackingId = trackedDetection.trackingId,
                boundingBox = RectF(
                    trackedDetection.x,
                    trackedDetection.y,
                    trackedDetection.x + trackedDetection.width,
                    trackedDetection.y + trackedDetection.height
                ),
                label = labels.getOrNull(index) ?: DEFAULT_LABEL,
                confidence = trackedDetection.confidence
            )
        }
    }

    fun parseTrackedDetectionAt(
        trackedDetections: FloatArray,
        index: Int
    ): TrackedDetectionFlat {
        requireStride(
            values = trackedDetections,
            stride = OUTPUT_STRIDE,
            name = "trackedDetections"
        )
        require(index in 0 until trackedDetections.size / OUTPUT_STRIDE) {
            "index is outside trackedDetections."
        }

        val offset = index * OUTPUT_STRIDE
        return TrackedDetectionFlat(
            trackingId = trackedDetections[offset].toInt(),
            x = trackedDetections[offset + 1],
            y = trackedDetections[offset + 2],
            width = trackedDetections[offset + 3],
            height = trackedDetections[offset + 4],
            confidence = trackedDetections[offset + 5]
        )
    }

    fun requireStride(values: FloatArray, stride: Int, name: String) {
        require(values.size % stride == 0) {
            "$name must be a multiple of $stride floats."
        }
    }

    data class TrackedDetectionFlat(
        val trackingId: Int,
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float,
        val confidence: Float
    )

    private const val DEFAULT_LABEL = "Object"
}

private object ByteTrackNativeLibrary {
    init {
        System.loadLibrary("bytetrack_jni")
    }

    fun ensureLoaded() = Unit
}
