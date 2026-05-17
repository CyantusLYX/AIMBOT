package com.cyantus.aimbot.detection

import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op

class ImageProxyPreprocessor(
    private val inputWidth: Int,
    private val inputHeight: Int,
    private val inputDataType: DataType,
    private val normalizeMean: Float = DEFAULT_NORMALIZE_MEAN,
    private val normalizeStd: Float = DEFAULT_NORMALIZE_STD
) {
    private val tensorImage = TensorImage(inputDataType)
    private var cachedRotationDegrees: Int? = null
    private var cachedImageProcessor: ImageProcessor? = null

    fun preprocess(imageProxy: ImageProxy): ByteBuffer {
        val bitmap = imageProxy.toBitmap()
        tensorImage.load(bitmap)

        val processedImage = processorFor(imageProxy.imageInfo.rotationDegrees)
            .process(tensorImage)
        return processedImage.buffer.apply {
            rewind()
        }
    }

    private fun processorFor(rotationDegrees: Int): ImageProcessor {
        val normalizedRotation = ((rotationDegrees % FULL_ROTATION_DEGREES) + FULL_ROTATION_DEGREES) %
            FULL_ROTATION_DEGREES
        val cachedProcessor = cachedImageProcessor
        if (cachedProcessor != null && cachedRotationDegrees == normalizedRotation) {
            return cachedProcessor
        }

        val clockwiseTurns = normalizedRotation / RIGHT_ANGLE_DEGREES
        val rot90Turns = (FULL_TURN_COUNT - clockwiseTurns) % FULL_TURN_COUNT
        val builder = ImageProcessor.Builder()
        if (rot90Turns != 0) {
            builder.add(Rot90Op(rot90Turns))
        }
        builder.add(
            ResizeOp(
                inputHeight,
                inputWidth,
                ResizeOp.ResizeMethod.BILINEAR
            )
        )
        if (inputDataType == DataType.FLOAT32) {
            builder.add(NormalizeOp(normalizeMean, normalizeStd))
        }

        return builder.build().also { imageProcessor ->
            cachedRotationDegrees = normalizedRotation
            cachedImageProcessor = imageProcessor
        }
    }

    private companion object {
        const val DEFAULT_NORMALIZE_MEAN = 0f
        const val DEFAULT_NORMALIZE_STD = 255f
        const val FULL_ROTATION_DEGREES = 360
        const val RIGHT_ANGLE_DEGREES = 90
        const val FULL_TURN_COUNT = 4
    }
}
