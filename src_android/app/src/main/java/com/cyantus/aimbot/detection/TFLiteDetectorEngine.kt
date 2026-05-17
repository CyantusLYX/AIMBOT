package com.cyantus.aimbot.detection

import android.content.Context
import android.content.res.AssetManager
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import android.util.Size
import androidx.camera.core.ImageProxy
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class TFLiteDetectorEngine(
    context: Context,
    modelAssetPath: String,
    private val acceleration: Acceleration = Acceleration.GPU,
    private val defaultInputWidth: Int = DEFAULT_INPUT_WIDTH,
    private val defaultInputHeight: Int = DEFAULT_INPUT_HEIGHT,
    private val confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
    private val nmsIouThreshold: Float = DEFAULT_NMS_IOU_THRESHOLD,
    private val byteTracker: ByteTracker = ByteTracker(),
    private val labels: List<String> = emptyList(),
    private val configuredNumClasses: Int? = null,
    private val targetInferenceFps: Int = DEFAULT_TARGET_INFERENCE_FPS
) : ObjectDetectorEngine {
    enum class Acceleration {
        CPU,
        NNAPI,
        GPU
    }

    private val modelBuffer: MappedByteBuffer
    private var runtime: DetectorRuntime? = null
    private var closed = false
    private var analyzedFrameCount = 0
    private var lastMaxConfidence = 0f
    private var lastInferenceCompletedAtMs = 0L
    private var lastTrackedDetections: List<DetectionResult> = emptyList()

    init {
        require(modelAssetPath.isNotBlank()) {
            "modelAssetPath must not be blank."
        }
        require(defaultInputWidth > 0 && defaultInputHeight > 0) {
            "Default input dimensions must be positive."
        }
        require(confidenceThreshold in 0f..1f) {
            "confidenceThreshold must be between 0 and 1."
        }
        require(nmsIouThreshold in 0f..1f) {
            "nmsIouThreshold must be between 0 and 1."
        }
        require(configuredNumClasses == null || configuredNumClasses > 0) {
            "configuredNumClasses must be positive when provided."
        }
        require(targetInferenceFps > 0) {
            "targetInferenceFps must be positive."
        }

        modelBuffer = loadMappedModel(
            assetManager = context.applicationContext.assets,
            assetPath = modelAssetPath
        )
    }

    override fun analyze(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit) {
        if (closed) {
            imageProxy.close()
            onResult(emptyList())
            return
        }

        val nowMs = SystemClock.elapsedRealtime()
        if (lastInferenceCompletedAtMs > 0L &&
            nowMs - lastInferenceCompletedAtMs < minInferenceIntervalMs
        ) {
            imageProxy.close()
            onResult(lastTrackedDetections)
            return
        }

        var ranRealInference = false
        var resultToEmit: List<DetectionResult> = emptyList()
        try {
            ranRealInference = true
            val startedAtMs = SystemClock.elapsedRealtime()
            val runtime = runtime ?: createRuntime().also { createdRuntime ->
                runtime = createdRuntime
            }
            val runtimeReadyAtMs = SystemClock.elapsedRealtime()
            val preprocessedFrame = runtime.preprocessor.preprocess(imageProxy)
            val preprocessedAtMs = SystemClock.elapsedRealtime()
            val outputBuffers = runInference(runtime, preprocessedFrame.inputBuffer)
            val inferenceCompletedAtMs = SystemClock.elapsedRealtime()
            val rawDetections = decodeYoloDetections(
                runtime = runtime,
                outputBuffers = outputBuffers,
                preprocessedFrame = preprocessedFrame
            )
            val nmsDetections = DetectionPostProcessor.nonMaxSuppression(
                detections = rawDetections,
                confidenceThreshold = confidenceThreshold,
                iouThreshold = nmsIouThreshold
            )
            val trackedDetections = byteTracker.track(nmsDetections)
            lastTrackedDetections = trackedDetections
            resultToEmit = trackedDetections

            logDetectionStats(
                rawCount = rawDetections.size,
                nmsCount = nmsDetections.size,
                trackedCount = trackedDetections.size,
                runtimeMs = runtimeReadyAtMs - startedAtMs,
                preprocessMs = preprocessedAtMs - runtimeReadyAtMs,
                inferenceMs = inferenceCompletedAtMs - preprocessedAtMs,
                postprocessMs = SystemClock.elapsedRealtime() - inferenceCompletedAtMs
            )
        } catch (exception: RuntimeException) {
            Log.e(TAG, "TFLite analysis failed.", exception)
        } finally {
            if (ranRealInference) {
                lastInferenceCompletedAtMs = SystemClock.elapsedRealtime()
            }
            imageProxy.close()
        }
        onResult(resultToEmit)
    }

    override fun close() {
        closed = true
        runtime?.close()
        runtime = null
    }

    private fun createRuntime(): DetectorRuntime {
        val gpuDelegate = GpuDelegateFactory.createIfSupported(acceleration)
        val interpreterOptions = Interpreter.Options().apply {
            when (acceleration) {
                Acceleration.CPU -> setNumThreads(CPU_THREAD_COUNT)
                Acceleration.NNAPI -> setUseNNAPI(true)
                Acceleration.GPU -> {
                    if (gpuDelegate != null) {
                        addDelegate(gpuDelegate)
                    } else {
                        setNumThreads(CPU_THREAD_COUNT)
                    }
                }
            }
        }
        val interpreter = Interpreter(modelBuffer, interpreterOptions)
        interpreter.allocateTensors()

        val inputTensor = interpreter.getInputTensor(0)
        val inputSize = resolveInputSize(inputTensor)
        val preprocessor = YoloV7ImagePreprocessor(
            inputWidth = inputSize.width,
            inputHeight = inputSize.height,
            modelInputDataType = inputTensor.dataType(),
            inputQuantization = QuantizationParams.from(inputTensor.quantizationParams())
        )
        val outputBuffers = List(interpreter.outputTensorCount) { outputIndex ->
            val outputTensor = interpreter.getOutputTensor(outputIndex)
            TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())
        }
        val outputTensor = interpreter.getOutputTensor(0)
        val outputConfig = resolveYoloV7OutputConfig(outputTensor)

        return DetectorRuntime(
            interpreter = interpreter,
            gpuDelegate = gpuDelegate,
            preprocessor = preprocessor,
            outputBuffers = outputBuffers,
            outputConfig = outputConfig,
            outputValues = FloatArray(outputTensor.numElements())
        )
    }

    private fun runInference(
        runtime: DetectorRuntime,
        inputBuffer: ByteBuffer
    ): List<TensorBuffer> {
        inputBuffer.rewind()
        runtime.outputBuffers.forEach { outputBuffer ->
            outputBuffer.buffer.rewind()
        }

        if (runtime.outputBuffers.size == 1) {
            runtime.interpreter.run(inputBuffer, runtime.outputBuffers.single().buffer)
            return runtime.outputBuffers
        }

        val outputMap = mutableMapOf<Int, Any>()
        runtime.outputBuffers.forEachIndexed { index, outputBuffer ->
            outputMap[index] = outputBuffer.buffer
        }
        runtime.interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
        return runtime.outputBuffers
    }

    private fun decodeYoloDetections(
        runtime: DetectorRuntime,
        outputBuffers: List<TensorBuffer>,
        preprocessedFrame: PreprocessedFrame
    ): List<RawDetection> {
        val outputBuffer = outputBuffers.firstOrNull() ?: return emptyList()
        fillOutputValues(
            outputBuffer = outputBuffer,
            outputValues = runtime.outputValues,
            outputConfig = runtime.outputConfig
        )

        val detections = ArrayList<RawDetection>()
        val outputValues = runtime.outputValues
        val outputConfig = runtime.outputConfig
        var proposalOffset = 0
        var maxConfidence = 0f

        repeat(outputConfig.proposalCount) {
            val cx = outputValues[proposalOffset]
            val cy = outputValues[proposalOffset + 1]
            val width = outputValues[proposalOffset + 2]
            val height = outputValues[proposalOffset + 3]
            val objectness = outputValues[proposalOffset + 4]

            var bestClassId = 0
            var bestClassScore = outputValues[proposalOffset + YOLO_BOX_ATTRIBUTE_COUNT]
            var classIndex = 1
            while (classIndex < outputConfig.numClasses) {
                val classScore = outputValues[proposalOffset + YOLO_BOX_ATTRIBUTE_COUNT + classIndex]
                if (classScore > bestClassScore) {
                    bestClassScore = classScore
                    bestClassId = classIndex
                }
                classIndex += 1
            }

            val confidence = objectness * bestClassScore
            if (confidence > maxConfidence) {
                maxConfidence = confidence
            }
            if (confidence > confidenceThreshold) {
                val halfWidth = width / 2f
                val halfHeight = height / 2f
                val modelBox = RectF(
                    cx - halfWidth,
                    cy - halfHeight,
                    cx + halfWidth,
                    cy + halfHeight
                )
                val imageBox = preprocessedFrame.mapModelRectToImage(modelBox)
                if (imageBox.width() > 0f && imageBox.height() > 0f) {
                    detections += RawDetection(
                        boundingBox = imageBox,
                        label = labelFor(bestClassId),
                        confidence = confidence,
                        classId = bestClassId
                    )
                }
            }

            proposalOffset += outputConfig.valuesPerProposal
        }

        lastMaxConfidence = maxConfidence
        return detections
    }

    private fun logDetectionStats(
        rawCount: Int,
        nmsCount: Int,
        trackedCount: Int,
        runtimeMs: Long,
        preprocessMs: Long,
        inferenceMs: Long,
        postprocessMs: Long
    ) {
        analyzedFrameCount += 1
        if (analyzedFrameCount % LOG_STATS_INTERVAL_FRAMES != 0) {
            return
        }

        Log.d(
            TAG,
            "YOLOv7 stats: maxConfidence=$lastMaxConfidence raw=$rawCount nms=$nmsCount " +
                "tracked=$trackedCount runtimeMs=$runtimeMs preprocessMs=$preprocessMs " +
                "inferenceMs=$inferenceMs postprocessMs=$postprocessMs"
        )
    }

    private val minInferenceIntervalMs: Long
        get() = MS_PER_SECOND / targetInferenceFps

    private fun fillOutputValues(
        outputBuffer: TensorBuffer,
        outputValues: FloatArray,
        outputConfig: YoloV7OutputConfig
    ) {
        val sourceBuffer = outputBuffer.buffer.duplicate().order(ByteOrder.nativeOrder())
        sourceBuffer.rewind()

        when (outputConfig.dataType) {
            DataType.FLOAT32 -> {
                outputValues.indices.forEach { index ->
                    outputValues[index] = sourceBuffer.float
                }
            }
            DataType.UINT8 -> {
                outputValues.indices.forEach { index ->
                    val quantizedValue = sourceBuffer.get().toInt() and UINT8_MASK
                    outputValues[index] = outputConfig.quantization.dequantize(quantizedValue)
                }
            }
            DataType.INT8 -> {
                outputValues.indices.forEach { index ->
                    outputValues[index] = outputConfig.quantization.dequantize(sourceBuffer.get().toInt())
                }
            }
            DataType.INT32 -> {
                outputValues.indices.forEach { index ->
                    outputValues[index] = sourceBuffer.int.toFloat()
                }
            }
            else -> {
                outputBuffer.floatArray.copyInto(outputValues)
            }
        }
    }

    private fun labelFor(classId: Int): String {
        return labels.getOrNull(classId) ?: if (configuredNumClasses == 1 || labels.size == 1) {
            DEFAULT_LABEL
        } else {
            "Class $classId"
        }
    }

    private fun resolveYoloV7OutputConfig(outputTensor: Tensor): YoloV7OutputConfig {
        val shape = outputTensor.shape()
        val inferredNumClasses = when {
            shape.isNotEmpty() && shape.last() > YOLO_BOX_ATTRIBUTE_COUNT -> {
                shape.last() - YOLO_BOX_ATTRIBUTE_COUNT
            }
            else -> DEFAULT_NUM_CLASSES
        }
        val resolvedNumClasses = configuredNumClasses ?: inferredNumClasses
        val valuesPerProposal = YOLO_BOX_ATTRIBUTE_COUNT + resolvedNumClasses
        val flatSize = outputTensor.numElements()
        val proposalCount = when {
            shape.size >= EXPECTED_YOLO_OUTPUT_RANK && shape.last() >= valuesPerProposal -> {
                shape[shape.size - 2]
            }
            shape.size == 2 && shape[1] >= valuesPerProposal -> {
                shape[0]
            }
            else -> flatSize / valuesPerProposal
        }

        require(proposalCount > 0 && flatSize >= proposalCount * valuesPerProposal) {
            "Unsupported YOLOv7 output tensor shape: ${shape.contentToString()}."
        }

        return YoloV7OutputConfig(
            proposalCount = proposalCount,
            valuesPerProposal = valuesPerProposal,
            numClasses = resolvedNumClasses,
            dataType = outputTensor.dataType(),
            quantization = QuantizationParams.from(outputTensor.quantizationParams())
        )
    }

    private fun resolveInputSize(inputTensor: Tensor): Size {
        val shape = inputTensor.shape()
        if (shape.size == EXPECTED_NHWC_RANK && shape[1] > 0 && shape[2] > 0) {
            return Size(shape[2], shape[1])
        }

        return Size(defaultInputWidth, defaultInputHeight)
    }

    private fun loadMappedModel(
        assetManager: AssetManager,
        assetPath: String
    ): MappedByteBuffer {
        return assetManager.openFd(assetPath).use { fileDescriptor ->
            FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
                inputStream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength
                )
            }
        }
    }

    private data class DetectorRuntime(
        val interpreter: Interpreter,
        val gpuDelegate: GpuDelegate?,
        val preprocessor: YoloV7ImagePreprocessor,
        val outputBuffers: List<TensorBuffer>,
        val outputConfig: YoloV7OutputConfig,
        val outputValues: FloatArray
    ) {
        fun close() {
            interpreter.close()
            gpuDelegate?.close()
        }
    }

    private object GpuDelegateFactory {
        fun createIfSupported(acceleration: Acceleration): GpuDelegate? {
            if (acceleration != Acceleration.GPU) {
                return null
            }

            val compatibilityList = CompatibilityList()
            return if (compatibilityList.isDelegateSupportedOnThisDevice) {
                GpuDelegate(compatibilityList.bestOptionsForThisDevice)
            } else {
                null
            }
        }
    }

    private companion object {
        const val TAG = "TFLiteDetectorEngine"
        const val CPU_THREAD_COUNT = 2
        const val DEFAULT_INPUT_WIDTH = 640
        const val DEFAULT_INPUT_HEIGHT = 640
        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.25f
        const val DEFAULT_NMS_IOU_THRESHOLD = 0.45f
        const val DEFAULT_TARGET_INFERENCE_FPS = 5
        const val EXPECTED_NHWC_RANK = 4
        const val EXPECTED_YOLO_OUTPUT_RANK = 3
        const val YOLO_BOX_ATTRIBUTE_COUNT = 5
        const val DEFAULT_NUM_CLASSES = 1
        const val MODEL_CHANNEL_COUNT = 3
        const val DEFAULT_LABEL = "Object"
        const val UINT8_MASK = 0xFF
        const val LOG_STATS_INTERVAL_FRAMES = 5
        const val MS_PER_SECOND = 1_000L
    }
}

private class YoloV7ImagePreprocessor(
    private val inputWidth: Int,
    private val inputHeight: Int,
    private val modelInputDataType: DataType,
    private val inputQuantization: QuantizationParams
) {
    private val tensorImage = TensorImage(DataType.FLOAT32)
    private val quantizedInputBuffer = when (modelInputDataType) {
        DataType.INT8,
        DataType.UINT8 -> ByteBuffer.allocateDirect(inputWidth * inputHeight * MODEL_CHANNEL_COUNT)
            .order(ByteOrder.nativeOrder())
        else -> null
    }
    private var cachedRotationDegrees: Int? = null
    private var cachedImageProcessor: ImageProcessor? = null

    fun preprocess(imageProxy: ImageProxy): PreprocessedFrame {
        val bitmap = imageProxy.toBitmap()
        tensorImage.load(bitmap)

        val imageProcessor = processorFor(imageProxy.imageInfo.rotationDegrees)
        val processedImage = imageProcessor.process(tensorImage)
        val normalizedInputBuffer = processedImage.buffer.apply {
            rewind()
        }
        val modelInputBuffer = when (modelInputDataType) {
            DataType.FLOAT32 -> normalizedInputBuffer
            DataType.INT8,
            DataType.UINT8 -> quantizeNormalizedInput(normalizedInputBuffer)
            else -> normalizedInputBuffer
        }

        return PreprocessedFrame(
            inputBuffer = modelInputBuffer,
            imageProcessor = imageProcessor,
            imageWidth = imageProxy.width,
            imageHeight = imageProxy.height
        )
    }

    private fun processorFor(rotationDegrees: Int): ImageProcessor {
        val normalizedRotation = ((rotationDegrees % FULL_ROTATION_DEGREES) + FULL_ROTATION_DEGREES) %
            FULL_ROTATION_DEGREES
        val cachedProcessor = cachedImageProcessor
        if (cachedProcessor != null && cachedRotationDegrees == normalizedRotation) {
            return cachedProcessor
        }

        val clockwiseTurns = normalizedRotation / RIGHT_ANGLE_DEGREES
        val counterClockwiseTurns = (FULL_TURN_COUNT - clockwiseTurns) % FULL_TURN_COUNT
        val builder = ImageProcessor.Builder()
        if (counterClockwiseTurns != 0) {
            builder.add(Rot90Op(counterClockwiseTurns))
        }
        builder.add(
            ResizeOp(
                inputHeight,
                inputWidth,
                ResizeOp.ResizeMethod.BILINEAR
            )
        )
        builder.add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD))

        return builder.build().also { imageProcessor ->
            cachedRotationDegrees = normalizedRotation
            cachedImageProcessor = imageProcessor
        }
    }

    private fun quantizeNormalizedInput(normalizedInputBuffer: ByteBuffer): ByteBuffer {
        val outputBuffer = requireNotNull(quantizedInputBuffer) {
            "Quantized input buffer was not initialized."
        }
        val sourceBuffer = normalizedInputBuffer.duplicate().order(ByteOrder.nativeOrder())
        sourceBuffer.rewind()
        outputBuffer.rewind()

        val quantizedRange = when (modelInputDataType) {
            DataType.INT8 -> INT8_MIN..INT8_MAX
            DataType.UINT8 -> UINT8_MIN..UINT8_MAX
            else -> error("Unsupported quantized input type: $modelInputDataType.")
        }

        repeat(inputWidth * inputHeight * MODEL_CHANNEL_COUNT) {
            val normalizedValue = sourceBuffer.float
            val quantizedValue = inputQuantization.quantize(normalizedValue)
                .coerceIn(quantizedRange.first, quantizedRange.last)
            outputBuffer.put(quantizedValue.toByte())
        }

        return outputBuffer.apply {
            rewind()
        }
    }

    private companion object {
        const val MODEL_CHANNEL_COUNT = 3
        const val NORMALIZE_MEAN = 0f
        const val NORMALIZE_STD = 255f
        const val FULL_ROTATION_DEGREES = 360
        const val RIGHT_ANGLE_DEGREES = 90
        const val FULL_TURN_COUNT = 4
        const val INT8_MIN = -128
        const val INT8_MAX = 127
        const val UINT8_MIN = 0
        const val UINT8_MAX = 255
    }
}

private data class PreprocessedFrame(
    val inputBuffer: ByteBuffer,
    private val imageProcessor: ImageProcessor,
    private val imageWidth: Int,
    private val imageHeight: Int
) {
    fun mapModelRectToImage(modelRect: RectF): RectF {
        val imageRect = imageProcessor.inverseTransform(modelRect, imageHeight, imageWidth)
        imageRect.sort()
        imageRect.left = imageRect.left.coerceIn(0f, imageWidth.toFloat())
        imageRect.top = imageRect.top.coerceIn(0f, imageHeight.toFloat())
        imageRect.right = imageRect.right.coerceIn(0f, imageWidth.toFloat())
        imageRect.bottom = imageRect.bottom.coerceIn(0f, imageHeight.toFloat())
        imageRect.sort()
        return imageRect
    }
}

private data class YoloV7OutputConfig(
    val proposalCount: Int,
    val valuesPerProposal: Int,
    val numClasses: Int,
    val dataType: DataType,
    val quantization: QuantizationParams
)

private data class QuantizationParams(
    val scale: Float,
    val zeroPoint: Int
) {
    fun quantize(value: Float): Int {
        return (value / safeScale + zeroPoint).roundToInt()
    }

    fun dequantize(value: Int): Float {
        return (value - zeroPoint) * safeScale
    }

    private val safeScale: Float
        get() = if (scale > 0f) {
            scale
        } else {
            DEFAULT_SCALE
        }

    companion object {
        fun from(params: Tensor.QuantizationParams): QuantizationParams {
            return QuantizationParams(
                scale = params.scale,
                zeroPoint = params.zeroPoint
            )
        }

        private const val DEFAULT_SCALE = 1f / 255f
    }
}

internal object DetectionPostProcessor {
    fun nonMaxSuppression(
        detections: List<RawDetection>,
        confidenceThreshold: Float,
        iouThreshold: Float,
        maxCandidates: Int = MAX_NMS_CANDIDATES,
        maxDetections: Int = MAX_DETECTIONS
    ): List<RawDetection> {
        val candidates = detections
            .filter { detection -> detection.confidence >= confidenceThreshold }
            .sortedByDescending { detection -> detection.confidence }
            .take(maxCandidates)
        val selectedDetections = mutableListOf<RawDetection>()

        for (candidate in candidates) {
            if (selectedDetections.size >= maxDetections) {
                break
            }
            val overlapsSelectedDetection = selectedDetections.any { selected ->
                selected.classId == candidate.classId &&
                    intersectionOverUnion(selected.boundingBox, candidate.boundingBox) > iouThreshold
            }
            if (!overlapsSelectedDetection) {
                selectedDetections += candidate
            }
        }

        return selectedDetections
    }

    private fun intersectionOverUnion(first: RectF, second: RectF): Float {
        val intersectionLeft = maxOf(first.left, second.left)
        val intersectionTop = maxOf(first.top, second.top)
        val intersectionRight = minOf(first.right, second.right)
        val intersectionBottom = minOf(first.bottom, second.bottom)
        val intersectionWidth = maxOf(0f, intersectionRight - intersectionLeft)
        val intersectionHeight = maxOf(0f, intersectionBottom - intersectionTop)
        val intersectionArea = intersectionWidth * intersectionHeight

        val firstArea = maxOf(0f, first.right - first.left) *
            maxOf(0f, first.bottom - first.top)
        val secondArea = maxOf(0f, second.right - second.left) *
            maxOf(0f, second.bottom - second.top)
        val unionArea = firstArea + secondArea - intersectionArea

        return if (unionArea <= 0f) {
            0f
        } else {
            intersectionArea / unionArea
        }
    }

    private const val MAX_NMS_CANDIDATES = 300
    private const val MAX_DETECTIONS = 50
}
