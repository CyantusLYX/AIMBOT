package com.cyantus.aimbot.eye

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.PowerManager
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.cyantus.aimbot.eye.network.StreamTarget
import com.cyantus.aimbot.eye.network.UdpStreamer
import com.cyantus.aimbot.eye.sensors.ImuSampler
import kotlinx.coroutines.CoroutineScope
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

data class SensorNodeConfig(
    val host: String,
    val port: Int,
    val jpegQuality: Int,
    val targetFps: Int,
)

private data class EncodedJpegFrame(
    val bytes: ByteArray,
    val width: Int,
    val height: Int,
)

class SensorNodeController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    scope: CoroutineScope,
    private val onStatus: (String) -> Unit,
) : AutoCloseable {
    private val streamer = UdpStreamer(scope, onStatus)
    private val imuSampler = ImuSampler(context) { sample -> streamer.submitImu(sample) }
    private var analyzerExecutor: ExecutorService? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var wakeLock: PowerManager.WakeLock? = null

    fun start(config: SensorNodeConfig) {
        val cleanHost = config.host.trim()
        require(cleanHost.isNotEmpty()) { "Target IP is required" }
        require(config.port in 1..65535) { "Port must be 1..65535" }

        stop()
        acquireWakeLock()
        streamer.start(StreamTarget(cleanHost, config.port))
        imuSampler.start()
        startCamera(config)
    }

    fun stop() {
        cameraProvider?.unbindAll()
        analyzerExecutor?.shutdownNow()
        analyzerExecutor = null
        imuSampler.stop()
        streamer.stop()
        releaseWakeLock()
    }

    private fun startCamera(config: SensorNodeConfig) {
        analyzerExecutor = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "camera-analysis")
        }

        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener(
            {
                try {
                    val provider = providerFuture.get()
                    cameraProvider = provider

                    val analysisBuilder = ImageAnalysis.Builder()
                        .setResolutionSelector(
                            ResolutionSelector.Builder()
                                .setResolutionStrategy(
                                    ResolutionStrategy(
                                        Size(640, 480),
                                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                    ),
                                )
                                .build(),
                        )
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)

                    Camera2Interop.Extender(analysisBuilder)
                        .setCaptureRequestOption(
                            android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                            Range(30, config.targetFps.coerceIn(30, 60)),
                        )

                    val analysis = analysisBuilder.build().also { useCase ->
                        useCase.setAnalyzer(analyzerExecutor!!) { image ->
                            analyzeImage(image, config.jpegQuality.coerceIn(1, 100))
                        }
                    }

                    provider.unbindAll()
                    provider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        analysis,
                    )
                    onStatus("Camera streaming ${config.host}:${config.port}")
                } catch (exc: Exception) {
                    onStatus("Camera error: ${exc.message ?: exc.javaClass.simpleName}")
                    stop()
                }
            },
            ContextCompat.getMainExecutor(context),
        )
    }

    private fun analyzeImage(image: ImageProxy, jpegQuality: Int) {
        try {
            val frame = imageToJpeg(image, jpegQuality)
            streamer.submitJpeg(
                jpegBytes = frame.bytes,
                width = frame.width,
                height = frame.height,
                jpegQuality = jpegQuality,
                imuSample = imuSampler.latestSample(),
            )
        } catch (exc: Exception) {
            onStatus("Frame encode error: ${exc.message ?: exc.javaClass.simpleName}")
        } finally {
            image.close()
        }
    }

    private fun imageToJpeg(image: ImageProxy, jpegQuality: Int): EncodedJpegFrame {
        val plane = image.planes[0]
        val source = plane.buffer.duplicate()
        val width = image.width
        val height = image.height
        val bytesPerPixel = plane.pixelStride
        val rowStride = plane.rowStride
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        if (rowStride == width * bytesPerPixel) {
            bitmap.copyPixelsFromBuffer(source)
        } else {
            val compact = ByteArray(width * height * bytesPerPixel)
            val row = ByteArray(rowStride)
            var dstOffset = 0
            for (y in 0 until height) {
                source.get(row, 0, rowStride)
                System.arraycopy(row, 0, compact, dstOffset, width * bytesPerPixel)
                dstOffset += width * bytesPerPixel
            }
            bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(compact))
        }

        val rotationDegrees = image.imageInfo.rotationDegrees
        val outputBitmap = if (rotationDegrees == 0) {
            bitmap
        } else {
            val matrix = Matrix().apply {
                postRotate(rotationDegrees.toFloat())
            }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true).also {
                bitmap.recycle()
            }
        }

        val output = ByteArrayOutputStream()
        outputBitmap.compress(Bitmap.CompressFormat.JPEG, jpegQuality, output)
        val encoded = EncodedJpegFrame(
            bytes = output.toByteArray(),
            width = outputBitmap.width,
            height = outputBitmap.height,
        )
        outputBitmap.recycle()
        return encoded
    }

    private fun acquireWakeLock() {
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "GimbalSensorNode:stream").apply {
            setReferenceCounted(false)
            acquire(10 * 60 * 1000L)
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
            }
        }
        wakeLock = null
    }

    override fun close() {
        stop()
        streamer.close()
    }
}
