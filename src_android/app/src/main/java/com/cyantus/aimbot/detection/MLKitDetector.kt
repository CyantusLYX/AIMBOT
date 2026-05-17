package com.cyantus.aimbot.detection

import android.graphics.RectF
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions

class MLKitDetector : ObjectDetectorEngine {
    private val detector = ObjectDetection.getClient(
        ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
            .enableMultipleObjects()
            .enableClassification()
            .build()
    )

    @androidx.annotation.OptIn(markerClass = [ExperimentalGetImage::class])
    override fun analyze(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit) {
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            onResult(emptyList())
            imageProxy.close()
            return
        }

        val image = InputImage.fromMediaImage(
            mediaImage,
            imageProxy.imageInfo.rotationDegrees
        )

        detector.process(image)
            .addOnSuccessListener { detectedObjects ->
                val results = detectedObjects.map { detectedObject ->
                    val bestLabel = detectedObject.labels.maxByOrNull { it.confidence }
                    DetectionResult(
                        trackingId = detectedObject.trackingId,
                        boundingBox = RectF(detectedObject.boundingBox),
                        label = bestLabel?.text ?: "Object",
                        confidence = bestLabel?.confidence ?: 0f
                    )
                }
                onResult(results)
            }
            .addOnFailureListener {
                onResult(emptyList())
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    override fun close() {
        detector.close()
    }
}
