package com.cyantus.aimbot.detection

import androidx.camera.core.ImageProxy

interface ObjectDetectorEngine {
    fun analyze(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit)
    fun close()
}
