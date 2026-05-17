package com.cyantus.aimbot.detection

import android.graphics.RectF

data class DetectionResult(
    val trackingId: Int?,
    val boundingBox: RectF,
    val label: String,
    val confidence: Float
)
