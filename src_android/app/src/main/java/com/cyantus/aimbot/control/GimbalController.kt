package com.cyantus.aimbot.control

import kotlin.math.abs
import kotlin.math.roundToInt

data class GimbalControllerConfig(
    val kp: Float = 4.0f,
    val maxAbsSpeed: Int = 4000,
    val deadbandPx: Float = 12f,
    val panSign: Int = 1,
    val tiltSign: Int = 1
)

class GimbalController(
    private val config: GimbalControllerConfig = GimbalControllerConfig()
) {
    fun buildVelocityCommand(
        targetX: Float,
        targetY: Float,
        currentX: Float,
        currentY: Float
    ): String {
        val panSpeed = calculateSpeed(currentX - targetX, config.panSign)
        val tiltSpeed = calculateSpeed(currentY - targetY, config.tiltSign)
        return formatVelocityCommand(panSpeed, tiltSpeed)
    }

    fun stopCommand(): String = STOP_COMMAND

    private fun calculateSpeed(deltaPx: Float, axisSign: Int): Int {
        if (abs(deltaPx) <= config.deadbandPx) {
            return 0
        }

        return (deltaPx * config.kp * axisSign)
            .roundToInt()
            .coerceIn(-config.maxAbsSpeed, config.maxAbsSpeed)
    }

    private fun formatVelocityCommand(panSpeed: Int, tiltSpeed: Int): String =
        "V:$panSpeed,$tiltSpeed\n"

    companion object {
        const val STOP_COMMAND = "V:0,0\n"
    }
}
