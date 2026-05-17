package com.cyantus.aimbot.control

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GimbalControllerTest {
    @Test
    fun zeroErrorProducesStopCommand() {
        val controller = GimbalController()

        val command = controller.buildVelocityCommand(
            targetX = 320f,
            targetY = 240f,
            currentX = 320f,
            currentY = 240f
        )

        assertEquals("V:0,0\n", command)
    }

    @Test
    fun errorInsideDeadbandProducesZeroSpeed() {
        val controller = GimbalController()

        val command = controller.buildVelocityCommand(
            targetX = 320f,
            targetY = 240f,
            currentX = 331f,
            currentY = 252f
        )

        assertEquals("V:0,0\n", command)
    }

    @Test
    fun positiveErrorProducesPositiveSpeeds() {
        val controller = GimbalController()

        val command = controller.buildVelocityCommand(
            targetX = 320f,
            targetY = 240f,
            currentX = 420f,
            currentY = 340f
        )

        assertEquals("V:400,400\n", command)
    }

    @Test
    fun negativeErrorProducesNegativeSpeeds() {
        val controller = GimbalController()

        val command = controller.buildVelocityCommand(
            targetX = 320f,
            targetY = 240f,
            currentX = 220f,
            currentY = 140f
        )

        assertEquals("V:-400,-400\n", command)
    }

    @Test
    fun outputIsClampedToMaxSpeed() {
        val controller = GimbalController()

        val command = controller.buildVelocityCommand(
            targetX = 0f,
            targetY = 0f,
            currentX = 2_000f,
            currentY = -2_000f
        )

        assertEquals("V:4000,-4000\n", command)
    }

    @Test
    fun signsCanBeInvertedPerAxis() {
        val controller = GimbalController(
            config = GimbalControllerConfig(
                panSign = -1,
                tiltSign = -1
            )
        )

        val command = controller.buildVelocityCommand(
            targetX = 0f,
            targetY = 0f,
            currentX = 100f,
            currentY = -100f
        )

        assertEquals("V:-400,400\n", command)
    }

    @Test
    fun commandFormatIncludesTrailingNewline() {
        val controller = GimbalController()

        val command = controller.buildVelocityCommand(
            targetX = 0f,
            targetY = 0f,
            currentX = 25f,
            currentY = 25f
        )

        assertEquals("V:100,100\n", command)
        assertTrue(command.endsWith("\n"))
    }
}
