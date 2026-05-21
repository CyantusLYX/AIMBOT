package com.cyantus.aimbot.eye.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.cyantus.aimbot.eye.network.ImuSample
import java.util.concurrent.atomic.AtomicInteger

private const val SOURCE_GYRO: Short = 1
private const val SOURCE_ACCEL: Short = 2
private const val SOURCE_ORIENTATION: Short = 4

class ImuSampler(
    context: Context,
    private val samplePeriodUs: Int = 10_000,
    private val onSample: (ImuSample) -> Unit,
) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val sequence = AtomicInteger(0)
    private val lock = Any()

    private var latest = ImuSample(
        sequence = 0,
        deviceTimeNs = System.nanoTime(),
        roll = 0f,
        pitch = 0f,
        yaw = 0f,
        gyroX = 0f,
        gyroY = 0f,
        gyroZ = 0f,
        accelX = 0f,
        accelY = 0f,
        accelZ = 0f,
        sourceFlags = 0,
    )
    private var lastPublishNs = 0L

    fun start() {
        register(Sensor.TYPE_GYROSCOPE)
        register(Sensor.TYPE_ACCELEROMETER)
        val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        if (rotationSensor != null) {
            sensorManager.registerListener(this, rotationSensor, samplePeriodUs)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    fun latestSample(): ImuSample = synchronized(lock) { latest }

    override fun onSensorChanged(event: SensorEvent) {
        val nowNs = System.nanoTime()
        val updated = synchronized(lock) {
            var sample = latest
            var flags = sample.sourceFlags
            when (event.sensor.type) {
                Sensor.TYPE_GYROSCOPE -> {
                    flags = (flags.toInt() or SOURCE_GYRO.toInt()).toShort()
                    sample = sample.copy(
                        gyroX = event.values.getOrElse(0) { 0f },
                        gyroY = event.values.getOrElse(1) { 0f },
                        gyroZ = event.values.getOrElse(2) { 0f },
                        sourceFlags = flags,
                    )
                }
                Sensor.TYPE_ACCELEROMETER -> {
                    flags = (flags.toInt() or SOURCE_ACCEL.toInt()).toShort()
                    sample = sample.copy(
                        accelX = event.values.getOrElse(0) { 0f },
                        accelY = event.values.getOrElse(1) { 0f },
                        accelZ = event.values.getOrElse(2) { 0f },
                        sourceFlags = flags,
                    )
                }
                Sensor.TYPE_GAME_ROTATION_VECTOR,
                Sensor.TYPE_ROTATION_VECTOR -> {
                    val rotationMatrix = FloatArray(9)
                    val orientation = FloatArray(3)
                    SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                    SensorManager.getOrientation(rotationMatrix, orientation)
                    flags = (flags.toInt() or SOURCE_ORIENTATION.toInt()).toShort()
                    sample = sample.copy(
                        yaw = orientation[0],
                        pitch = orientation[1],
                        roll = orientation[2],
                        sourceFlags = flags,
                    )
                }
            }

            if (nowNs - lastPublishNs >= samplePeriodUs * 1_000L) {
                lastPublishNs = nowNs
                sample = sample.copy(
                    sequence = sequence.incrementAndGet(),
                    deviceTimeNs = nowNs,
                )
                latest = sample
                sample
            } else {
                latest = sample
                null
            }
        }
        if (updated != null) {
            onSample(updated)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun register(sensorType: Int) {
        val sensor = sensorManager.getDefaultSensor(sensorType) ?: return
        sensorManager.registerListener(this, sensor, samplePeriodUs)
    }
}
