package com.cyantus.aimbot.eye.network

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.asCoroutineDispatcher
import java.io.Closeable
import java.net.InetSocketAddress
import java.nio.ByteBuffer
import java.nio.channels.DatagramChannel
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class StreamTarget(
    val host: String,
    val port: Int,
)

private data class EncodedFrame(
    val frameId: Int,
    val jpegBytes: ByteArray,
    val width: Int,
    val height: Int,
    val jpegQuality: Int,
    val imuSample: ImuSample,
)

class UdpStreamer(
    private val scope: CoroutineScope,
    private val onStatus: (String) -> Unit = {},
) : Closeable {
    private val frameChannel = Channel<EncodedFrame>(
        capacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    private val imuChannel = Channel<ImuSample>(
        capacity = 64,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    private val running = AtomicBoolean(false)
    private var dispatcher: ExecutorCoroutineDispatcher? = null
    private var senderJob: Job? = null
    private var channel: DatagramChannel? = null
    private var sequence = 0
    private var frameId = 0

    fun start(target: StreamTarget) {
        if (running.getAndSet(true)) {
            return
        }
        dispatcher = Executors
            .newSingleThreadExecutor { runnable -> Thread(runnable, "udp-streamer") }
            .asCoroutineDispatcher()
        senderJob = scope.launch(dispatcher!!) {
            runSender(target)
        }
    }

    fun stop() {
        if (!running.getAndSet(false)) {
            return
        }
        senderJob?.cancel()
        senderJob = null
        channel?.close()
        channel = null
        dispatcher?.close()
        dispatcher = null
        onStatus("Stopped")
    }

    fun submitJpeg(jpegBytes: ByteArray, width: Int, height: Int, jpegQuality: Int, imuSample: ImuSample) {
        if (!running.get()) {
            return
        }
        frameId += 1
        frameChannel.trySend(
            EncodedFrame(
                frameId = frameId,
                jpegBytes = jpegBytes,
                width = width,
                height = height,
                jpegQuality = jpegQuality,
                imuSample = imuSample,
            ),
        )
    }

    fun submitImu(sample: ImuSample) {
        if (!running.get()) {
            return
        }
        imuChannel.trySend(sample)
    }

    private suspend fun runSender(target: StreamTarget) {
        try {
            channel = DatagramChannel.open().apply {
                configureBlocking(false)
                connect(InetSocketAddress(target.host, target.port))
            }
            onStatus("Streaming to ${target.host}:${target.port}")
            while (running.get() && scope.isActive) {
                var sentAny = false

                while (true) {
                    val sample = imuChannel.tryReceive().getOrNull() ?: break
                    sendPacket(PacketWriter.buildImuPacket(nextSequence(), sample))
                    sentAny = true
                }

                val frame = frameChannel.tryReceive().getOrNull()
                if (frame != null) {
                    sendFrame(frame)
                    sentAny = true
                }

                if (!sentAny) {
                    delay(1)
                }
            }
        } catch (_: CancellationException) {
            // Normal stop path.
        } catch (exc: Exception) {
            onStatus("UDP error: ${exc.message ?: exc.javaClass.simpleName}")
        } finally {
            channel?.close()
            channel = null
        }
    }

    private fun sendFrame(frame: EncodedFrame) {
        val fragmentCount = PacketWriter.videoFragmentCount(frame.jpegBytes.size)
        var offset = 0
        for (fragmentIndex in 0 until fragmentCount) {
            val packet = PacketWriter.buildVideoFragment(
                sequence = nextSequence(),
                frameId = frame.frameId,
                fragmentIndex = fragmentIndex,
                fragmentCount = fragmentCount,
                jpegBytes = frame.jpegBytes,
                payloadOffset = offset,
                width = frame.width,
                height = frame.height,
                jpegQuality = frame.jpegQuality,
                imuSample = frame.imuSample,
            )
            sendPacket(packet)
            offset += PacketProtocol.VIDEO_PAYLOAD_BYTES
        }
    }

    private fun sendPacket(bytes: ByteArray) {
        val currentChannel = channel ?: return
        currentChannel.write(ByteBuffer.wrap(bytes))
    }

    private fun nextSequence(): Int {
        sequence += 1
        return sequence
    }

    override fun close() {
        stop()
        frameChannel.close()
        imuChannel.close()
    }
}
