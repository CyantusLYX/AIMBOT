package com.cyantus.aimbot

import android.content.Context
import android.graphics.RectF
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.camera.view.TransformExperimental
import androidx.camera.view.transform.CoordinateTransform
import androidx.camera.view.transform.ImageProxyTransformFactory
import androidx.camera.view.transform.OutputTransform
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.cyantus.aimbot.bluetooth.AndroidBluetoothManager
import com.cyantus.aimbot.bluetooth.BluetoothConnectionState
import com.cyantus.aimbot.bluetooth.BluetoothDeviceInfo
import com.cyantus.aimbot.bluetooth.BluetoothManager
import com.cyantus.aimbot.control.GimbalController
import com.cyantus.aimbot.detection.DetectionResult
import com.cyantus.aimbot.detection.MLKitDetector
import com.cyantus.aimbot.detection.ObjectDetectorEngine
import com.cyantus.aimbot.detection.TFLiteDetectorEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

enum class DetectorPipeline(val displayName: String) {
    ML_KIT("ML Kit"),
    CUSTOM_TFLITE("Custom YOLOv7")
}

class GimbalViewModel(
    private var detectorEngine: ObjectDetectorEngine,
    private val bluetoothManager: BluetoothManager,
    private val gimbalController: GimbalController,
    private val detectorEngineFactory: (DetectorPipeline) -> ObjectDetectorEngine,
    private val commandIntervalMs: Long = 33L,
    private val clockMs: () -> Long = { SystemClock.elapsedRealtime() }
) : ViewModel() {
    private val _detections = MutableStateFlow<List<DetectionResult>>(emptyList())
    val detections: StateFlow<List<DetectionResult>> = _detections.asStateFlow()

    private val _detectorPipeline = MutableStateFlow(DetectorPipeline.ML_KIT)
    val detectorPipeline: StateFlow<DetectorPipeline> = _detectorPipeline.asStateFlow()

    private val _detectorPipelineError = MutableStateFlow<String?>(null)
    val detectorPipelineError: StateFlow<String?> = _detectorPipelineError.asStateFlow()

    private val _enabledClasses = MutableStateFlow(TrackableClassLabels.toSet())
    val enabledClasses: StateFlow<Set<String>> = _enabledClasses.asStateFlow()

    private val _trackingEngaged = MutableStateFlow(false)
    val trackingEngaged: StateFlow<Boolean> = _trackingEngaged.asStateFlow()

    private val _motorsEnabledRequested = MutableStateFlow(false)
    val motorsEnabledRequested: StateFlow<Boolean> =
        _motorsEnabledRequested.asStateFlow()

    private val _lockedTargetId = MutableStateFlow<Int?>(null)
    val lockedTargetId: StateFlow<Int?> = _lockedTargetId.asStateFlow()

    private val _selectedBluetoothDevice = MutableStateFlow<BluetoothDeviceInfo?>(null)
    val selectedBluetoothDevice: StateFlow<BluetoothDeviceInfo?> =
        _selectedBluetoothDevice.asStateFlow()

    val bluetoothConnectionState: StateFlow<BluetoothConnectionState> =
        bluetoothManager.connectionState
    val pairedBluetoothDevices: StateFlow<List<BluetoothDeviceInfo>> =
        bluetoothManager.pairedDevices

    private var lastCommandSentAtMs = 0L
    @Volatile
    private var detectorGeneration = 0L
    private val detectorLock = Any()
    private val mainHandler = Handler(Looper.getMainLooper())

    init {
        viewModelScope.launch {
            bluetoothManager.connectionState.collect { connectionState ->
                if (connectionState !is BluetoothConnectionState.Connected) {
                    _trackingEngaged.value = false
                    _lockedTargetId.value = null
                    _motorsEnabledRequested.value = false
                }
            }
        }
    }

    @androidx.annotation.OptIn(markerClass = [TransformExperimental::class])
    fun analyzeFrame(imageProxy: ImageProxy, previewView: PreviewView) {
        val sourceTransform = createImageProxyTransform(imageProxy)
        val activeDetectorGeneration: Long
        synchronized(detectorLock) {
            activeDetectorGeneration = detectorGeneration
            detectorEngine.analyze(imageProxy) { imageSpaceResults ->
                if (activeDetectorGeneration != detectorGeneration) {
                    return@analyze
                }

                handleDetectorResults(
                    detectorGeneration = activeDetectorGeneration,
                    previewView = previewView,
                    sourceTransform = sourceTransform,
                    imageSpaceResults = imageSpaceResults
                )
            }
        }
    }

    @androidx.annotation.OptIn(markerClass = [TransformExperimental::class])
    private fun handleDetectorResults(
        detectorGeneration: Long,
        previewView: PreviewView,
        sourceTransform: OutputTransform?,
        imageSpaceResults: List<DetectionResult>
    ) {
        mainHandler.post {
            if (detectorGeneration != this.detectorGeneration) {
                return@post
            }

            val filteredImageSpaceResults = filterDetectionsForCurrentPipeline(imageSpaceResults)
            val mappedResults = mapToPreviewCoordinates(
                previewView = previewView,
                sourceTransform = sourceTransform,
                detections = filteredImageSpaceResults
            )

            _detections.value = mappedResults
            maybeSendTrackingCommand(
                detections = mappedResults,
                previewWidth = previewView.width.toFloat(),
                previewHeight = previewView.height.toFloat()
            )
        }
    }

    fun refreshPairedBluetoothDevices() {
        viewModelScope.launch {
            bluetoothManager.refreshPairedDevices()
            val pairedDevices = pairedBluetoothDevices.value
            val selectedDevice = selectedBluetoothDevice.value
            if (selectedDevice == null ||
                pairedDevices.none { it.macAddress == selectedDevice.macAddress }
            ) {
                _selectedBluetoothDevice.value = pairedDevices.firstOrNull()
            }
        }
    }

    fun selectBluetoothDevice(device: BluetoothDeviceInfo) {
        _selectedBluetoothDevice.value = device
    }

    fun connectSelectedBluetoothDevice() {
        val macAddress = selectedBluetoothDevice.value?.macAddress ?: return
        _trackingEngaged.value = false
        _lockedTargetId.value = null
        _motorsEnabledRequested.value = false
        viewModelScope.launch {
            bluetoothManager.connect(macAddress)
            if (bluetoothConnectionState.value is BluetoothConnectionState.Connected) {
                sendMotorEnableCommand(enabled = false)
            }
        }
    }

    fun disconnectBluetooth() {
        viewModelScope.launch {
            sendStopCommand()
            sendMotorEnableCommand(enabled = false)
            _trackingEngaged.value = false
            _lockedTargetId.value = null
            _motorsEnabledRequested.value = false
            bluetoothManager.disconnect()
        }
    }

    fun lockOnTarget(id: Int?) {
        _lockedTargetId.value = id
    }

    fun selectDetectorPipeline(pipeline: DetectorPipeline) {
        if (pipeline == detectorPipeline.value) {
            _detectorPipelineError.value = null
            return
        }

        val newDetectorEngine = runCatching {
            detectorEngineFactory(pipeline)
        }.getOrElse { exception ->
            _detectorPipelineError.value = "Unable to start ${pipeline.displayName}: ${exception.message}"
            return
        }

        val wasTrackingEngaged = _trackingEngaged.value
        val previousDetectorEngine = synchronized(detectorLock) {
            val oldDetectorEngine = detectorEngine
            detectorEngine = newDetectorEngine
            detectorGeneration += 1
            oldDetectorEngine
        }

        previousDetectorEngine.close()
        _detectorPipeline.value = pipeline
        _detectorPipelineError.value = null
        _detections.value = emptyList()
        _lockedTargetId.value = null
        _trackingEngaged.value = false
        lastCommandSentAtMs = 0L

        if (wasTrackingEngaged) {
            viewModelScope.launch {
                sendStopCommand()
            }
        }
    }

    fun toggleClassVisibility(label: String, isEnabled: Boolean) {
        if (label !in TrackableClassLabels) {
            return
        }

        val updatedClasses = if (isEnabled) {
            _enabledClasses.value + label
        } else {
            _enabledClasses.value - label
        }
        _enabledClasses.value = updatedClasses
        if (detectorPipeline.value == DetectorPipeline.CUSTOM_TFLITE) {
            _detections.value = _detections.value.filterEnabledClasses(updatedClasses)
        }
    }

    fun setMotorsEnabledRequested(enabled: Boolean) {
        if (bluetoothConnectionState.value !is BluetoothConnectionState.Connected) {
            _trackingEngaged.value = false
            _motorsEnabledRequested.value = false
            return
        }

        _motorsEnabledRequested.value = enabled
        if (!enabled) {
            _trackingEngaged.value = false
            _lockedTargetId.value = null
        }

        viewModelScope.launch {
            if (!enabled) {
                sendStopCommand()
            }
            val sent = sendMotorEnableCommand(enabled)
            if (!sent) {
                _trackingEngaged.value = false
                _motorsEnabledRequested.value = false
            }
        }
    }

    fun setTrackingEngaged(engaged: Boolean) {
        if (engaged) {
            if (bluetoothConnectionState.value is BluetoothConnectionState.Connected &&
                _motorsEnabledRequested.value
            ) {
                _trackingEngaged.value = true
                lastCommandSentAtMs = 0L
            }
            return
        }

        if (_trackingEngaged.value) {
            viewModelScope.launch {
                sendStopCommand()
            }
        }
        _trackingEngaged.value = false
    }

    fun getPrimaryTarget(results: List<DetectionResult>): DetectionResult? {
        val currentLock = _lockedTargetId.value
        if (currentLock != null) {
            return results.find { it.trackingId == currentLock }
        }

        return results.maxByOrNull { detection ->
            detection.boundingBox.width() * detection.boundingBox.height()
        }
    }

    private fun maybeSendTrackingCommand(
        detections: List<DetectionResult>,
        previewWidth: Float,
        previewHeight: Float
    ) {
        if (!_trackingEngaged.value ||
            !_motorsEnabledRequested.value ||
            bluetoothConnectionState.value !is BluetoothConnectionState.Connected ||
            previewWidth <= 0f ||
            previewHeight <= 0f
        ) {
            return
        }

        val now = clockMs()
        if (now - lastCommandSentAtMs < commandIntervalMs) {
            return
        }
        lastCommandSentAtMs = now

        val primaryTarget = getPrimaryTarget(detections)
        val command = if (primaryTarget == null) {
            gimbalController.stopCommand()
        } else {
            val box = primaryTarget.boundingBox
            gimbalController.buildVelocityCommand(
                targetX = previewWidth / 2f,
                targetY = previewHeight / 2f,
                currentX = box.centerX(),
                currentY = box.centerY()
            )
        }

        viewModelScope.launch {
            bluetoothManager.send(command)
        }
    }

    @androidx.annotation.OptIn(markerClass = [TransformExperimental::class])
    private fun createImageProxyTransform(imageProxy: ImageProxy): OutputTransform? =
        runCatching {
            ImageProxyTransformFactory().apply {
                setUsingCropRect(false)
                setUsingRotationDegrees(true)
            }.getOutputTransform(imageProxy)
        }.getOrNull()

    @androidx.annotation.OptIn(markerClass = [TransformExperimental::class])
    private fun mapToPreviewCoordinates(
        previewView: PreviewView,
        sourceTransform: OutputTransform?,
        detections: List<DetectionResult>
    ): List<DetectionResult> {
        if (detections.isEmpty() || sourceTransform == null) {
            return emptyList()
        }

        val targetTransform = previewView.outputTransform ?: return emptyList()
        val coordinateTransform = runCatching {
            CoordinateTransform(sourceTransform, targetTransform)
        }.getOrNull() ?: return emptyList()

        return detections.mapNotNull { detection ->
            val mappedBox = RectF(detection.boundingBox)
            runCatching {
                coordinateTransform.mapRect(mappedBox)
            }.getOrNull() ?: return@mapNotNull null

            detection.copy(boundingBox = mappedBox)
        }
    }

    private fun List<DetectionResult>.filterEnabledClasses(
        enabledLabels: Set<String>
    ): List<DetectionResult> =
        filter { detection -> detection.label in enabledLabels }

    private fun filterDetectionsForCurrentPipeline(
        detections: List<DetectionResult>
    ): List<DetectionResult> =
        if (detectorPipeline.value == DetectorPipeline.CUSTOM_TFLITE) {
            detections.filterEnabledClasses(enabledClasses.value)
        } else {
            detections
        }

    private suspend fun sendStopCommand() {
        bluetoothManager.send(gimbalController.stopCommand())
    }

    private suspend fun sendMotorEnableCommand(enabled: Boolean): Boolean =
        bluetoothManager.send(if (enabled) MOTOR_ENABLE_COMMAND else MOTOR_DISABLE_COMMAND)

    override fun onCleared() {
        detectorEngine.close()
        bluetoothManager.close()
        super.onCleared()
    }

    companion object {
        private const val CUSTOM_TFLITE_MODEL_ASSET = "models/epoch_149_int8.tflite"
        private const val CUSTOM_TFLITE_TARGET_FPS = 4

        val TrackableClassLabels = listOf(
            "pedestrian",
            "person",
            "car",
            "van",
            "bus",
            "truck",
            "motor",
            "bicycle",
            "awning-tricycle",
            "tricycle",
            "others"
        )

        private const val MOTOR_ENABLE_COMMAND = "E:1\n"
        private const val MOTOR_DISABLE_COMMAND = "E:0\n"

        fun factory(context: Context): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T {
                    val applicationContext = context.applicationContext
                    return GimbalViewModel(
                        detectorEngine = MLKitDetector(),
                        bluetoothManager = AndroidBluetoothManager(applicationContext),
                        gimbalController = GimbalController(),
                        detectorEngineFactory = { pipeline ->
                            when (pipeline) {
                                DetectorPipeline.ML_KIT -> MLKitDetector()
                                DetectorPipeline.CUSTOM_TFLITE -> TFLiteDetectorEngine(
                                    context = applicationContext,
                                    modelAssetPath = CUSTOM_TFLITE_MODEL_ASSET,
                                    acceleration = TFLiteDetectorEngine.Acceleration.CPU,
                                    labels = TrackableClassLabels,
                                    configuredNumClasses = TrackableClassLabels.size,
                                    targetInferenceFps = CUSTOM_TFLITE_TARGET_FPS
                                )
                            }
                        }
                    ) as T
                }
            }
    }
}
