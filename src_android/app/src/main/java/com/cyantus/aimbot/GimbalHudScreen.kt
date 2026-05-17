package com.cyantus.aimbot

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.RectF
import android.util.Size
import android.view.Surface
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.FilterList
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as ComposeSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.zIndex
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.cyantus.aimbot.bluetooth.BluetoothConnectionState
import com.cyantus.aimbot.bluetooth.BluetoothDeviceInfo
import com.cyantus.aimbot.detection.DetectionResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private val AnalysisResolution = Size(640, 480)
private val BluetoothPermissions = listOf(
    Manifest.permission.BLUETOOTH_CONNECT,
    Manifest.permission.BLUETOOTH_SCAN
)
private val RequiredPermissions = listOf(Manifest.permission.CAMERA) + BluetoothPermissions

@Composable
fun GimbalCameraRoute(
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val viewModel: GimbalViewModel = viewModel(
        factory = remember(context) {
            GimbalViewModel.factory(context.applicationContext)
        }
    )
    val detections by viewModel.detections.collectAsStateWithLifecycle()
    val trackingEngaged by viewModel.trackingEngaged.collectAsStateWithLifecycle()
    val motorsEnabledRequested by viewModel.motorsEnabledRequested.collectAsStateWithLifecycle()
    val lockedTargetId by viewModel.lockedTargetId.collectAsStateWithLifecycle()
    val connectionState by viewModel.bluetoothConnectionState.collectAsStateWithLifecycle()
    val pairedDevices by viewModel.pairedBluetoothDevices.collectAsStateWithLifecycle()
    val selectedDevice by viewModel.selectedBluetoothDevice.collectAsStateWithLifecycle()
    val enabledClasses by viewModel.enabledClasses.collectAsStateWithLifecycle()
    val detectorPipeline by viewModel.detectorPipeline.collectAsStateWithLifecycle()
    val detectorPipelineError by viewModel.detectorPipelineError.collectAsStateWithLifecycle()

    var grantedPermissions by remember {
        mutableStateOf(context.grantedRequiredPermissions())
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { permissionResults ->
        grantedPermissions = RequiredPermissions
            .filter { permission ->
                permissionResults[permission] == true || context.hasPermission(permission)
            }
            .toSet()
    }

    LaunchedEffect(context) {
        val granted = context.grantedRequiredPermissions()
        grantedPermissions = granted
        if (granted.size < RequiredPermissions.size) {
            permissionLauncher.launch(RequiredPermissions.toTypedArray())
        }
    }

    val hasCameraPermission = Manifest.permission.CAMERA in grantedPermissions
    val hasBluetoothPermissions = BluetoothPermissions.all { it in grantedPermissions }

    LaunchedEffect(hasBluetoothPermissions) {
        if (hasBluetoothPermissions) {
            viewModel.refreshPairedBluetoothDevices()
        }
    }

    GimbalCameraScreen(
        modifier = modifier.fillMaxSize(),
        detections = detections,
        trackingEngaged = trackingEngaged,
        lockedTargetId = lockedTargetId,
        connectionState = connectionState,
        pairedDevices = pairedDevices,
        selectedDevice = selectedDevice,
        detectorPipeline = detectorPipeline,
        detectorPipelineError = detectorPipelineError,
        enabledClasses = enabledClasses,
        classLabels = GimbalViewModel.TrackableClassLabels,
        motorsEnabledRequested = motorsEnabledRequested,
        cameraPermissionGranted = hasCameraPermission,
        bluetoothPermissionsGranted = hasBluetoothPermissions,
        lifecycleOwner = lifecycleOwner,
        viewModel = viewModel,
        onSelectDevice = viewModel::selectBluetoothDevice,
        onRefreshDevices = viewModel::refreshPairedBluetoothDevices,
        onConnect = viewModel::connectSelectedBluetoothDevice,
        onDisconnect = viewModel::disconnectBluetooth,
        onDetectorPipelineSelected = viewModel::selectDetectorPipeline,
        onMotorsEnabledRequestedChange = viewModel::setMotorsEnabledRequested,
        onTrackingEngagedChange = viewModel::setTrackingEngaged,
        onClassEnabledChange = viewModel::toggleClassVisibility,
        onLockTarget = viewModel::lockOnTarget
    )
}

@Composable
private fun GimbalCameraScreen(
    modifier: Modifier,
    detections: List<DetectionResult>,
    trackingEngaged: Boolean,
    lockedTargetId: Int?,
    connectionState: BluetoothConnectionState,
    pairedDevices: List<BluetoothDeviceInfo>,
    selectedDevice: BluetoothDeviceInfo?,
    detectorPipeline: DetectorPipeline,
    detectorPipelineError: String?,
    enabledClasses: Set<String>,
    classLabels: List<String>,
    motorsEnabledRequested: Boolean,
    cameraPermissionGranted: Boolean,
    bluetoothPermissionsGranted: Boolean,
    lifecycleOwner: LifecycleOwner,
    viewModel: GimbalViewModel,
    onSelectDevice: (BluetoothDeviceInfo) -> Unit,
    onRefreshDevices: () -> Unit,
    onConnect: () -> Unit,
    onDisconnect: () -> Unit,
    onDetectorPipelineSelected: (DetectorPipeline) -> Unit,
    onMotorsEnabledRequestedChange: (Boolean) -> Unit,
    onTrackingEngagedChange: (Boolean) -> Unit,
    onClassEnabledChange: (String, Boolean) -> Unit,
    onLockTarget: (Int?) -> Unit
) {
    val context = LocalContext.current
    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }
    val analysisExecutor = remember {
        Executors.newSingleThreadExecutor()
    }

    DisposableEffect(Unit) {
        onDispose {
            analysisExecutor.shutdown()
        }
    }

    Box(modifier = modifier) {
        if (cameraPermissionGranted) {
            AndroidView(
                factory = { previewView },
                modifier = Modifier.fillMaxSize()
            )

            CameraBindingEffect(
                context = context,
                lifecycleOwner = lifecycleOwner,
                previewView = previewView,
                viewModel = viewModel,
                detectorPipeline = detectorPipeline,
                analysisExecutor = analysisExecutor
            )
        }

        HudOverlay(
            detections = detections,
            lockedTargetId = lockedTargetId,
            onLockTarget = onLockTarget,
            modifier = Modifier
                .fillMaxSize()
                .zIndex(1f)
        )

        BluetoothControlPanel(
            bluetoothPermissionsGranted = bluetoothPermissionsGranted,
            pairedDevices = pairedDevices,
            selectedDevice = selectedDevice,
            connectionState = connectionState,
            detectorPipeline = detectorPipeline,
            detectorPipelineError = detectorPipelineError,
            trackingEngaged = trackingEngaged,
            enabledClasses = enabledClasses,
            classLabels = classLabels,
            motorsEnabledRequested = motorsEnabledRequested,
            onSelectDevice = onSelectDevice,
            onRefreshDevices = onRefreshDevices,
            onConnect = onConnect,
            onDisconnect = onDisconnect,
            onDetectorPipelineSelected = onDetectorPipelineSelected,
            onMotorsEnabledRequestedChange = onMotorsEnabledRequestedChange,
            onTrackingEngagedChange = onTrackingEngagedChange,
            onClassEnabledChange = onClassEnabledChange,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
                .fillMaxWidth()
                .zIndex(2f)
        )
    }
}

@Composable
private fun CameraBindingEffect(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    previewView: PreviewView,
    viewModel: GimbalViewModel,
    detectorPipeline: DetectorPipeline,
    analysisExecutor: ExecutorService
) {
    DisposableEffect(
        context,
        lifecycleOwner,
        previewView,
        viewModel,
        detectorPipeline,
        analysisExecutor
    ) {
        var disposed = false
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val mainExecutor = ContextCompat.getMainExecutor(context)

        cameraProviderFuture.addListener(
            {
                if (disposed) {
                    return@addListener
                }

                val cameraProvider = cameraProviderFuture.get()
                previewView.post {
                    if (!disposed) {
                        bindCameraUseCases(
                            cameraProvider = cameraProvider,
                            lifecycleOwner = lifecycleOwner,
                            previewView = previewView,
                            viewModel = viewModel,
                            detectorPipeline = detectorPipeline,
                            analysisExecutor = analysisExecutor
                        )
                    }
                }
            },
            mainExecutor
        )

        onDispose {
            disposed = true
            if (cameraProviderFuture.isDone) {
                runCatching {
                    cameraProviderFuture.get().unbindAll()
                }
            }
        }
    }
}

private fun bindCameraUseCases(
    cameraProvider: ProcessCameraProvider,
    lifecycleOwner: LifecycleOwner,
    previewView: PreviewView,
    viewModel: GimbalViewModel,
    detectorPipeline: DetectorPipeline,
    analysisExecutor: ExecutorService
) {
    val targetRotation = previewView.display?.rotation ?: Surface.ROTATION_0
    val preview = Preview.Builder()
        .setTargetRotation(targetRotation)
        .build()
        .also { previewUseCase ->
            previewUseCase.setSurfaceProvider(previewView.surfaceProvider)
        }

    val analysisResolutionSelector = ResolutionSelector.Builder()
        .setResolutionStrategy(
            ResolutionStrategy(
                AnalysisResolution,
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
            )
        )
        .build()

    val imageAnalysisBuilder = ImageAnalysis.Builder()
        .setTargetRotation(targetRotation)
        .setResolutionSelector(analysisResolutionSelector)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setBackgroundExecutor(analysisExecutor)

    if (detectorPipeline == DetectorPipeline.CUSTOM_TFLITE) {
        imageAnalysisBuilder.setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
    } else {
        imageAnalysisBuilder.setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
    }

    val imageAnalysis = imageAnalysisBuilder
        .build()
        .also { analysisUseCase ->
            analysisUseCase.setAnalyzer(analysisExecutor) { imageProxy ->
                viewModel.analyzeFrame(imageProxy, previewView)
            }
        }

    cameraProvider.unbindAll()
    val viewPort = previewView.viewPort
    if (viewPort != null) {
        val useCaseGroup = UseCaseGroup.Builder()
            .addUseCase(preview)
            .addUseCase(imageAnalysis)
            .setViewPort(viewPort)
            .build()

        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            useCaseGroup
        )
    } else {
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalysis
        )
    }
}

@Composable
private fun BluetoothControlPanel(
    bluetoothPermissionsGranted: Boolean,
    pairedDevices: List<BluetoothDeviceInfo>,
    selectedDevice: BluetoothDeviceInfo?,
    connectionState: BluetoothConnectionState,
    detectorPipeline: DetectorPipeline,
    detectorPipelineError: String?,
    trackingEngaged: Boolean,
    enabledClasses: Set<String>,
    classLabels: List<String>,
    motorsEnabledRequested: Boolean,
    onSelectDevice: (BluetoothDeviceInfo) -> Unit,
    onRefreshDevices: () -> Unit,
    onConnect: () -> Unit,
    onDisconnect: () -> Unit,
    onDetectorPipelineSelected: (DetectorPipeline) -> Unit,
    onMotorsEnabledRequestedChange: (Boolean) -> Unit,
    onTrackingEngagedChange: (Boolean) -> Unit,
    onClassEnabledChange: (String, Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    var menuExpanded by remember {
        mutableStateOf(false)
    }
    var classFilterSheetVisible by remember {
        mutableStateOf(false)
    }
    val connected = connectionState is BluetoothConnectionState.Connected
    val connecting = connectionState is BluetoothConnectionState.Connecting
    val selectedLabel = selectedDevice?.let { "${it.name} (${it.macAddress})" }
        ?: "No paired device"

    Box(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color.Black.copy(alpha = 0.48f))
            .border(
                width = 1.dp,
                color = Color.White.copy(alpha = 0.22f),
                shape = RoundedCornerShape(8.dp)
            )
            .padding(12.dp)
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = connectionStateLabel(
                        bluetoothPermissionsGranted = bluetoothPermissionsGranted,
                        connectionState = connectionState
                    ),
                    color = Color.White.copy(alpha = 0.78f),
                    style = MaterialTheme.typography.labelMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )

                IconButton(
                    onClick = { classFilterSheetVisible = true },
                    modifier = Modifier.size(40.dp)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.FilterList,
                        contentDescription = "Filter classes",
                        tint = Color.White.copy(alpha = 0.88f)
                    )
                }
            }

            DetectorPipelineSelector(
                selectedPipeline = detectorPipeline,
                pipelineError = detectorPipelineError,
                onPipelineSelected = onDetectorPipelineSelected
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(modifier = Modifier.weight(1f)) {
                    OutlinedButton(
                        onClick = { menuExpanded = true },
                        enabled = bluetoothPermissionsGranted &&
                            pairedDevices.isNotEmpty() &&
                            !connecting &&
                            !connected,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = selectedLabel,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }

                    DropdownMenu(
                        expanded = menuExpanded,
                        onDismissRequest = { menuExpanded = false }
                    ) {
                        pairedDevices.forEach { device ->
                            DropdownMenuItem(
                                text = {
                                    Column {
                                        Text(
                                            text = device.name,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                        Text(
                                            text = device.macAddress,
                                            color = Color.White.copy(alpha = 0.68f),
                                            style = MaterialTheme.typography.labelSmall,
                                            maxLines = 1
                                        )
                                    }
                                },
                                onClick = {
                                    onSelectDevice(device)
                                    menuExpanded = false
                                }
                            )
                        }
                    }
                }

                OutlinedButton(
                    onClick = onRefreshDevices,
                    enabled = bluetoothPermissionsGranted && !connecting && !connected
                ) {
                    Text("Refresh")
                }

                Button(
                    onClick = {
                        if (connected) {
                            onDisconnect()
                        } else {
                            onConnect()
                        }
                    },
                    enabled = bluetoothPermissionsGranted &&
                        !connecting &&
                        (connected || selectedDevice != null)
                ) {
                    Text(
                        text = when {
                            connected -> "Disconnect"
                            else -> "Connect"
                        }
                    )
                }
            }

            FilledTonalButton(
                onClick = {
                    onMotorsEnabledRequestedChange(!motorsEnabledRequested)
                },
                enabled = connected,
                colors = ButtonDefaults.filledTonalButtonColors(
                    containerColor = if (motorsEnabledRequested) {
                        Color(0xFF39FF14).copy(alpha = 0.24f)
                    } else {
                        Color(0xFFB3261E).copy(alpha = 0.46f)
                    },
                    contentColor = Color.White,
                    disabledContainerColor = Color.White.copy(alpha = 0.10f),
                    disabledContentColor = Color.White.copy(alpha = 0.42f)
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
            ) {
                Text(
                    text = if (motorsEnabledRequested) {
                        "Motors: On"
                    } else {
                        "Motors: Off"
                    }
                )
            }

            FilledTonalButton(
                onClick = { onTrackingEngagedChange(!trackingEngaged) },
                enabled = connected && motorsEnabledRequested,
                colors = ButtonDefaults.filledTonalButtonColors(
                    containerColor = if (trackingEngaged) {
                        Color(0xFFB3261E).copy(alpha = 0.92f)
                    } else {
                        Color(0xFF39FF14).copy(alpha = 0.26f)
                    },
                    contentColor = Color.White,
                    disabledContainerColor = Color.White.copy(alpha = 0.10f),
                    disabledContentColor = Color.White.copy(alpha = 0.42f)
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                Text(
                    text = if (trackingEngaged) {
                        "Disengage Tracking"
                    } else {
                        "Engage Tracking"
                    }
                )
            }
        }
    }

    if (classFilterSheetVisible) {
        ClassFilterBottomSheet(
            classLabels = classLabels,
            enabledClasses = enabledClasses,
            onClassEnabledChange = onClassEnabledChange,
            onDismiss = { classFilterSheetVisible = false }
        )
    }
}

@Composable
private fun DetectorPipelineSelector(
    selectedPipeline: DetectorPipeline,
    pipelineError: String?,
    onPipelineSelected: (DetectorPipeline) -> Unit
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            DetectorPipelineButton(
                text = DetectorPipeline.ML_KIT.displayName,
                selected = selectedPipeline == DetectorPipeline.ML_KIT,
                onClick = { onPipelineSelected(DetectorPipeline.ML_KIT) },
                modifier = Modifier.weight(1f)
            )
            DetectorPipelineButton(
                text = DetectorPipeline.CUSTOM_TFLITE.displayName,
                selected = selectedPipeline == DetectorPipeline.CUSTOM_TFLITE,
                onClick = { onPipelineSelected(DetectorPipeline.CUSTOM_TFLITE) },
                modifier = Modifier.weight(1f)
            )
        }

        if (pipelineError != null) {
            Text(
                text = pipelineError,
                color = Color(0xFFFFB4AB),
                style = MaterialTheme.typography.labelSmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun DetectorPipelineButton(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    if (selected) {
        FilledTonalButton(
            onClick = onClick,
            modifier = modifier.height(40.dp),
            colors = ButtonDefaults.filledTonalButtonColors(
                containerColor = Color(0xFF39FF14).copy(alpha = 0.24f),
                contentColor = Color.White
            )
        ) {
            Text(
                text = text,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    } else {
        OutlinedButton(
            onClick = onClick,
            modifier = modifier.height(40.dp)
        ) {
            Text(
                text = text,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ClassFilterBottomSheet(
    classLabels: List<String>,
    enabledClasses: Set<String>,
    onClassEnabledChange: (String, Boolean) -> Unit,
    onDismiss: () -> Unit
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Color(0xFF101418),
        contentColor = Color.White
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 24.dp, end = 24.dp, bottom = 28.dp)
        ) {
            Text(
                text = "Class Filters",
                style = MaterialTheme.typography.titleLarge,
                color = Color.White,
                modifier = Modifier.padding(bottom = 12.dp)
            )

            LazyColumn(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = 440.dp)
            ) {
                items(classLabels, key = { label -> label }) { label ->
                    ClassFilterRow(
                        label = label,
                        checked = label in enabledClasses,
                        onCheckedChange = { isChecked ->
                            onClassEnabledChange(label, isChecked)
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun ClassFilterRow(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Column {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { onCheckedChange(!checked) }
                .padding(vertical = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyLarge,
                color = Color.White,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            Switch(
                checked = checked,
                onCheckedChange = onCheckedChange,
                modifier = Modifier.width(52.dp)
            )
        }
        HorizontalDivider(color = Color.White.copy(alpha = 0.10f))
    }
}

@Composable
private fun HudOverlay(
    detections: List<DetectionResult>,
    lockedTargetId: Int?,
    onLockTarget: (Int?) -> Unit,
    modifier: Modifier = Modifier
) {
    Canvas(
        modifier = modifier.pointerInput(detections) {
            detectTapGestures { tapOffset ->
                val tappedTargetId = detections
                    .asReversed()
                    .firstOrNull { detection ->
                        detection.trackingId != null &&
                            detection.boundingBox.contains(tapOffset.x, tapOffset.y)
                    }
                    ?.trackingId

                onLockTarget(tappedTargetId)
            }
        }
    ) {
        val center = Offset(size.width / 2f, size.height / 2f)
        val crosshairColor = Color.White.copy(alpha = 0.56f)
        val crosshairStroke = 1.dp.toPx()
        val armLength = 26.dp.toPx()
        val gap = 7.dp.toPx()

        drawLine(
            color = crosshairColor,
            start = Offset(center.x - armLength, center.y),
            end = Offset(center.x - gap, center.y),
            strokeWidth = crosshairStroke
        )
        drawLine(
            color = crosshairColor,
            start = Offset(center.x + gap, center.y),
            end = Offset(center.x + armLength, center.y),
            strokeWidth = crosshairStroke
        )
        drawLine(
            color = crosshairColor,
            start = Offset(center.x, center.y - armLength),
            end = Offset(center.x, center.y - gap),
            strokeWidth = crosshairStroke
        )
        drawLine(
            color = crosshairColor,
            start = Offset(center.x, center.y + gap),
            end = Offset(center.x, center.y + armLength),
            strokeWidth = crosshairStroke
        )

        detections.forEach { detection ->
            val isLocked = detection.trackingId != null &&
                detection.trackingId == lockedTargetId
            drawDetectionBox(
                box = detection.boundingBox,
                isLocked = isLocked
            )
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawDetectionBox(
    box: RectF,
    isLocked: Boolean
) {
    val neonGreen = Color(0xFF39FF14)
    val lockRed = Color(0xFFFF1744)
    val strokeWidth = if (isLocked) 4.dp.toPx() else 2.dp.toPx()

    drawRect(
        color = if (isLocked) lockRed else neonGreen,
        topLeft = Offset(box.left, box.top),
        size = ComposeSize(box.width(), box.height()),
        style = Stroke(width = strokeWidth),
        alpha = 0.9f
    )
}

private fun connectionStateLabel(
    bluetoothPermissionsGranted: Boolean,
    connectionState: BluetoothConnectionState
): String {
    if (!bluetoothPermissionsGranted) {
        return "Bluetooth permission required"
    }

    return when (connectionState) {
        BluetoothConnectionState.Unavailable -> "Bluetooth unavailable"
        BluetoothConnectionState.Disconnected -> "Bluetooth disconnected"
        BluetoothConnectionState.Connecting -> "Connecting..."
        is BluetoothConnectionState.Connected -> {
            "Connected: ${connectionState.device.name}"
        }
        is BluetoothConnectionState.Error -> connectionState.message
    }
}

private fun Context.grantedRequiredPermissions(): Set<String> =
    RequiredPermissions.filter(::hasPermission).toSet()

private fun Context.hasPermission(permission: String): Boolean =
    ContextCompat.checkSelfPermission(
        this,
        permission
    ) == PackageManager.PERMISSION_GRANTED
