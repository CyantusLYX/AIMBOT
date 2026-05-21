package com.cyantus.aimbot.eye

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.cyantus.aimbot.eye.ui.theme.AimbotEyeTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AimbotEyeTheme {
                SensorNodeApp(this)
            }
        }
    }
}

@Composable
private fun SensorNodeApp(activity: MainActivity) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var status by remember { mutableStateOf("Idle") }
    var streaming by remember { mutableStateOf(false) }
    var targetIp by remember { mutableStateOf("192.168.1.100") }
    var targetPort by remember { mutableStateOf("5005") }
    var jpegQuality by remember { mutableIntStateOf(70) }
    var targetFps by remember { mutableIntStateOf(60) }
    var pendingStart by remember { mutableStateOf(false) }

    val controller = remember {
        SensorNodeController(
            context = context.applicationContext,
            lifecycleOwner = activity,
            scope = scope,
            onStatus = { status = it },
        )
    }
    DisposableEffect(controller) {
        onDispose { controller.close() }
    }

    fun startStreaming() {
        val port = targetPort.toIntOrNull()
        if (port == null) {
            status = "Invalid UDP port"
            streaming = false
            return
        }
        try {
            controller.start(
                SensorNodeConfig(
                    host = targetIp,
                    port = port,
                    jpegQuality = jpegQuality,
                    targetFps = targetFps,
                ),
            )
            streaming = true
        } catch (exc: Exception) {
            status = exc.message ?: exc.javaClass.simpleName
            streaming = false
        }
    }

    fun stopStreaming() {
        controller.stop()
        streaming = false
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted && pendingStart) {
            pendingStart = false
            startStreaming()
        } else if (!granted) {
            pendingStart = false
            streaming = false
            status = "Camera permission denied"
        }
    }

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        Column(
            modifier = Modifier
                .padding(innerPadding)
                .padding(16.dp)
                .fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("Gimbal Sensor Node", style = MaterialTheme.typography.headlineSmall)

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    OutlinedTextField(
                        value = targetIp,
                        onValueChange = { targetIp = it },
                        label = { Text("PC target IP") },
                        singleLine = true,
                        enabled = !streaming,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    OutlinedTextField(
                        value = targetPort,
                        onValueChange = { targetPort = it.filter(Char::isDigit).take(5) },
                        label = { Text("UDP port") },
                        singleLine = true,
                        enabled = !streaming,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.fillMaxWidth(),
                    )
                    LabeledSlider(
                        label = "JPEG quality",
                        value = jpegQuality,
                        range = 35..95,
                        enabled = !streaming,
                        onValueChange = { jpegQuality = it },
                    )
                    LabeledSlider(
                        label = "Target FPS",
                        value = targetFps,
                        range = 30..60,
                        enabled = !streaming,
                        onValueChange = { targetFps = it },
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Text(if (streaming) "Streaming" else "Stopped")
                        Switch(
                            checked = streaming,
                            onCheckedChange = { checked ->
                                if (checked) {
                                    val granted = ContextCompat.checkSelfPermission(
                                        context,
                                        Manifest.permission.CAMERA,
                                    ) == PackageManager.PERMISSION_GRANTED
                                    if (granted) {
                                        startStreaming()
                                    } else {
                                        pendingStart = true
                                        permissionLauncher.launch(Manifest.permission.CAMERA)
                                    }
                                } else {
                                    stopStreaming()
                                }
                            },
                        )
                    }
                }
            }

            Text(status, style = MaterialTheme.typography.bodyMedium)

            if (streaming) {
                Button(onClick = { stopStreaming() }) {
                    Text("Stop Streaming")
                }
            }
        }
    }
}

@Composable
private fun LabeledSlider(
    label: String,
    value: Int,
    range: IntRange,
    enabled: Boolean,
    onValueChange: (Int) -> Unit,
) {
    Column {
        Row {
            Text(label)
            Spacer(modifier = Modifier.width(8.dp))
            Text(value.toString())
        }
        Spacer(modifier = Modifier.height(4.dp))
        Slider(
            value = value.toFloat(),
            onValueChange = { onValueChange(it.toInt().coerceIn(range.first, range.last)) },
            valueRange = range.first.toFloat()..range.last.toFloat(),
            steps = (range.last - range.first - 1).coerceAtLeast(0),
            enabled = enabled,
        )
    }
}
