package com.cyantus.aimbot.bluetooth

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothSocket
import android.content.Context
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import java.io.IOException
import java.util.UUID
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import android.bluetooth.BluetoothManager as PlatformBluetoothManager

class AndroidBluetoothManager(
    context: Context,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO
) : BluetoothManager {
    private val appContext = context.applicationContext
    private val bluetoothAdapter =
        appContext.getSystemService(PlatformBluetoothManager::class.java)?.adapter
    private val socketMutex = Mutex()
    private var socket: BluetoothSocket? = null

    private val _connectionState = MutableStateFlow<BluetoothConnectionState>(
        if (bluetoothAdapter == null) {
            BluetoothConnectionState.Unavailable
        } else {
            BluetoothConnectionState.Disconnected
        }
    )
    override val connectionState: StateFlow<BluetoothConnectionState> =
        _connectionState.asStateFlow()

    private val _pairedDevices = MutableStateFlow<List<BluetoothDeviceInfo>>(emptyList())
    override val pairedDevices: StateFlow<List<BluetoothDeviceInfo>> =
        _pairedDevices.asStateFlow()

    @SuppressLint("MissingPermission")
    override suspend fun refreshPairedDevices() = withContext(ioDispatcher) {
        val adapter = bluetoothAdapter ?: run {
            _pairedDevices.value = emptyList()
            _connectionState.value = BluetoothConnectionState.Unavailable
            return@withContext
        }

        if (!hasBluetoothConnectPermission()) {
            _pairedDevices.value = emptyList()
            _connectionState.value = BluetoothConnectionState.Error(
                "Bluetooth permission is required"
            )
            return@withContext
        }

        if (!adapter.isEnabled) {
            _pairedDevices.value = emptyList()
            _connectionState.value = BluetoothConnectionState.Error(
                "Bluetooth is turned off"
            )
            return@withContext
        }

        _pairedDevices.value = adapter.bondedDevices
            .map { device ->
                BluetoothDeviceInfo(
                    name = device.name ?: "Unnamed device",
                    macAddress = device.address
                )
            }
            .sortedWith(compareBy({ it.name.lowercase() }, { it.macAddress }))

        if (_connectionState.value !is BluetoothConnectionState.Connected &&
            _connectionState.value !is BluetoothConnectionState.Connecting
        ) {
            _connectionState.value = BluetoothConnectionState.Disconnected
        }
    }

    @SuppressLint("MissingPermission")
    override suspend fun connect(macAddress: String) = withContext(ioDispatcher) {
        socketMutex.withLock {
            val adapter = bluetoothAdapter ?: run {
                _connectionState.value = BluetoothConnectionState.Unavailable
                return@withLock
            }

            if (!hasBluetoothConnectPermission()) {
                _connectionState.value = BluetoothConnectionState.Error(
                    "Bluetooth permission is required"
                )
                return@withLock
            }

            if (!adapter.isEnabled) {
                _connectionState.value = BluetoothConnectionState.Error(
                    "Bluetooth is turned off"
                )
                return@withLock
            }

            closeSocketLocked()
            _connectionState.value = BluetoothConnectionState.Connecting

            try {
                adapter.cancelDiscovery()
                val device = adapter.getRemoteDevice(macAddress)
                val nextSocket = device.createRfcommSocketToServiceRecord(SPP_UUID)
                nextSocket.connect()
                socket = nextSocket

                _connectionState.value = BluetoothConnectionState.Connected(
                    BluetoothDeviceInfo(
                        name = device.name ?: "ESP32",
                        macAddress = device.address
                    )
                )
            } catch (exception: IOException) {
                closeSocketLocked()
                _connectionState.value = BluetoothConnectionState.Error(
                    exception.message ?: "Bluetooth connection failed"
                )
            } catch (exception: SecurityException) {
                closeSocketLocked()
                _connectionState.value = BluetoothConnectionState.Error(
                    exception.message ?: "Bluetooth permission denied"
                )
            } catch (exception: IllegalArgumentException) {
                closeSocketLocked()
                _connectionState.value = BluetoothConnectionState.Error(
                    exception.message ?: "Invalid Bluetooth MAC address"
                )
            }
        }
    }

    override suspend fun send(data: String): Boolean = withContext(ioDispatcher) {
        socketMutex.withLock {
            val activeSocket = socket ?: return@withLock false
            if (_connectionState.value !is BluetoothConnectionState.Connected) {
                return@withLock false
            }

            try {
                activeSocket.outputStream.write(data.toByteArray(Charsets.US_ASCII))
                activeSocket.outputStream.flush()
                true
            } catch (exception: IOException) {
                closeSocketLocked()
                _connectionState.value = BluetoothConnectionState.Error(
                    exception.message ?: "Bluetooth write failed"
                )
                false
            } catch (exception: SecurityException) {
                closeSocketLocked()
                _connectionState.value = BluetoothConnectionState.Error(
                    exception.message ?: "Bluetooth permission denied"
                )
                false
            }
        }
    }

    override suspend fun disconnect() = withContext(ioDispatcher) {
        socketMutex.withLock {
            closeSocketLocked()
            _connectionState.value = if (bluetoothAdapter == null) {
                BluetoothConnectionState.Unavailable
            } else {
                BluetoothConnectionState.Disconnected
            }
        }
    }

    override fun close() {
        runCatching {
            socket?.close()
        }
        socket = null
        _connectionState.value = if (bluetoothAdapter == null) {
            BluetoothConnectionState.Unavailable
        } else {
            BluetoothConnectionState.Disconnected
        }
    }

    private fun closeSocketLocked() {
        runCatching {
            socket?.close()
        }
        socket = null
    }

    private fun hasBluetoothConnectPermission(): Boolean =
        ContextCompat.checkSelfPermission(
            appContext,
            Manifest.permission.BLUETOOTH_CONNECT
        ) == PackageManager.PERMISSION_GRANTED

    companion object {
        val SPP_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
    }
}
