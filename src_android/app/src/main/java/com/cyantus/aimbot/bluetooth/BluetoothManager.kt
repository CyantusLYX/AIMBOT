package com.cyantus.aimbot.bluetooth

import kotlinx.coroutines.flow.StateFlow

interface BluetoothManager {
    val connectionState: StateFlow<BluetoothConnectionState>
    val pairedDevices: StateFlow<List<BluetoothDeviceInfo>>

    suspend fun refreshPairedDevices()
    suspend fun connect(macAddress: String)
    suspend fun send(data: String): Boolean
    suspend fun disconnect()
    fun close()
}
