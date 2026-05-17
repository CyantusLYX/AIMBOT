package com.cyantus.aimbot.bluetooth

sealed interface BluetoothConnectionState {
    data object Unavailable : BluetoothConnectionState
    data object Disconnected : BluetoothConnectionState
    data object Connecting : BluetoothConnectionState
    data class Connected(val device: BluetoothDeviceInfo) : BluetoothConnectionState
    data class Error(val message: String) : BluetoothConnectionState
}
