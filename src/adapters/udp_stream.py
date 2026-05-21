"""UDP packet parsing and low-latency JPEG frame reassembly."""
from __future__ import annotations

import select
import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

MAGIC = b"GBR1"
VERSION = 1
TYPE_VIDEO_FRAGMENT = 1
TYPE_IMU_SAMPLE = 2
COMMON_HEADER_BYTES = 24
VIDEO_HEADER_BYTES = 84
IMU_HEADER_BYTES = 68
MAX_DATAGRAM_BYTES = 1460
VIDEO_PAYLOAD_BYTES = MAX_DATAGRAM_BYTES - VIDEO_HEADER_BYTES

COMMON_STRUCT = struct.Struct("<4sBBHHHIQ")
VIDEO_STRUCT = struct.Struct("<4sBBHHHIQIHHIHHB3xIfffffffff")
IMU_STRUCT = struct.Struct("<4sBBHHHIQIfffffffffHH")


class PacketParseError(ValueError):
    """Raised when a UDP datagram does not match the GBR1 protocol."""


@dataclass(frozen=True)
class ImuPacket:
    sequence: int
    device_time_ns: int
    imu_sequence: int
    roll: float
    pitch: float
    yaw: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    accel_x: float
    accel_y: float
    accel_z: float
    source_flags: int


@dataclass(frozen=True)
class VideoFragment:
    sequence: int
    device_time_ns: int
    frame_id: int
    fragment_index: int
    fragment_count: int
    jpeg_length: int
    width: int
    height: int
    jpeg_quality: int
    imu: ImuPacket
    payload: bytes


@dataclass(frozen=True)
class DecodedFrame:
    frame_id: int
    frame: np.ndarray
    width: int
    height: int
    jpeg_quality: int
    device_time_ns: int
    imu: ImuPacket


def parse_packet(data: bytes) -> VideoFragment | ImuPacket:
    """Parse one GBR1 UDP packet."""
    if len(data) < COMMON_HEADER_BYTES:
        raise PacketParseError("packet shorter than common header")

    magic, version, packet_type, header_len, _flags, _reserved, sequence, device_time_ns = COMMON_STRUCT.unpack_from(data)
    if magic != MAGIC:
        raise PacketParseError("invalid magic")
    if version != VERSION:
        raise PacketParseError(f"unsupported protocol version: {version}")

    if packet_type == TYPE_VIDEO_FRAGMENT:
        if header_len != VIDEO_HEADER_BYTES or len(data) < VIDEO_HEADER_BYTES:
            raise PacketParseError("invalid video header length")
        fields = VIDEO_STRUCT.unpack_from(data)
        (
            _magic,
            _version,
            _packet_type,
            _header_len,
            _flags,
            _reserved,
            sequence,
            device_time_ns,
            frame_id,
            fragment_index,
            fragment_count,
            jpeg_length,
            width,
            height,
            jpeg_quality,
            imu_sequence,
            roll,
            pitch,
            yaw,
            gyro_x,
            gyro_y,
            gyro_z,
            accel_x,
            accel_y,
            accel_z,
        ) = fields
        if fragment_count <= 0:
            raise PacketParseError("fragment_count must be positive")
        if fragment_index >= fragment_count:
            raise PacketParseError("fragment_index out of range")
        payload = data[VIDEO_HEADER_BYTES:]
        if len(payload) > VIDEO_PAYLOAD_BYTES:
            raise PacketParseError("video payload exceeds MTU stride")
        if jpeg_length < len(payload):
            raise PacketParseError("jpeg_length shorter than payload")
        imu = ImuPacket(
            sequence=sequence,
            device_time_ns=device_time_ns,
            imu_sequence=imu_sequence,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            gyro_x=gyro_x,
            gyro_y=gyro_y,
            gyro_z=gyro_z,
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            source_flags=0,
        )
        return VideoFragment(
            sequence=sequence,
            device_time_ns=device_time_ns,
            frame_id=frame_id,
            fragment_index=fragment_index,
            fragment_count=fragment_count,
            jpeg_length=jpeg_length,
            width=width,
            height=height,
            jpeg_quality=jpeg_quality,
            imu=imu,
            payload=payload,
        )

    if packet_type == TYPE_IMU_SAMPLE:
        if header_len != IMU_HEADER_BYTES or len(data) != IMU_HEADER_BYTES:
            raise PacketParseError("invalid IMU header length")
        fields = IMU_STRUCT.unpack_from(data)
        (
            _magic,
            _version,
            _packet_type,
            _header_len,
            _flags,
            _reserved,
            sequence,
            device_time_ns,
            imu_sequence,
            roll,
            pitch,
            yaw,
            gyro_x,
            gyro_y,
            gyro_z,
            accel_x,
            accel_y,
            accel_z,
            source_flags,
            _imu_reserved,
        ) = fields
        return ImuPacket(
            sequence=sequence,
            device_time_ns=device_time_ns,
            imu_sequence=imu_sequence,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            gyro_x=gyro_x,
            gyro_y=gyro_y,
            gyro_z=gyro_z,
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            source_flags=source_flags,
        )

    raise PacketParseError(f"unknown packet type: {packet_type}")


def pack_imu_packet(sequence: int, imu: ImuPacket) -> bytes:
    """Build an IMU packet. Used by tests and small simulators."""
    return IMU_STRUCT.pack(
        MAGIC,
        VERSION,
        TYPE_IMU_SAMPLE,
        IMU_HEADER_BYTES,
        0,
        0,
        sequence,
        imu.device_time_ns,
        imu.imu_sequence,
        imu.roll,
        imu.pitch,
        imu.yaw,
        imu.gyro_x,
        imu.gyro_y,
        imu.gyro_z,
        imu.accel_x,
        imu.accel_y,
        imu.accel_z,
        imu.source_flags,
        0,
    )


def pack_video_fragment(
    *,
    sequence: int,
    frame_id: int,
    fragment_index: int,
    fragment_count: int,
    jpeg: bytes,
    payload_offset: int,
    width: int,
    height: int,
    jpeg_quality: int,
    imu: ImuPacket,
) -> bytes:
    """Build one video fragment packet. Used by tests and small simulators."""
    payload = jpeg[payload_offset : payload_offset + VIDEO_PAYLOAD_BYTES]
    return (
        VIDEO_STRUCT.pack(
            MAGIC,
            VERSION,
            TYPE_VIDEO_FRAGMENT,
            VIDEO_HEADER_BYTES,
            0,
            0,
            sequence,
            imu.device_time_ns,
            frame_id,
            fragment_index,
            fragment_count,
            len(jpeg),
            width,
            height,
            jpeg_quality,
            imu.imu_sequence,
            imu.roll,
            imu.pitch,
            imu.yaw,
            imu.gyro_x,
            imu.gyro_y,
            imu.gyro_z,
            imu.accel_x,
            imu.accel_y,
            imu.accel_z,
        )
        + payload
    )


class FrameReassembler:
    """Reassemble fragmented JPEG frames and drop stale incomplete frames."""

    def __init__(self, stale_frame_window: int = 8, fragment_ttl_s: float = 0.25) -> None:
        self.stale_frame_window = max(1, int(stale_frame_window))
        self.fragment_ttl_s = max(0.05, float(fragment_ttl_s))
        self._buffers: dict[int, dict] = {}
        self._latest_seen_frame_id = -1
        self._latest_completed_frame_id = -1

    def push(self, fragment: VideoFragment) -> Optional[DecodedFrame]:
        now = time.monotonic()
        self._latest_seen_frame_id = max(self._latest_seen_frame_id, fragment.frame_id)
        self._evict_stale(now)

        if fragment.frame_id <= self._latest_completed_frame_id:
            return None
        if fragment.frame_id < self._latest_seen_frame_id - self.stale_frame_window:
            return None

        buffer = self._buffers.get(fragment.frame_id)
        if buffer is None:
            buffer = {
                "created_at": now,
                "fragment_count": fragment.fragment_count,
                "jpeg_length": fragment.jpeg_length,
                "width": fragment.width,
                "height": fragment.height,
                "quality": fragment.jpeg_quality,
                "device_time_ns": fragment.device_time_ns,
                "imu": fragment.imu,
                "fragments": {},
            }
            self._buffers[fragment.frame_id] = buffer

        if (
            buffer["fragment_count"] != fragment.fragment_count
            or buffer["jpeg_length"] != fragment.jpeg_length
            or buffer["width"] != fragment.width
            or buffer["height"] != fragment.height
        ):
            self._buffers.pop(fragment.frame_id, None)
            return None

        fragments: dict[int, bytes] = buffer["fragments"]
        fragments.setdefault(fragment.fragment_index, fragment.payload)
        if len(fragments) != fragment.fragment_count:
            return None

        try:
            jpeg = b"".join(fragments[index] for index in range(fragment.fragment_count))
        except KeyError:
            return None
        self._buffers.pop(fragment.frame_id, None)
        if len(jpeg) != fragment.jpeg_length:
            return None

        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None

        self._latest_completed_frame_id = max(self._latest_completed_frame_id, fragment.frame_id)
        self._drop_older_than(fragment.frame_id)
        return DecodedFrame(
            frame_id=fragment.frame_id,
            frame=image,
            width=fragment.width,
            height=fragment.height,
            jpeg_quality=fragment.jpeg_quality,
            device_time_ns=fragment.device_time_ns,
            imu=fragment.imu,
        )

    def _evict_stale(self, now: float) -> None:
        too_old = [
            frame_id
            for frame_id, buffer in self._buffers.items()
            if now - float(buffer["created_at"]) > self.fragment_ttl_s
            or frame_id < self._latest_seen_frame_id - self.stale_frame_window
        ]
        for frame_id in too_old:
            self._buffers.pop(frame_id, None)

    def _drop_older_than(self, frame_id: int) -> None:
        for old_frame_id in [candidate for candidate in self._buffers if candidate < frame_id]:
            self._buffers.pop(old_frame_id, None)


class UdpFrameReceiver:
    """Non-blocking UDP receiver that returns the latest fully decoded frame."""

    def __init__(self, host: str, port: int, reassembler: Optional[FrameReassembler] = None) -> None:
        self.reassembler = reassembler or FrameReassembler()
        self.last_imu: Optional[ImuPacket] = None
        self.invalid_packets = 0
        self.received_packets = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind((host, port))

    def recv_latest(self, timeout_s: float = 0.0) -> Optional[DecodedFrame]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        latest_frame: Optional[DecodedFrame] = None

        while True:
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select([self._socket], [], [], remaining)
            if not readable:
                break

            while True:
                try:
                    data, _addr = self._socket.recvfrom(MAX_DATAGRAM_BYTES)
                except BlockingIOError:
                    break
                self.received_packets += 1
                try:
                    packet = parse_packet(data)
                except PacketParseError:
                    self.invalid_packets += 1
                    continue
                if isinstance(packet, ImuPacket):
                    self.last_imu = packet
                else:
                    self.last_imu = packet.imu
                    decoded = self.reassembler.push(packet)
                    if decoded is not None:
                        latest_frame = decoded

            if time.monotonic() >= deadline:
                break

        return latest_frame

    def close(self) -> None:
        self._socket.close()
