from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adapters.udp_stream import (  # noqa: E402
    IMU_HEADER_BYTES,
    VIDEO_HEADER_BYTES,
    VIDEO_PAYLOAD_BYTES,
    FrameReassembler,
    ImuPacket,
    VideoFragment,
    pack_imu_packet,
    pack_video_fragment,
    parse_packet,
)


def synthetic_jpeg(width: int = 96, height: int = 64) -> bytes:
    yy, xx = np.indices((height, width))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (xx * 3 + yy * 5) % 256
    frame[:, :, 1] = (xx * 7 + yy * 11) % 256
    frame[:, :, 2] = (xx * 13 + yy * 17) % 256
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    assert ok
    return encoded.tobytes()


def imu_sample(sequence: int = 10) -> ImuPacket:
    return ImuPacket(
        sequence=sequence,
        device_time_ns=123456,
        imu_sequence=sequence,
        roll=1.0,
        pitch=2.0,
        yaw=3.0,
        gyro_x=4.0,
        gyro_y=5.0,
        gyro_z=6.0,
        accel_x=7.0,
        accel_y=8.0,
        accel_z=9.0,
        source_flags=7,
    )


def video_packets(frame_id: int, jpeg: bytes) -> list[VideoFragment]:
    count = max(1, math.ceil(len(jpeg) / VIDEO_PAYLOAD_BYTES))
    packets = []
    for index in range(count):
        packet = pack_video_fragment(
            sequence=index + 1,
            frame_id=frame_id,
            fragment_index=index,
            fragment_count=count,
            jpeg=jpeg,
            payload_offset=index * VIDEO_PAYLOAD_BYTES,
            width=96,
            height=64,
            jpeg_quality=80,
            imu=imu_sample(),
        )
        parsed = parse_packet(packet)
        assert isinstance(parsed, VideoFragment)
        packets.append(parsed)
    return packets


class UdpStreamTests(unittest.TestCase):
    def test_imu_packet_unpacking(self) -> None:
        parsed = parse_packet(pack_imu_packet(99, imu_sample(77)))

        self.assertEqual(IMU_HEADER_BYTES, len(pack_imu_packet(99, imu_sample(77))))
        self.assertIsInstance(parsed, ImuPacket)
        self.assertEqual(99, parsed.sequence)
        self.assertEqual(77, parsed.imu_sequence)
        self.assertEqual(7, parsed.source_flags)
        self.assertAlmostEqual(3.0, parsed.yaw)

    def test_video_header_unpacking(self) -> None:
        jpeg = synthetic_jpeg()
        fragment = video_packets(42, jpeg)[0]

        self.assertEqual(42, fragment.frame_id)
        self.assertEqual(0, fragment.fragment_index)
        self.assertEqual(len(jpeg), fragment.jpeg_length)
        self.assertEqual(96, fragment.width)
        self.assertEqual(64, fragment.height)
        self.assertLessEqual(len(fragment.payload) + VIDEO_HEADER_BYTES, 1460)

    def test_out_of_order_reassembly_decodes_jpeg(self) -> None:
        jpeg = synthetic_jpeg()
        fragments = video_packets(7, jpeg)
        reassembler = FrameReassembler()

        decoded = None
        for fragment in reversed(fragments):
            decoded = reassembler.push(fragment) or decoded

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(7, decoded.frame_id)
        self.assertEqual((64, 96, 3), decoded.frame.shape)

    def test_duplicate_fragment_is_ignored(self) -> None:
        jpeg = synthetic_jpeg()
        fragments = video_packets(8, jpeg)
        reassembler = FrameReassembler()

        self.assertIsNone(reassembler.push(fragments[0]))
        self.assertIsNone(reassembler.push(fragments[0]))
        decoded = None
        for fragment in fragments[1:]:
            decoded = reassembler.push(fragment) or decoded

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(8, decoded.frame_id)

    def test_stale_incomplete_frame_is_dropped(self) -> None:
        jpeg = synthetic_jpeg()
        stale = video_packets(1, jpeg)
        fresh = video_packets(20, jpeg)
        reassembler = FrameReassembler(stale_frame_window=2)

        self.assertIsNone(reassembler.push(stale[0]))
        decoded = None
        for fragment in fresh:
            decoded = reassembler.push(fragment) or decoded
        self.assertIsNotNone(decoded)

        late = None
        for fragment in stale[1:]:
            late = reassembler.push(fragment) or late
        self.assertIsNone(late)


if __name__ == "__main__":
    unittest.main()
