"""I/O adapters for external systems (video, serial, UI, etc.)."""

from adapters.video_source import create_capture

__all__ = ["create_capture"]
