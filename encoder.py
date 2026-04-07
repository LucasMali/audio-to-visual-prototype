"""Video encoding module using opencv and ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = 'ffmpeg'


class VideoEncoder:
    def __init__(self, output_path: str, width: int, height: int, fps: int, audio_path: Optional[str] = None):
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_path = audio_path
        self.writer = None
        self.frame_count = 0
        self._init_writer()

    def _init_writer(self) -> None:
        """Initialize video writer."""
        if not HAS_OPENCV:
            raise RuntimeError("opencv-python required. Run: pip install opencv-python")
        
        # Use H.264 codec for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
            isColor=True
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self.output_path}")

    def write_frame(self, rgba_data: bytes) -> None:
        """Write a single RGBA frame."""
        # Convert bytes to numpy array
        frame = np.frombuffer(rgba_data, dtype=np.uint8).reshape((self.height, self.width, 4))
        
        # Convert RGBA to BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        self.writer.write(bgr)
        self.frame_count += 1

    def close(self) -> None:
        """Finalize video and optionally mux with audio."""
        if self.writer is not None:
            self.writer.release()
            print(f"✓ Video written: {self.frame_count} frames to {self.output_path}", flush=True)
        
        if self.audio_path and Path(self.audio_path).exists():
            self._mux_audio()

    def _mux_audio(self) -> None:
        """Mux audio into video file."""
        temp_video = self.output_path.with_stem(f"{self.output_path.stem}_tmp")
        
        try:
            temp_video.write_bytes(self.output_path.read_bytes())
            
            print(f"Muxing audio {self.audio_path}...", flush=True)
            subprocess.run([
                FFMPEG_EXE, '-y',
                '-i', str(temp_video),
                '-i', self.audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                str(self.output_path)
            ], check=True, capture_output=True, timeout=300)
            
            temp_video.unlink()
            print(f"✓ Audio muxed: {self.output_path}", flush=True)
        except Exception as e:
            print(f"✗ Audio muxing failed: {e}", flush=True)
            if temp_video.exists():
                temp_video.unlink()
