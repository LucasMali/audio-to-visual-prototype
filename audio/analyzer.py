"""Audio loading + analysis (amplitude and FFT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np


@dataclass(frozen=True)
class AudioFrameData:
    frame_index: int
    time_seconds: float
    amplitude: float
    fft_bands: np.ndarray
    energy: float


class AudioAnalyzer:
    def __init__(self, sample_rate: int = 44100, fft_size: int = 2048, hop_length: int = 512) -> None:
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length

    def _load_wav(self, wav_path: str) -> np.ndarray:
        """Load WAV/MP3 as mono float32 in [-1, 1]."""
        import subprocess
        import os
        import tempfile
        
        # Get ffmpeg from imageio_ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise RuntimeError("imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")
        
        # Convert to WAV if needed
        input_file = wav_path
        if wav_path.lower().endswith('.mp3'):
            tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            try:
                subprocess.run([
                    ffmpeg_exe, '-y', '-i', wav_path, '-acodec', 'pcm_s16le',
                    '-ar', str(self.sample_rate), tmp_wav
                ], check=True, capture_output=True, timeout=60)
                input_file = tmp_wav
            except Exception as e:
                raise RuntimeError(f"MP3 to WAV conversion failed: {e}")
        
        # Load WAV with scipy
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(input_file)
            
            # Clean up temp file
            if input_file != wav_path:
                try:
                    os.unlink(input_file)
                except:
                    pass
            
            # Normalize to float32 [-1, 1]
            samples = data.astype(np.float32) / 32768.0
            
            # Convert stereo to mono
            if len(samples.shape) > 1:
                samples = np.mean(samples, axis=1)
            
            return samples.astype(np.float32)
        except ImportError:
            raise RuntimeError("scipy not installed. Run: pip install scipy")

    def get_duration(self, wav_path: str) -> float:
        """Return the duration of the audio file in seconds."""
        samples = self._load_wav(wav_path)
        return len(samples) / self.sample_rate

    def iter_audio_frames(self, wav_path: str, fps: int) -> Generator[AudioFrameData, None, None]:
        samples = self._load_wav(wav_path)
        samples_per_frame = max(1, int(self.sample_rate / fps))
        total_frames = int(np.ceil(len(samples) / samples_per_frame))

        for idx in range(total_frames):
            start = idx * samples_per_frame
            end = min(len(samples), start + samples_per_frame)
            frame = samples[start:end]
            if frame.size == 0:
                break

            amplitude = float(np.mean(np.abs(frame)))

            # Zero-pad to fft_size for stable spectrum length.
            window = np.zeros(self.fft_size, dtype=np.float32)
            n = min(frame.size, self.fft_size)
            window[:n] = frame[:n]
            spectrum = np.fft.rfft(window)
            mag = np.abs(spectrum).astype(np.float32)

            # Compress into coarse bands for visuals.
            band_count = 32
            bands = self._compress_bands(mag, band_count=band_count)
            energy = float(np.mean(bands))

            yield AudioFrameData(
                frame_index=idx,
                time_seconds=idx / fps,
                amplitude=amplitude,
                fft_bands=bands,
                energy=energy,
            )

    @staticmethod
    def _compress_bands(magnitude: np.ndarray, band_count: int) -> np.ndarray:
        if magnitude.size == 0:
            return np.zeros(band_count, dtype=np.float32)

        # TODO: Replace with perceptual/log-frequency binning.
        chunks = np.array_split(magnitude, band_count)
        bands = np.array([float(np.mean(c)) if c.size else 0.0 for c in chunks], dtype=np.float32)

        # Normalize safely.
        max_val = float(np.max(bands)) if bands.size else 1.0
        if max_val > 1e-8:
            bands /= max_val
        return bands
