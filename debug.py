"""Debug script to test audio loading and GL context."""

import sys
from config import RenderConfig
from audio.analyzer import AudioAnalyzer

print("Loading audio...", file=sys.stderr)
cfg = RenderConfig()
analyzer = AudioAnalyzer(sample_rate=cfg.audio_sample_rate, fft_size=cfg.fft_size, hop_length=cfg.hop_length)

try:
    print(f"Loading french_voice.mp3...", file=sys.stderr)
    audio_frames = list(analyzer.iter_audio_frames("french_voice.mp3", fps=cfg.fps))
    print(f"✓ Audio loaded: {len(audio_frames)} frames", file=sys.stderr)
except Exception as e:
    print(f"✗ Audio loading failed: {e}", file=sys.stderr)
    sys.exit(1)

print("Testing GL context...", file=sys.stderr)
from renderer.context import GLContextFactory

try:
    context = GLContextFactory.create(width=cfg.width, height=cfg.height, force_headless=False, window_mode="hidden")
    print(f"✓ Context created: {type(context).__name__}", file=sys.stderr)
    context.initialize()
    print(f"✓ Context initialized", file=sys.stderr)
    context.release()
    print(f"✓ Context released", file=sys.stderr)
except Exception as e:
    print(f"✗ GL context failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("All tests passed!", file=sys.stderr)
