# Audio Visualizer

> **Prototype** — This codebase is functional but not production-ready and needs to be refactored before use in any larger system.

An audio-reactive video generator that renders GPU-accelerated visuals (rings, particles, glowing dot) synchronized to an audio file, then encodes the result as an MP4.

## How It Works

1. Audio is analyzed using `librosa` (FFT, hop frames at target FPS)
2. Each frame is rendered offscreen via OpenGL (PyOpenGL + GLFW)
3. Frames are encoded to MP4 using OpenCV + FFmpeg
4. Optionally, captions are transcribed via `faster-whisper` and burned in

## Requirements

- Python 3.10+
- FFmpeg installed and on your `PATH` (or installed via `imageio-ffmpeg`)
- A GPU with OpenGL support (for GLFW context on Windows)

Install Python dependencies:

```bash
pip install -r requirements.txt
pip install opencv-python imageio-ffmpeg faster-whisper
```

## Usage

### Single file

```bash
python render.py --input path/to/audio.mp3 --output output.mp4
```

### Batch mode (directory of audio files)

```bash
python render.py --input-dir audio/ --output-dir output/
```

### Parallel batch processing

```bash
python render.py --input-dir audio/ --output-dir output/ --parallel 4
```

> Only use `--parallel > 1` if your GPU driver supports concurrent OpenGL contexts.

### Options

| Flag | Default | Description |
|---|---|---|
| `--input` | — | Path to a single `.mp3` or `.wav` file |
| `--output` | `output.mp4` | Output MP4 path |
| `--input-dir` | — | Directory of audio files for batch mode |
| `--output-dir` | `output/` | Output directory for batch mode |
| `--parallel` | `1` | Number of parallel render workers |
| `--headless` | off | Force headless rendering (Linux/EGL) |
| `--window-mode` | `hidden` | `visible` or `hidden` GLFW window (Windows, useful for debugging) |

## Configuration

All visual parameters are in `settings.yaml`. Edit and re-run — no code changes needed.

Key sections:

- `output` — resolution (default 1080×1920), FPS, background color
- `audio` — sample rate, FFT size, hop length
- `dot` — the central "speaker" dot color, size, glow, and movement
- `ripples` — expanding ring count, speed, width, and alpha
- `atmosphere` — volumetric haze around the dot
- `palette` — blue/purple/cyan color values
- `particles` — count, size, glow reactivity

## Debugging

To verify audio loading and GL context without rendering a full video:

```bash
python debug.py
```

## Project Structure

```
render.py          # Entry point & pipeline orchestration
config.py          # Loads settings.yaml into RenderConfig dataclass
encoder.py         # OpenCV + FFmpeg video encoding
settings.yaml      # All visual/audio tuning parameters
audio/
  analyzer.py      # librosa-based FFT analysis, frame iteration
captions/
  transcribe.py    # faster-whisper speech-to-text with word timestamps
  renderer.py      # Burns captions onto frames
renderer/
  context.py       # GL context factory (GLFW on Windows, EGL on Linux)
  gl_renderer.py   # Offscreen FBO rendering orchestration
  ring.py          # Ring/ripple visualizer
  particles.py     # Particle system
shaders/           # GLSL vertex + fragment shaders
```

## License

MIT © 2026 Lucas Maliszewski — see [LICENSE](LICENSE) for details.