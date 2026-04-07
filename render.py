"""Entrypoint for cross-platform audio-reactive rendering pipeline.

Pipeline:
1) Load + analyze audio
2) Initialize GL context (EGL on Linux, GLFW on Windows)
3) Render offscreen RGBA frames
4) Write frames to video file with audio sync
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

from config import RenderConfig
from audio.analyzer import AudioAnalyzer, AudioFrameData
from renderer.context import GLContextFactory
from renderer.gl_renderer import GLRenderer, GENERIC_EFFECTS
from encoder import VideoEncoder
from captions.transcribe import transcribe
from captions.renderer import CaptionRenderer, CaptionStyle


def _resolve_audio_input(input_path: str) -> str:
    """Resolve input path from CWD or project input directory."""
    p = Path(input_path)
    if p.exists():
        return str(p)

    input_dir_candidate = Path("input") / input_path
    if input_dir_candidate.exists():
        return str(input_dir_candidate)

    # Try relative to this file's directory in case command runs elsewhere.
    root_candidate = Path(__file__).parent / "input" / input_path
    if root_candidate.exists():
        return str(root_candidate)

    raise FileNotFoundError(f"Audio file not found: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio-reactive visual renderer")
    parser.add_argument("--input", help="Path to audio file (WAV/MP3)")
    parser.add_argument("--output", default="output.mp4", help="Output MP4 path")
    parser.add_argument("--input-dir", help="Directory containing audio files (.mp3/.wav)")
    parser.add_argument("--output-dir", default="output", help="Directory for generated MP4 files")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers for batch mode (use >1 only if GPU/driver supports concurrent contexts)",
    )
    parser.add_argument("--headless", action="store_true", help="Force headless rendering when supported")
    parser.add_argument(
        "--effect",
        choices=GENERIC_EFFECTS,
        default=GENERIC_EFFECTS[0],
        help="Generic visual effect preset",
    )
    parser.add_argument("--list-effects", action="store_true", help="List available generic effects and exit")
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Single-file mode only: render one output per effect preset",
    )
    parser.add_argument(
        "--window-mode",
        choices=["visible", "hidden"],
        default="hidden",
        help="Windows only: context window visibility for debugging",
    )
    args = parser.parse_args()

    if not args.list_effects:
        if not args.input and not args.input_dir:
            parser.error("Provide either --input or --input-dir")
        if args.input and args.input_dir:
            parser.error("Use only one of --input or --input-dir")
        if args.parallel < 1:
            parser.error("--parallel must be >= 1")
    if args.examples and args.input_dir:
        parser.error("--examples is only supported with --input")

    return args


def render_to_video(renderer: GLRenderer, audio_frames: Iterable[AudioFrameData], encoder: VideoEncoder, cfg: RenderConfig, caption_renderer: CaptionRenderer = None) -> None:
    """Render each frame and write to video file."""
    for i, audio_data in enumerate(audio_frames):
        rgba = renderer.render_frame(audio_data)
        
        # Composite caption overlay if available
        if caption_renderer is not None:
            frame = np.frombuffer(rgba, dtype=np.uint8).reshape((cfg.height, cfg.width, 4))
            overlay = caption_renderer.render_caption_overlay(audio_data.time_seconds)
            if overlay is not None:
                frame = _alpha_composite(frame, overlay)
            rgba = bytes(frame.flatten())
        
        encoder.write_frame(rgba)
        
        # Progress feedback
        if (i + 1) % 30 == 0:
            print(f"Rendered {i + 1} frames...", flush=True)
    
    encoder.close()


def _alpha_composite(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Composite RGBA overlay onto RGBA base using alpha blending."""
    # Work in float for precision
    b = base.astype(np.float32) / 255.0
    o = overlay.astype(np.float32) / 255.0
    
    oa = o[:, :, 3:4]  # overlay alpha
    ba = b[:, :, 3:4]  # base alpha
    
    # Standard porter-duff source-over
    out_a = oa + ba * (1.0 - oa)
    safe_a = np.where(out_a > 0, out_a, 1.0)
    out_rgb = (o[:, :, :3] * oa + b[:, :, :3] * ba * (1.0 - oa)) / safe_a
    
    result = np.zeros_like(base)
    result[:, :, :3] = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
    result[:, :, 3] = np.clip(out_a[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
    return result


def render_one_file(input_path: str, output_path: str, args: argparse.Namespace) -> str:
    cfg = RenderConfig()
    resolved_input = _resolve_audio_input(input_path)

    # Transcribe audio for captions
    caption_renderer = None
    if cfg.captions_enabled:
        print(f"Transcribing: {resolved_input} (model: {cfg.captions_model})...", flush=True)
        words = transcribe(resolved_input, model_size=cfg.captions_model)
        print(f"  Found {len(words)} words.", flush=True)
        if words:
            style = CaptionStyle(
                font_size=cfg.captions_font_size,
                y_position=cfg.captions_y_position,
                window_size=cfg.captions_window_size,
                glow_radius=cfg.captions_glow_radius,
                text_color=tuple(cfg.captions_text_color),
                highlight_color=tuple(cfg.captions_highlight_color),
                glow_color=tuple(cfg.captions_glow_color),
            )
            caption_renderer = CaptionRenderer(
                width=cfg.width, height=cfg.height,
                words=words, style=style,
            )

    print(f"Loading audio: {resolved_input}", flush=True)
    analyzer = AudioAnalyzer(sample_rate=cfg.audio_sample_rate, fft_size=cfg.fft_size, hop_length=cfg.hop_length)
    audio_frames = analyzer.iter_audio_frames(resolved_input, fps=cfg.fps)

    print(f"Initializing GL context ({cfg.width}x{cfg.height})...", flush=True)
    context = GLContextFactory.create(
        width=cfg.width,
        height=cfg.height,
        force_headless=args.headless,
        window_mode=args.window_mode,
    )

    with context:
        renderer = GLRenderer(context=context, config=cfg, effect=args.effect)
        renderer.initialize()

        encoder = VideoEncoder(
            output_path=output_path,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            audio_path=resolved_input,
        )

        print(f"Rendering to {output_path}...", flush=True)
        render_to_video(renderer, audio_frames, encoder, cfg, caption_renderer)

    return output_path


def _batch_worker(task: tuple[str, str, bool, str, str]) -> str:
    input_path, output_path, headless, window_mode, effect = task
    ns = argparse.Namespace(headless=headless, window_mode=window_mode, effect=effect)
    return render_one_file(input_path, output_path, ns)


def collect_audio_files(input_dir: str) -> list[Path]:
    base = Path(input_dir)
    files = sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in {".mp3", ".wav"}])
    return files


def main() -> None:
    args = parse_args()

    if args.list_effects:
        print("Available effects:", flush=True)
        for name in GENERIC_EFFECTS:
            print(f"- {name}", flush=True)
        return

    if args.input:
        if args.examples:
            out = Path(args.output)
            for effect in GENERIC_EFFECTS:
                ex_args = argparse.Namespace(
                    headless=args.headless,
                    window_mode=args.window_mode,
                    effect=effect,
                )
                out_path = str(out.with_stem(f"{out.stem}_{effect}"))
                print(f"Rendering example effect '{effect}' -> {out_path}", flush=True)
                render_one_file(args.input, out_path, ex_args)
            return
        render_one_file(args.input, args.output, args)
        return

    input_files = collect_audio_files(args.input_dir)
    if not input_files:
        raise RuntimeError(f"No .mp3 or .wav files found in: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    tasks: list[tuple[str, str, bool, str, str]] = []
    for input_file in input_files:
        out_file = Path(args.output_dir) / f"{input_file.stem}.mp4"
        tasks.append((str(input_file), str(out_file), args.headless, args.window_mode, args.effect))

    print(f"Found {len(tasks)} audio files. Overwrite mode enabled.", flush=True)

    if args.parallel == 1:
        for input_path, output_path, headless, window_mode, effect in tasks:
            ns = argparse.Namespace(headless=headless, window_mode=window_mode, effect=effect)
            render_one_file(input_path, output_path, ns)
        return

    print(f"Running batch with {args.parallel} workers.", flush=True)
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(_batch_worker, task) for task in tasks]
        for future in as_completed(futures):
            done_out = future.result()
            print(f"Completed: {done_out}", flush=True)


if __name__ == "__main__":
    main()
