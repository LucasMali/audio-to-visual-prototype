"""Global render/audio configuration — loads from settings.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _load_settings() -> Dict[str, Any]:
    path = Path(__file__).parent / "settings.yaml"
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get(d: Dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


_S = _load_settings()


@dataclass(frozen=True)
class RenderConfig:
    width: int = _get(_S, "output", "width", default=1080)
    height: int = _get(_S, "output", "height", default=1920)
    fps: int = _get(_S, "output", "fps", default=30)

    audio_sample_rate: int = _get(_S, "audio", "sample_rate", default=44100)
    fft_size: int = _get(_S, "audio", "fft_size", default=2048)
    hop_length: int = _get(_S, "audio", "hop_length", default=512)

    background_color: Tuple[float, ...] = tuple(_get(_S, "output", "background_color", default=[0.01, 0.01, 0.02, 1.0]))

    # Dot
    dot_color: Tuple[float, ...] = tuple(_get(_S, "dot", "color", default=[0.3, 0.8, 1.0]))
    dot_color_bright: Tuple[float, ...] = tuple(_get(_S, "dot", "color_bright", default=[1.0, 1.0, 1.0]))
    dot_radius: float = _get(_S, "dot", "radius", default=0.06)
    dot_glow_size: float = _get(_S, "dot", "glow_size", default=0.12)
    dot_base_brightness: float = _get(_S, "dot", "base_brightness", default=0.5)
    dot_active_brightness: float = _get(_S, "dot", "active_brightness", default=4.0)
    dot_move_speed: float = _get(_S, "dot", "move_speed", default=0.15)
    dot_move_radius: float = _get(_S, "dot", "move_radius", default=0.12)

    # Ripple waves
    ripple_count: float = _get(_S, "ripples", "count", default=8.0)
    ripple_speed: float = _get(_S, "ripples", "speed", default=0.6)
    ripple_width: float = _get(_S, "ripples", "width", default=0.006)
    ripple_base_alpha: float = _get(_S, "ripples", "base_alpha", default=0.04)
    ripple_active_alpha: float = _get(_S, "ripples", "active_alpha", default=0.6)
    ripple_max_radius: float = _get(_S, "ripples", "max_radius", default=0.45)

    # Atmosphere
    atmo_radius: float = _get(_S, "atmosphere", "radius", default=0.5)
    atmo_intensity: float = _get(_S, "atmosphere", "intensity", default=0.15)

    # Color palette
    color_blue: Tuple[float, ...] = tuple(_get(_S, "palette", "blue", default=[0.15, 0.4, 1.0]))
    color_purple: Tuple[float, ...] = tuple(_get(_S, "palette", "purple", default=[0.55, 0.2, 0.9]))
    color_cyan: Tuple[float, ...] = tuple(_get(_S, "palette", "cyan", default=[0.3, 0.85, 1.0]))

    # Particles
    max_particles: int = _get(_S, "particles", "count", default=800)
    particle_color: Tuple[float, ...] = tuple(_get(_S, "particles", "color", default=[0.5, 0.8, 1.0]))
    particle_size_idle: float = _get(_S, "particles", "size_idle", default=3.0)
    particle_size_active: float = _get(_S, "particles", "size_active", default=14.0)
    particle_alpha_idle: float = _get(_S, "particles", "alpha_idle", default=0.6)
    particle_glow_idle: float = _get(_S, "particles", "glow_idle", default=1.0)
    particle_glow_active: float = _get(_S, "particles", "glow_active", default=4.0)
    particle_reactivity: float = _get(_S, "particles", "reactivity", default=0.5)
    particle_spread: float = _get(_S, "particles", "spread", default=0.5)

    # Captions
    captions_enabled: bool = _get(_S, "captions", "enabled", default=True)
    captions_model: str = _get(_S, "captions", "model", default="base")
    captions_font_size: int = _get(_S, "captions", "font_size", default=64)
    captions_y_position: float = _get(_S, "captions", "y_position", default=0.78)
    captions_window_size: int = _get(_S, "captions", "window_size", default=6)
    captions_glow_radius: int = _get(_S, "captions", "glow_radius", default=8)
    captions_text_color: Tuple[int, ...] = tuple(_get(_S, "captions", "text_color", default=[255, 255, 255, 255]))
    captions_highlight_color: Tuple[int, ...] = tuple(_get(_S, "captions", "highlight_color", default=[120, 200, 255, 255]))
    captions_glow_color: Tuple[int, ...] = tuple(_get(_S, "captions", "glow_color", default=[80, 160, 255, 180]))
