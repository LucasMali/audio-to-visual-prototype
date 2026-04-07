"""Stylized caption renderer — animated word-by-word captions with glow effects.

Renders captions onto RGBA frames using Pillow. Words pop in with scale/fade
animations and get a soft glow outline for that premium broadcast look.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from captions.transcribe import WordTiming


def _find_font(name: str, fallback: str = "arial.ttf") -> str:
    """Locate a system font file by name."""
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    path = os.path.join(fonts_dir, name)
    if os.path.exists(path):
        return path
    fb = os.path.join(fonts_dir, fallback)
    if os.path.exists(fb):
        return fb
    return name  # Let Pillow try to resolve it


@dataclass
class CaptionStyle:
    """Visual style for captions."""
    font_name: str = "segoeuib.ttf"
    font_size: int = 64
    text_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    highlight_color: Tuple[int, int, int, int] = (120, 200, 255, 255)
    glow_color: Tuple[int, int, int, int] = (80, 160, 255, 180)
    glow_radius: int = 8
    shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 120)
    shadow_offset: Tuple[int, int] = (3, 3)
    # Position (fraction of screen height from top, 0.0-1.0)
    y_position: float = 0.78
    # Max width as fraction of screen width
    max_width_frac: float = 0.85
    # Animation
    pop_duration: float = 0.12   # seconds for scale-in pop
    fade_in: float = 0.08        # seconds for opacity fade-in
    fade_out: float = 0.25       # seconds for opacity fade-out after last word
    # How many words to show at once (sliding window)
    window_size: int = 6
    # Word spacing
    word_gap: int = 14


@dataclass
class CaptionRenderer:
    """Renders animated captions onto RGBA frame buffers."""
    width: int
    height: int
    words: List[WordTiming]
    style: CaptionStyle = field(default_factory=CaptionStyle)
    _font: Optional[ImageFont.FreeTypeFont] = field(default=None, init=False)
    _font_path: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self._font_path = _find_font(self.style.font_name)
        self._font = ImageFont.truetype(self._font_path, self.style.font_size)

    def render_caption_overlay(self, time_sec: float) -> Optional[np.ndarray]:
        """Render caption overlay for the given timestamp.

        Returns an RGBA numpy array (height, width, 4) or None if no caption visible.
        """
        if not self.words:
            return None

        # Find active words at this timestamp
        active_indices = self._get_visible_indices(time_sec)
        if not active_indices:
            return None

        # Current word being spoken
        current_idx = self._get_current_word_index(time_sec)

        s = self.style
        max_w = int(self.width * s.max_width_frac)

        # Build lines from visible words (word wrap)
        lines = self._layout_lines(active_indices, max_w)
        if not lines:
            return None

        # Calculate total text block height
        line_height = int(self.style.font_size * 1.4)
        total_h = line_height * len(lines)
        y_start = int(self.height * s.y_position) - total_h // 2

        # Create transparent overlay
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

        # Glow layer (blurred text underneath)
        glow_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)

        # Shadow layer
        shadow_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)

        # Main text layer
        text_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)

        for line_idx, line_words in enumerate(lines):
            # Calculate line width for centering
            word_widths = []
            for (w_idx, word) in line_words:
                bbox = self._font.getbbox(word)
                word_widths.append(bbox[2] - bbox[0])
            
            total_line_w = sum(word_widths) + s.word_gap * max(len(line_words) - 1, 0)
            x_cursor = (self.width - total_line_w) // 2
            y = y_start + line_idx * line_height

            for i, (w_idx, word) in enumerate(line_words):
                w_timing = self.words[w_idx]
                ww = word_widths[i]

                # Animation state
                pop_scale, opacity = self._word_animation(w_idx, time_sec)

                # Is this the currently-spoken word?
                is_current = (current_idx is not None and w_idx == current_idx)
                is_spoken = time_sec >= w_timing.start

                # Choose color
                if is_current:
                    color = s.highlight_color
                elif is_spoken:
                    # Slightly dimmer for already-spoken words
                    color = s.text_color
                else:
                    # Upcoming words are dimmer
                    color = (s.text_color[0], s.text_color[1], s.text_color[2], 
                             int(s.text_color[3] * 0.4))

                # Apply opacity from animation
                final_alpha = int(color[3] * opacity)
                final_color = (color[0], color[1], color[2], final_alpha)

                # Scale animation (font size change for pop-in)
                if pop_scale < 0.99:
                    scaled_size = max(8, int(s.font_size * pop_scale))
                    scaled_font = ImageFont.truetype(self._font_path, scaled_size)
                    # Center the scaled word at the same position
                    sbbox = scaled_font.getbbox(word)
                    sw = sbbox[2] - sbbox[0]
                    sx = x_cursor + (ww - sw) // 2
                    sh = sbbox[3] - sbbox[1]
                    sy = y + (self.style.font_size - sh) // 2
                else:
                    scaled_font = self._font
                    sx = x_cursor
                    sy = y

                # Draw glow (for current/spoken words)
                if is_spoken and opacity > 0.3:
                    glow_alpha = int(s.glow_color[3] * opacity * (1.5 if is_current else 0.6))
                    glow_c = (s.glow_color[0], s.glow_color[1], s.glow_color[2], 
                              min(255, glow_alpha))
                    glow_draw.text((sx, sy), word, font=scaled_font, fill=glow_c)

                # Draw shadow
                if opacity > 0.1:
                    sh_alpha = int(s.shadow_color[3] * opacity)
                    shadow_draw.text(
                        (sx + s.shadow_offset[0], sy + s.shadow_offset[1]),
                        word, font=scaled_font,
                        fill=(s.shadow_color[0], s.shadow_color[1], s.shadow_color[2], sh_alpha)
                    )

                # Draw main text
                text_draw.text((sx, sy), word, font=scaled_font, fill=final_color)

                x_cursor += ww + s.word_gap

        # Blur the glow layer
        if s.glow_radius > 0:
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(s.glow_radius))

        # Blur shadow slightly
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(4))

        # Composite: glow → shadow → text
        overlay = Image.alpha_composite(overlay, glow_layer)
        overlay = Image.alpha_composite(overlay, shadow_layer)
        overlay = Image.alpha_composite(overlay, text_layer)

        return np.array(overlay)

    def _get_visible_indices(self, time_sec: float) -> List[int]:
        """Get indices of words that should be visible at this time."""
        s = self.style
        
        # Find the current word being spoken
        current_idx = self._get_current_word_index(time_sec)
        if current_idx is None:
            # Check if we're in a gap — show last spoken words fading out
            last_spoken = None
            for i, w in enumerate(self.words):
                if w.end <= time_sec:
                    last_spoken = i
            if last_spoken is not None and time_sec - self.words[last_spoken].end < s.fade_out:
                # Show the last window during fade-out
                start = max(0, last_spoken - s.window_size + 1)
                return list(range(start, last_spoken + 1))
            return []

        # Show a window of words centered around current
        half = s.window_size // 2
        start = max(0, current_idx - half)
        end = min(len(self.words), start + s.window_size)
        # Adjust start if we hit the end
        start = max(0, end - s.window_size)
        
        return list(range(start, end))

    def _get_current_word_index(self, time_sec: float) -> Optional[int]:
        """Find the word currently being spoken."""
        for i, w in enumerate(self.words):
            if w.start <= time_sec <= w.end:
                return i
        return None

    def _word_animation(self, word_idx: int, time_sec: float) -> Tuple[float, float]:
        """Calculate pop scale and opacity for a word at the given time.
        
        Returns (scale 0-1, opacity 0-1).
        """
        s = self.style
        w = self.words[word_idx]

        # Before the word starts
        if time_sec < w.start:
            # Pre-show: slightly visible and small
            anticipation = max(0.0, 1.0 - (w.start - time_sec) / 0.5)
            return 0.85 + anticipation * 0.15, anticipation * 0.35

        # Pop-in animation
        t_since_start = time_sec - w.start
        if t_since_start < s.pop_duration:
            progress = t_since_start / s.pop_duration
            # Ease-out cubic for snappy pop
            ease = 1.0 - (1.0 - progress) ** 3
            scale = 0.7 + ease * 0.3
            # Slight overshoot for bounce feel
            if progress > 0.5:
                overshoot = math.sin((progress - 0.5) * math.pi) * 0.05
                scale += overshoot
            return min(scale, 1.05), min(ease * 1.2, 1.0)

        # Fade-in (solidify opacity)
        if t_since_start < s.pop_duration + s.fade_in:
            return 1.0, 1.0

        # During word
        if time_sec <= w.end:
            return 1.0, 1.0

        # After word ends — gradual fade
        t_after = time_sec - w.end
        # Don't fade while still in the visible window
        if t_after < s.fade_out * 2:
            fade = max(0.0, 1.0 - t_after / (s.fade_out * 3))
            return 1.0, max(fade, 0.5)
        
        fade = max(0.0, 1.0 - t_after / (s.fade_out * 5))
        return 1.0, max(fade, 0.3)

    def _layout_lines(self, indices: List[int], max_width: int) -> List[List[Tuple[int, str]]]:
        """Word-wrap visible words into lines that fit within max_width."""
        s = self.style
        lines: List[List[Tuple[int, str]]] = []
        current_line: List[Tuple[int, str]] = []
        current_w = 0

        for idx in indices:
            word = self.words[idx].word
            bbox = self._font.getbbox(word)
            ww = bbox[2] - bbox[0]
            
            needed = ww + (s.word_gap if current_line else 0)
            if current_line and current_w + needed > max_width:
                lines.append(current_line)
                current_line = []
                current_w = 0

            current_line.append((idx, word))
            current_w += ww + (s.word_gap if len(current_line) > 1 else 0)

        if current_line:
            lines.append(current_line)

        return lines
