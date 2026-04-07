"""Shared generic visual rendering orchestration with selectable effects."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from OpenGL import GL
from audio.analyzer import AudioFrameData
from config import RenderConfig


GENERIC_EFFECTS = ("bars", "pulse", "wave")


@dataclass
class GLRenderer:
    context: object
    config: RenderConfig
    effect: str = "bars"
    _fbo: int = 0
    _rbo_color: int = 0
    _frame_buffer: np.ndarray = None
    _fullscreen_quad_vao: int = 0
    _smooth_energy: float = 0.0

    def __post_init__(self) -> None:
        if self.effect not in GENERIC_EFFECTS:
            self.effect = "bars"

    def initialize(self) -> None:
        """Set up FBO, shaders, and geometry."""
        # Create offscreen framebuffer
        self._fbo = GL.glGenFramebuffers(1)
        self._rbo_color = GL.glGenRenderbuffers(1)
        
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._rbo_color)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, self.config.width, self.config.height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self._rbo_color)
        
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("FBO setup failed")
        
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        
        # Pre-allocate frame buffer for glReadPixels
        self._frame_buffer = np.zeros((self.config.height, self.config.width, 4), dtype=np.uint8)
        
        # Create fullscreen quad VAO
        quad_verts = np.array(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32
        )
        quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        ebo = GL.glGenBuffers(1)
        
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad_verts.nbytes, quad_verts, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 8, GL.ctypes.c_void_p(0))
        
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)
        
        self._fullscreen_quad_vao = vao

    def render_frame(self, audio_data: AudioFrameData) -> bytes:
        """Render one frame with the selected generic effect."""
        # Bind offscreen FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glViewport(0, 0, self.config.width, self.config.height)
        
        # Clear
        bg = self.config.background_color
        GL.glClearColor(bg[0], bg[1], bg[2], bg[3])
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        h, w = self.config.height, self.config.width
        self._frame_buffer[:] = [int(bg[0] * 255), int(bg[1] * 255), int(bg[2] * 255), 255]

        bands = np.asarray(audio_data.fft_bands, dtype=np.float32)
        if bands.size == 0:
            bands = np.zeros(32, dtype=np.float32)

        # Use raw amplitude (not normalized energy) so silence = truly zero visuals.
        # boost factor makes moderate audio fill the screen while silence stays flat.
        amplitude = float(audio_data.amplitude)
        gate = float(np.clip(amplitude * 10.0, 0.0, 1.0))

        # Smooth the gate to avoid hard pops between frames.
        self._smooth_energy = 0.75 * self._smooth_energy + 0.25 * gate

        # Bands are normalized per-frame so multiply by gate to restore silence.
        gated_bands = bands * self._smooth_energy

        if self.effect == "bars":
            self._draw_bars(self._frame_buffer, gated_bands, self._smooth_energy)
        elif self.effect == "pulse":
            self._draw_pulse(self._frame_buffer, gated_bands, self._smooth_energy)
        else:
            self._draw_wave(self._frame_buffer, gated_bands, self._smooth_energy, audio_data.time_seconds)
        
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # Flip vertically (OpenGL reads bottom-to-top)
        flipped = np.flipud(self._frame_buffer).copy()
        return bytes(flipped.reshape(-1))

    def _draw_bars(self, frame: np.ndarray, bands: np.ndarray, energy: float) -> None:
        h, w = frame.shape[0], frame.shape[1]
        count = min(32, int(bands.size))
        sampled = bands[:count]
        bar_w = max(6, w // (count * 3))
        gap = max(4, bar_w // 2)
        total_w = count * bar_w + (count - 1) * gap
        x0 = max(0, (w - total_w) // 2)
        center_y = h // 2

        for i, val in enumerate(sampled):
            amp = float(np.clip(val, 0.0, 1.0))
            # No idle offset — bar height is zero when silent
            half_h = int(amp * 0.48 * h)
            if half_h < 1:
                continue
            left = x0 + i * (bar_w + gap)
            right = min(w, left + bar_w)
            top = max(0, center_y - half_h)
            bottom = min(h, center_y + half_h)

            r = int(60 + 180 * amp)
            g = int(160 - 80 * amp)
            b = int(255 - 60 * amp)
            frame[top:bottom, left:right, 0] = r
            frame[top:bottom, left:right, 1] = g
            frame[top:bottom, left:right, 2] = b

    def _draw_pulse(self, frame: np.ndarray, bands: np.ndarray, energy: float) -> None:
        h, w = frame.shape[0], frame.shape[1]
        cx, cy = w // 2, h // 2
        # No idle radius — core is invisible on silence
        base_r = int(min(w, h) * 0.22 * energy)
        if base_r < 2:
            return

        x = np.arange(w, dtype=np.int32)
        y = np.arange(h, dtype=np.int32)
        xx, yy = np.meshgrid(x, y)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

        core = dist2 <= base_r ** 2
        frame[core, 0] = int(90 + 140 * energy)
        frame[core, 1] = int(180 + 60 * energy)
        frame[core, 2] = 255

        ring_strength = float(np.mean(bands[: max(1, bands.size // 3)]))
        for ring in range(1, 5):
            rr = base_r + int(ring * (18 + 20 * ring_strength))
            band = (dist2 >= (rr - 4) ** 2) & (dist2 <= (rr + 4) ** 2)
            alpha = max(0, 1.0 - ring * 0.2) * energy
            frame[band, 0] = int(50 * alpha + 20 * ring)
            frame[band, 1] = int(140 * alpha + 18 * ring)
            frame[band, 2] = int(230 * alpha)

    def _draw_wave(self, frame: np.ndarray, bands: np.ndarray, energy: float, t: float) -> None:
        h, w = frame.shape[0], frame.shape[1]
        mid = h // 2
        xs = np.arange(w)

        interp = np.interp(
            np.linspace(0, max(1, bands.size - 1), w),
            np.arange(bands.size),
            bands,
        )

        # Flat line when silent; amplitude scales with audio energy only
        if energy < 0.005:
            frame[mid, xs, 0] = 60
            frame[mid, xs, 1] = 160
            frame[mid, xs, 2] = 220
            return

        phase = 3.0 * t
        curve = np.sin(np.linspace(0.0, 10.0 * np.pi, w) + phase)
        # Displacement is purely driven by gated band amplitudes — no idle offset
        ys = mid + (interp * 0.40 * h * curve).astype(np.int32)

        thickness = max(1, int(1 + 8 * energy))
        for off in range(-thickness, thickness + 1):
            yo = np.clip(ys + off, 0, h - 1)
            frame[yo, xs, 0] = np.clip(60 + interp * 160, 0, 255).astype(np.uint8)
            frame[yo, xs, 1] = np.clip(160 + 60 * np.sin(phase + off * 0.2), 0, 255).astype(np.uint8)
            frame[yo, xs, 2] = np.clip(210 + interp * 45, 0, 255).astype(np.uint8)
