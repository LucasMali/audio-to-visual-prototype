"""Platform-specific OpenGL context creation.

Linux:
- EGL headless pbuffer context (no X11/Wayland)

Windows:
- GLFW context with visible/hidden debug window
- TODO: Add native WGL path if GLFW is unavailable/undesired
"""

from __future__ import annotations

import os
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class BaseGLContext(ABC):
    width: int
    height: int

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def make_current(self) -> None:
        pass

    @abstractmethod
    def swap_buffers(self) -> None:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    def __enter__(self) -> "BaseGLContext":
        self.initialize()
        self.make_current()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


@dataclass
class EGLHeadlessContext(BaseGLContext):
    width: int
    height: int

    def initialize(self) -> None:
        # TODO: Implement EGL display/config/context/pbuffer creation.
        # Suggested libs: PyOpenGL + ctypes bindings for EGL or moderngl/egl backend.
        raise NotImplementedError("EGL headless context setup not implemented yet")

    def make_current(self) -> None:
        # TODO: eglMakeCurrent(display, surface, surface, context)
        raise NotImplementedError

    def swap_buffers(self) -> None:
        # Pbuffer/offscreen path typically does not need buffer swap.
        return

    def release(self) -> None:
        # TODO: Destroy EGL resources in reverse creation order.
        pass


@dataclass
class GLFWContext(BaseGLContext):
    width: int
    height: int
    visible: bool = False
    _window: Optional[object] = None

    def initialize(self) -> None:
        """Initialize GLFW and create offscreen window."""
        try:
            import glfw
            from OpenGL import GL
        except ImportError:
            raise RuntimeError("glfw or PyOpenGL not installed. Run: pip install glfw PyOpenGL")
        
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Configure for offscreen/hidden rendering
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if self.visible else glfw.FALSE)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        self._window = glfw.create_window(self.width, self.height, "AudioViz", None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self._window)
        glfw.swap_interval(0)  # Disable vsync
        
        # Initialize OpenGL state
        GL.glClearColor(0.01, 0.01, 0.02, 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)  # Additive for neon glow

    def make_current(self) -> None:
        """Make this context current."""
        try:
            import glfw
        except ImportError:
            return
        if self._window:
            glfw.make_context_current(self._window)

    def swap_buffers(self) -> None:
        """Swap (typically unused in offscreen pipeline)."""
        try:
            import glfw
        except ImportError:
            return
        if self._window:
            glfw.swap_buffers(self._window)

    def release(self) -> None:
        """Clean up GLFW resources."""
        try:
            import glfw
        except ImportError:
            return
        if self._window:
            glfw.destroy_window(self._window)
        glfw.terminate()


class GLContextFactory:
    """Factory for selecting platform-specific context backend."""

    @staticmethod
    def create(width: int, height: int, force_headless: bool = False, window_mode: str = "hidden") -> BaseGLContext:
        system = platform.system().lower()

        if system == "linux":
            # Linux must be fully headless in production.
            return EGLHeadlessContext(width=width, height=height)

        if system == "windows":
            visible = window_mode == "visible" and not force_headless
            # Windows debug path uses GLFW hidden/visible context window.
            return GLFWContext(width=width, height=height, visible=visible)

        # TODO: Add macOS support if needed.
        raise RuntimeError(f"Unsupported OS for renderer: {system}")
