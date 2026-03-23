"""image_overlay.py — Faint NYC photo collage for fidart.

Drop JPEG/PNG images into  assets/nyc/  and they will slowly materialize
and dissolve over the course of the show — very faint, like a memory.

Timing philosophy (multi-hour installation):
  · An image appears at most every 90–180 s — long gaps of pure colour.
  · Fade-in:  12 s   Hold: 30–50 s   Fade-out: 12 s
  · Max opacity: 0.07  (barely there — atmosphere, not distraction).
  · Images are shown in random order, never repeating until all are shown.

Requires Pillow:  pip install Pillow
"""
from __future__ import annotations

import os
import random
import time
from typing import Optional, Tuple

import numpy as np

try:
    from PIL import Image as PILImage
    _PIL = True
except ImportError:
    _PIL = False

# ── tunable timings ────────────────────────────────────────────────────────────
FADE_IN_SEC   = 12.0
HOLD_SEC_MIN  = 30.0
HOLD_SEC_MAX  = 50.0
FADE_OUT_SEC  = 12.0
GAP_SEC_MIN   = 90.0     # quiet gap between images
GAP_SEC_MAX   = 180.0
MAX_ALPHA     = 0.07      # opacity at peak — just a ghost


class ImageOverlay:
    """Loads images from a folder and cycles through them with slow fade."""

    def __init__(self, image_dir: str = "assets/nyc",
                 target_w: int = 1280, target_h: int = 720) -> None:
        self._dir     = image_dir
        self._tw      = target_w
        self._th      = target_h
        self._paths:  list = []
        self._queue:  list = []   # shuffled playback order
        self._alpha   = 0.0
        self._phase   = "gap"     # gap | fade_in | hold | fade_out
        self._phase_t = 0.0
        self._hold_dur = HOLD_SEC_MIN
        self._gap_dur  = GAP_SEC_MIN
        self._current: Optional[Tuple[bytes, int, int]] = None  # (rgba, w, h)
        self._loaded  = False

        if not _PIL:
            print("[ImageOverlay] Pillow not installed — images disabled.\n"
                  "               Install with:  pip install Pillow")
            return

        self._scan()
        if self._paths:
            print(f"[ImageOverlay] found {len(self._paths)} image(s) in {image_dir!r}")
        else:
            print(f"[ImageOverlay] no images found in {image_dir!r}  "
                  "(create the folder and drop JPG/PNG files in)")

    # ── internals ──────────────────────────────────────────────────────────────

    def _scan(self) -> None:
        if not os.path.isdir(self._dir):
            return
        exts = {'.jpg', '.jpeg', '.png', '.webp'}
        self._paths = [
            os.path.join(self._dir, f)
            for f in os.listdir(self._dir)
            if os.path.splitext(f)[1].lower() in exts
        ]

    def _next_path(self) -> Optional[str]:
        if not self._paths:
            return None
        if not self._queue:
            self._queue = self._paths[:]
            random.shuffle(self._queue)
        return self._queue.pop()

    def _load(self, path: str) -> Optional[Tuple[bytes, int, int]]:
        try:
            img = PILImage.open(path).convert("RGBA")
            # Scale to fill target (crop edges rather than letterbox)
            src_ratio = img.width / img.height
            tgt_ratio = self._tw / self._th
            if src_ratio > tgt_ratio:
                # image is wider — fit height, crop sides
                new_h = self._th
                new_w = int(img.width * self._th / img.height)
            else:
                # image is taller — fit width, crop top/bottom
                new_w = self._tw
                new_h = int(img.height * self._tw / img.width)
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
            ox = (new_w - self._tw) // 2
            oy = (new_h - self._th) // 2
            img = img.crop((ox, oy, ox + self._tw, oy + self._th))
            return img.tobytes(), self._tw, self._th
        except Exception as exc:
            print(f"[ImageOverlay] failed to load {path}: {exc}")
            return None

    # ── public API ─────────────────────────────────────────────────────────────

    def update(self, t: float) -> None:
        """Advance the state machine. Call once per frame."""
        if not _PIL or not self._paths:
            return

        age = t - self._phase_t

        if self._phase == "gap":
            if age >= self._gap_dur:
                path = self._next_path()
                if path:
                    loaded = self._load(path)
                    if loaded:
                        self._current = loaded
                        self._phase   = "fade_in"
                        self._phase_t = t
                        self._hold_dur = random.uniform(HOLD_SEC_MIN, HOLD_SEC_MAX)
                        self._gap_dur  = random.uniform(GAP_SEC_MIN,  GAP_SEC_MAX)

        elif self._phase == "fade_in":
            self._alpha = min(1.0, age / FADE_IN_SEC) * MAX_ALPHA
            if age >= FADE_IN_SEC:
                self._phase   = "hold"
                self._phase_t = t

        elif self._phase == "hold":
            self._alpha = MAX_ALPHA
            if age >= self._hold_dur:
                self._phase   = "fade_out"
                self._phase_t = t

        elif self._phase == "fade_out":
            self._alpha = max(0.0, 1.0 - age / FADE_OUT_SEC) * MAX_ALPHA
            if age >= FADE_OUT_SEC:
                self._current = None
                self._alpha   = 0.0
                self._phase   = "gap"
                self._phase_t = t

    @property
    def ready(self) -> bool:
        return self._current is not None and self._alpha > 0.001

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def image(self) -> Optional[Tuple[bytes, int, int]]:
        """Returns (rgba_bytes, width, height) or None."""
        return self._current
