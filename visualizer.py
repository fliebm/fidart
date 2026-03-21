"""Visualizer — pygame orchestrator for the Luminous Flow visual engine.

Blend philosophy
────────────────
Rather than a fixed timer, the FLOW ↔ INTERFERENCE blend weight responds
organically to what's happening on screen:

  · **Activity** — smoothed fiducial screen velocity.  When people move fast,
    interference energy rises; when they're still, the particle flow takes over.

  · **Mood** — four incommensurable sinusoidal oscillators (periods 20 / 35 /
    53 / 83 minutes) that drift independently and sum to a slowly wandering
    background pressure.  Because the periods share no common factor, the
    combination never exactly repeats during a 5-hour event.

  blend_weight = slow EMA of (activity × 0.8 + mood × 0.2)  ∈ [0, 1]
  τ ≈ 2 s at 30 fps → silky smooth, never mechanical.

Horizontal / Vertical elements
───────────────────────────────
Each fiducial's screen velocity feeds update_planes() in the renderer.
X-motion → horizontal sinusoidal stripes near the person's column.
Y-motion → vertical stripes near the person's row.
The structural layer persists ~3 s (decay 0.955/frame) and is composited
on top of the blended FLOW+INTERFERENCE display — visible in both modes,
bridging the transition with spatial structure.

Controls
────────
  M           — toggle CONSTELLATION (pauses organic blend while active)
  G           — toggle fiducial halos + aurora ribbons
  F           — toggle fullscreen
  ESC / Q     — quit
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from tracker.base import Frame
from visual_engine import (
    FlowField, ParticleSystem, LuminousRenderer,
    GRID_W, GRID_H, N_PARTICLES,
)
from audio import AudioProcessor, AudioFeatures

HUE_PERIOD      = 120.0   # seconds for full colour-palette rotation
VORTEX_STRENGTH = 0.55

# Four mood oscillators — incommensurable periods so the combo never repeats
# (weight, period_seconds)
_MOOD_WAVES: List[Tuple[float, float]] = [
    (0.35, 20 * 60),   # 20 min
    (0.25, 35 * 60),   # 35 min
    (0.20, 53 * 60),   # 53 min
    (0.20, 83 * 60),   # 83 min
]


def _mood(t: float) -> float:
    """Returns a slowly wandering value in [0, 1] — never mechanical."""
    val = sum(w * (math.sin(2 * math.pi * t / p) * 0.5 + 0.5)
              for w, p in _MOOD_WAVES)
    # val ∈ [0, sum(weights)] = [0, 1] already since weights sum to 1.0
    return float(np.clip(val, 0.0, 1.0))


class Visualizer:
    """Immersive party visualizer."""

    def __init__(
        self,
        width: int  = 1280,
        height: int = 720,
        fullscreen: bool = False,
        show_overlays: bool = True,
        n_particles: int = N_PARTICLES,
        audio: bool = True,
        # kept for API compatibility
        trail_length: int = 0,
        show_glow: bool = True,
        show_particles: bool = True,
        title: str = "fidart — spryTrack 300 party visuals",
    ) -> None:
        self.width, self.height = width, height
        self._fullscreen    = fullscreen
        self._show_overlays = show_overlays
        self._title         = title
        self._constellation = False   # True = CONSTELLATION mode (M key)

        self._flow_field = FlowField(GRID_W, GRID_H)
        self._particles  = ParticleSystem(n_particles, width, height)
        self._renderer   = LuminousRenderer(width, height)
        self._audio      = AudioProcessor() if audio else None

        self._screen: Optional[pygame.Surface] = None
        self._clock:  Optional[pygame.time.Clock] = None
        self._t0: float = 0.0
        self._frame_count: int = 0

        # Organic blend state
        self._prev_screen_pos: Dict[int, Tuple[float, float]] = {}
        self._vel_smooth: Dict[int, Tuple[float, float]] = {}   # EMA velocities
        self._activity: float = 0.0          # ∈ [0, 1] — mean speed normalised
        self._blend_weight: float = 0.0      # current display blend weight

        # Per-fiducial spin (+1 CCW, -1 CW): assigned once when first seen.
        # Alternating spins on nearby people create complex non-circular patterns.
        self._fid_spin: Dict[int, float] = {}
        self._spin_counter: int = 0

        # Slumber / awake state: 0 = no one present, 1 = crowd fully active
        # Wakes up quickly when people arrive; fades slowly when room empties
        self._awake: float = 0.0

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        pygame.init()
        pygame.display.set_caption(self._title)
        flags = pygame.FULLSCREEN if self._fullscreen else pygame.RESIZABLE
        self._screen = pygame.display.set_mode((self.width, self.height), flags)
        self._clock  = pygame.time.Clock()
        self._renderer.init_surfaces()
        if self._audio:
            self._audio.open()
        self._t0 = time.perf_counter()

        # Pre-warm: develop the flow field so first frame looks good.
        # Use fast single-pixel splat — multi-size is only needed at runtime.
        print("[Visualizer] pre-warming …", end=" ", flush=True)
        for step in range(120):
            t = step / 30.0
            self._flow_field.update(t, [])
            self._particles.update(self._flow_field, [], pulse=0.0)
            if step > 40:
                self._renderer.splat_particles(self._particles, 2.0)
        print("done")

    def close(self) -> None:
        if self._audio:
            self._audio.close()
        pygame.quit()
        print(f"[Visualizer] closed after {self._frame_count} frames")

    def __enter__(self):  self.open();  return self
    def __exit__(self, *_): self.close()

    # ── per-frame update ───────────────────────────────────────────────────────

    def update(self, frame: Frame) -> bool:
        """Feed one tracking frame.  Returns False when the user quits."""
        if not self._handle_events():
            return False

        t   = time.perf_counter() - self._t0
        fps = self._clock.get_fps() if self._clock else 0.0

        # ── audio features ────────────────────────────────────────────────
        af = self._audio.features if self._audio else AudioFeatures()

        # ── global colour + breathing pulse ───────────────────────────────
        # Bass replaces the slow sine pulse — each kick drum = visual burst.
        # Centroid speeds up hue rotation: bright/trebly sound = fast cycling.
        pulse_sine = (math.sin(t * math.pi * 0.5) * 0.5 + 0.5) ** 2
        pulse      = max(pulse_sine * 0.35, af.bass * 0.85 + af.beat * 0.45)
        hue_speed  = 1.0 + af.centroid * 3.5
        global_hue = (t * hue_speed / HUE_PERIOD) % 1.0

        # ── fiducial projection + velocity tracking ────────────────────────
        n_fid = max(len(frame.fiducials), 1)
        fid_screen: List[Tuple[float, float, float]] = []
        fid_grid:   List[Tuple[float, float, float]] = []
        fid_vels:   List[Tuple[float, float]] = []

        for fid in frame.fiducials:
            hue = (fid.index / n_fid + global_hue) % 1.0
            sx, sy = self._project(fid)
            fid_screen.append((sx, sy, hue))

            # Per-fiducial EMA velocity (screen pixels/frame)
            idx = fid.index
            if idx in self._prev_screen_pos:
                px0, py0 = self._prev_screen_pos[idx]
                raw_vx = sx - px0
                raw_vy = sy - py0
                if idx in self._vel_smooth:
                    evx, evy = self._vel_smooth[idx]
                    evx = evx * 0.7 + raw_vx * 0.3
                    evy = evy * 0.7 + raw_vy * 0.3
                else:
                    evx, evy = raw_vx, raw_vy
                self._vel_smooth[idx] = (evx, evy)
                fid_vels.append((evx, evy))
            else:
                fid_vels.append((0.0, 0.0))
            self._prev_screen_pos[idx] = (sx, sy)

            # Per-fiducial energy ∈ [0,1] from smoothed speed.
            evx, evy = self._vel_smooth.get(idx, (0.0, 0.0))
            energy = float(np.clip(math.hypot(evx, evy) / 55.0, 0.0, 1.0))

            # Spin: assign once per new fiducial, alternating CW/CCW
            if idx not in self._fid_spin:
                self._fid_spin[idx] = 1.0 if self._spin_counter % 2 == 0 else -1.0
                self._spin_counter += 1
            spin = self._fid_spin[idx]

            # Normalised velocity direction for directional wake in flow field
            spd = math.hypot(evx, evy)
            vdx = evx / spd if spd > 0.5 else 0.0
            vdy = evy / spd if spd > 0.5 else 0.0

            # Field strength: sitting=0.12, dancing=1.72
            vortex_str = 0.12 + energy * 1.6
            gx = sx / self.width  * (GRID_W - 1)
            gy = sy / self.height * (GRID_H - 1)
            fid_grid.append((gx, gy, vortex_str, vdx, vdy, energy, spin))

        # ── awake / slumber ───────────────────────────────────────────────
        # Presence: 4+ fiducials = fully awake.  Loud music also wakes the
        # visuals — so an empty dance floor with music is still alive.
        target_awake = float(np.clip(len(fid_screen) / 4.0, 0.0, 1.0))
        target_awake = max(target_awake, af.rms * 0.65)
        if target_awake > self._awake:
            self._awake = self._awake * 0.967 + target_awake * 0.033  # fast wake
        else:
            self._awake = self._awake * 0.996 + target_awake * 0.004  # slow sleep

        # ── per-fiducial energies (for renderer) ──────────────────────────
        fid_energies = [
            float(np.clip(math.hypot(*self._vel_smooth.get(fid.index, (0.0, 0.0))) / 55.0,
                          0.0, 1.0))
            for fid in frame.fiducials
        ]

        # ── activity: mean energy across all present fiducials ────────────
        if fid_energies:
            target_activity = float(np.clip(np.mean(fid_energies) * 2.5, 0.0, 1.0))
        else:
            target_activity = 0.0
        self._activity = self._activity * 0.92 + target_activity * 0.08  # τ≈12f

        # ── organic blend weight ──────────────────────────────────────────
        if not self._constellation:
            mood        = _mood(t)
            blend_target = float(np.clip(
                self._activity * 0.8 + (mood - 0.5) * 0.5, 0.0, 1.0))
            # Slow EMA: τ ≈ 2 s at 30 fps
            self._blend_weight = self._blend_weight * 0.985 + blend_target * 0.015
            # Beat kick: each beat nudges the blend sharply toward interference
            # — the display flashes with the drum hit — then drifts back organically
            self._blend_weight = min(1.0, self._blend_weight + af.beat * 0.22)

        blend_weight = self._blend_weight

        # ── simulation (always runs) ───────────────────────────────────────
        self._flow_field.update(t, fid_grid)
        self._particles.update(self._flow_field, fid_screen, pulse=pulse)

        # ── structural plane waves + spatial energy map ───────────────────
        fid_screen_vels = list(zip(fid_screen, fid_vels))
        self._renderer.update_planes(fid_screen_vels, t)
        # Audio-driven global bands (bass breathes, high shimmers)
        self._renderer.update_audio_planes(af.bass, af.high, t)
        # Energy map: (sx, sy, energy) — drives per-pixel brightness modulation
        fid_screen_energies = [(sx, sy, e)
                               for (sx, sy, _), e in zip(fid_screen, fid_energies)]
        self._renderer.update_energy_map(fid_screen_energies)

        # ── render ────────────────────────────────────────────────────────
        # In slumber: dim bloom, very faint particles, slow ambient pulse only.
        # As crowd grows, everything brightens and energises.
        a = self._awake
        bloom_live        = (0.06 + a * 0.40) + pulse * (a * 0.18)
        particle_bright   = 0.3 + a * 1.7      # 0.3 (slumber) → 2.0 (full)

        if self._constellation:
            self._renderer.render_constellation(
                fid_screen, self._particles, global_hue)
            blend_weight = 0.0
        else:
            # Only compute a buffer when it contributes visibly (> 2 %).
            # The non-active buffer still decays so it fades gracefully when
            # it re-enters the blend.
            if blend_weight < 0.98:
                self._renderer.splat_particles(self._particles, particle_bright)
            else:
                self._renderer._buf_flow *= 0.935   # decay only

            if blend_weight > 0.02:
                # Mid frequencies speed up the interference wave propagation
                t_interf = t * (1.0 + af.mid * 0.55)
                self._renderer.render_interference(fid_screen, t_interf, global_hue,
                                                   fid_energies)
            else:
                self._renderer._buf_interf *= 0.72  # decay only

        display = self._renderer.get_display_array(blend_weight, bloom_live)
        pygame.surfarray.blit_array(self._screen, display.transpose(1, 0, 2))

        # ── overlays ──────────────────────────────────────────────────────
        if self._show_overlays:
            self._renderer.draw_aurora_ribbons(self._screen, fid_screen, t)
            self._renderer.draw_halos(self._screen, fid_screen, t, pulse)

        self._draw_hud(fid_screen, fps, t, blend_weight)
        pygame.display.flip()
        self._clock.tick(0)
        self._frame_count += 1
        return True

    # ── helpers ────────────────────────────────────────────────────────────────

    def _project(self, fid) -> Tuple[float, float]:
        focal = 900.0
        z_safe = max(fid.z, 1.0)
        sx = self.width  / 2 + fid.x * focal / z_safe
        sy = self.height / 2 - fid.y * focal / z_safe
        sx = float(np.clip(sx, self.width  * 0.05, self.width  * 0.95))
        sy = float(np.clip(sy, self.height * 0.05, self.height * 0.95))
        return sx, sy

    def _draw_hud(self, fid_screen, fps: float, t: float,
                  blend_weight: float) -> None:
        font = self._renderer._font
        if font is None:
            return
        if self._constellation:
            mode_str = "CONSTELLATION"
        else:
            act_bar   = "|" * int(self._activity * 8) + "." * (8 - int(self._activity * 8))
            awake_bar = "|" * int(self._awake    * 8) + "." * (8 - int(self._awake    * 8))
            pct = int(blend_weight * 100)
            if pct < 5:
                blend_str = "FLOW"
            elif pct > 95:
                blend_str = "INTERFERENCE"
            else:
                blend_str = f"FLOW <{pct:3d}%> INTERF"
            slumber_str = "" if self._awake > 0.85 else f"  zz[{awake_bar}]"
            mode_str = f"{blend_str}  act[{act_bar}]{slumber_str}"
        # Audio meter: show bass/beat when audio is active
        if self._audio and self._audio._active:
            af_hud = self._audio.features
            b_bar  = "|" * int(af_hud.bass * 6) + "." * (6 - int(af_hud.bass * 6))
            h_bar  = "|" * int(af_hud.high * 6) + "." * (6 - int(af_hud.high * 6))
            beat_s = "*BEAT*" if af_hud.beat > 0.3 else "  -  "
            audio_str = f"  audio bass[{b_bar}] hi[{h_bar}] {beat_s}"
        else:
            audio_str = ""
        lines = [
            f"{mode_str}  |  fids: {len(fid_screen)}  |  fps: {fps:.0f}{audio_str}",
            "M=constellation   G=overlays   F=fullscreen   Q=quit",
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, (90, 90, 90))
            self._screen.blit(surf, (12, 10 + i * 20))

    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if ev.key == pygame.K_m:
                    self._constellation = not self._constellation
                    self._renderer._buf_flow[:]   = 0
                    self._renderer._buf_interf[:] = 0
                    self._renderer._voronoi_cache = None
                    label = "CONSTELLATION" if self._constellation else "FLOW↔INTERFERENCE"
                    print(f"[Visualizer] mode → {label}")
                if ev.key == pygame.K_g:
                    self._show_overlays = not self._show_overlays
                if ev.key == pygame.K_f:
                    self._fullscreen = not self._fullscreen
                    flags = pygame.FULLSCREEN if self._fullscreen else pygame.RESIZABLE
                    self._screen = pygame.display.set_mode(
                        (self.width, self.height), flags)
            if ev.type == pygame.VIDEORESIZE:
                self.width, self.height = ev.w, ev.h
        return True
