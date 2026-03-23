"""Visualizer — pygame + moderngl orchestrator for the 3D Luminous Flow engine.

Blend philosophy  (unchanged from 2-D version)
────────────────
  · Activity — smoothed fiducial world-space velocity.
  · Mood     — four incommensurable sinusoidal oscillators (20/35/53/83 min).
  blend_weight = slow EMA of (activity × 0.8 + mood × 0.2) ∈ [0, 1]

3-D additions
─────────────
  · Fiducials are kept in normalised world space (X:±1, Y:±0.5625, Z:0–2).
  · Particles move in 3D; perspective projection makes near ones larger/brighter.
  · Interference uses actual fiducial Z → spherical waves with depth-correct hue.
  · Aurora ribbons and halos are screen-space overlays rendered via GPU lines.

Controls  (unchanged)
────────
  M — toggle CONSTELLATION   G — toggle overlays
  F — toggle fullscreen       ESC/Q — quit
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import moderngl

from tracker.base import Frame
from visual_engine import (
    FlowField3D, ParticleSystem3D, Renderer3D,
    GRID_W, GRID_H, N_PARTICLES,
    WX, WY, WZ, TRAIL_DECAY,
    world_to_screen, _hue_to_rgb_f,
)
from audio import AudioProcessor, AudioFeatures
from image_overlay import ImageOverlay

HUE_PERIOD = 120.0

_MOOD_WAVES: List[Tuple[float, float]] = [
    (0.35, 20 * 60),
    (0.25, 35 * 60),
    (0.20, 53 * 60),
    (0.20, 83 * 60),
]


def _mood(t: float) -> float:
    val = sum(w * (math.sin(2 * math.pi * t / p) * 0.5 + 0.5)
              for w, p in _MOOD_WAVES)
    return float(np.clip(val, 0.0, 1.0))


class Visualizer:
    """Immersive 3D party visualizer."""

    def __init__(
        self,
        width: int  = 1280,
        height: int = 720,
        fullscreen: bool = False,
        maximize: bool = False,
        show_overlays: bool = True,
        n_particles: int = N_PARTICLES,
        audio: bool = True,
        audio_device = None,
        audio_loopback: bool = False,
        trail_length: int = 0,
        show_glow: bool = True,
        show_particles: bool = True,
        title: str = "fidart — spryTrack 300 party visuals",
    ) -> None:
        self.width, self.height = width, height
        self._fullscreen    = fullscreen
        self._maximize      = maximize
        self._show_overlays = show_overlays
        self._title         = title
        self._constellation = False

        self._flow_field = FlowField3D(GRID_W, GRID_H)
        self._particles  = ParticleSystem3D(n_particles)
        self._renderer:  Optional[Renderer3D] = None
        self._ctx:       Optional[moderngl.Context] = None
        self._audio = (AudioProcessor(device=audio_device, loopback=audio_loopback)
                       if audio else None)

        self._screen = None
        self._clock: Optional[pygame.time.Clock] = None
        self._t0 = 0.0
        self._frame_count = 0

        self._prev_world_pos: Dict[int, Tuple[float, float, float]] = {}
        self._vel_smooth:     Dict[int, Tuple[float, float]] = {}
        self._activity:       float = 0.0
        self._blend_weight:   float = 0.0
        self._fid_spin:       Dict[int, float] = {}
        self._spin_counter:   int = 0
        self._awake:          float = 0.0
        self._energy_state:   float = 0.0
        self._prev_beat:      float = 0.0
        self._wave_rings:     list  = []   # [(birth_t, cx_ndc, cy_ndc, r, g, b)]
        self._images = ImageOverlay()

        # Ghost constellation — occasional collage panel
        self._ghost_active  = False
        self._ghost_alpha   = 0.0          # smoothed 0..1
        self._ghost_next_t  = float(np.random.uniform(20.0, 40.0))
        self._ghost_end_t   = 0.0
        self._ghost_rect    = (0, 0, 1, 1)  # (x, y_gl, w, h) in pixels; set on trigger

        # Optional callbacks wired up by main.py
        self.on_person_add:    Optional[callable] = None
        self.on_person_remove: Optional[callable] = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        pygame.init()
        pygame.display.set_caption(self._title)
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self._fullscreen:
            flags |= pygame.FULLSCREEN
        else:
            flags |= pygame.RESIZABLE
        self._screen = pygame.display.set_mode((self.width, self.height), flags)
        self._clock  = pygame.time.Clock()

        if self._maximize:
            import ctypes
            import ctypes.wintypes
            hwnd = pygame.display.get_wm_info()['window']
            ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
            # Read actual client area so the renderer uses the real pixel dimensions
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
            w, h = rect.right - rect.left, rect.bottom - rect.top
            if w > 0 and h > 0:
                self.width, self.height = w, h

        self._ctx      = moderngl.create_context()
        self._renderer = Renderer3D(self._ctx, self.width, self.height)
        self._renderer.init()

        if self._audio:
            self._audio.open()
        self._t0 = time.perf_counter()

        # Pre-warm
        print("[Visualizer] pre-warming …", end=" ", flush=True)
        for step in range(120):
            t = step / 30.0
            self._flow_field.update(t, [])
            self._particles.update(self._flow_field, [], pulse=0.0)
            if step > 40:
                self._particles.upload_to(self._renderer._particle_vbo)
                self._renderer._flow_fbos[self._renderer._flow_idx][0].use()
                self._ctx.blend_func = moderngl.ONE, moderngl.ONE
                self._renderer._progs['particle']['u_brightness'] = 1.0
                self._renderer._particle_vao.render(moderngl.POINTS)
        print("done")

    def close(self) -> None:
        if self._audio:
            self._audio.close()
        pygame.quit()
        print(f"[Visualizer] closed after {self._frame_count} frames")

    def __enter__(self):  self.open();  return self
    def __exit__(self, *_): self.close()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _normalize_fid(self, fid) -> Tuple[float, float, float]:
        wx = float(np.clip( fid.x / 700.0,             -WX,  WX))
        wy = float(np.clip( fid.y / 400.0 * WY,        -WY,  WY))
        wz = float(np.clip((fid.z - 300.0) / 1700.0 * WZ, 0.0, WZ))
        return wx, wy, wz

    # ── per-frame update ───────────────────────────────────────────────────────

    def update(self, frame: Frame) -> bool:
        if not self._handle_events():
            return False

        t   = time.perf_counter() - self._t0
        fps = self._clock.get_fps() if self._clock else 0.0

        # Audio
        af = self._audio.features if self._audio else AudioFeatures()

        # Audio energy state (calm ↔ party)
        af_energy = af.energy if self._audio else 0.0
        if af_energy > self._energy_state:
            self._energy_state = self._energy_state * 0.85 + af_energy * 0.15
        else:
            self._energy_state = self._energy_state * 0.998 + af_energy * 0.002

        # Pulse — only kick/bass drives it; highs and onset ignored
        pulse_sine = (math.sin(t * math.pi * 0.5) * 0.5 + 0.5) ** 2
        sub_bass   = af.sub_bass if self._audio else 0.0
        pulse      = max(pulse_sine * 0.20, sub_bass * 0.8 + af.beat * 0.35)
        hue_speed  = 1.0 + self._energy_state * 0.8   # slow drift; no high/centroid influence
        global_hue = (t * hue_speed / HUE_PERIOD) % 1.0

        # Build fiducial world-space list
        n_fid = max(len(frame.fiducials), 1)
        fid_world: List[Tuple[float, float, float, float]] = []
        fid_grid:  list = []
        fid_vels:  List[Tuple[float, float]] = []

        for fid in frame.fiducials:
            hue_f = (fid.index / n_fid + global_hue) % 1.0
            wx, wy, wz = self._normalize_fid(fid)
            fid_world.append((wx, wy, wz, hue_f))

            idx = fid.index
            if idx in self._prev_world_pos:
                px0, py0, pz0 = self._prev_world_pos[idx]
                raw_vx = wx - px0
                raw_vy = wy - py0
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
            self._prev_world_pos[idx] = (wx, wy, wz)

            evx, evy = self._vel_smooth.get(idx, (0.0, 0.0))
            energy = float(np.clip(math.hypot(evx, evy) / 0.009, 0.0, 1.0))

            if idx not in self._fid_spin:
                self._fid_spin[idx] = 1.0 if self._spin_counter % 2 == 0 else -1.0
                self._spin_counter += 1
            spin = self._fid_spin[idx]

            spd = math.hypot(evx, evy)
            vdx = evx / spd if spd > 5e-5 else 0.0
            vdy = evy / spd if spd > 5e-5 else 0.0

            vortex_str = 0.12 + energy * 1.6
            gx = (wx + WX) / (2 * WX) * (GRID_W - 1)
            gy = (wy + WY) / (2 * WY) * (GRID_H - 1)
            fid_grid.append((gx, gy, vortex_str, vdx, vdy, energy, spin))

        # Beat events → one wave ring on a single random fiducial, only on strong kicks
        if af.beat > 0.82 and self._prev_beat <= 0.82 and fid_world:
            wx, wy, wz, fh = fid_world[int(np.random.randint(0, len(fid_world)))]
            sx, sy = world_to_screen(wx, wy, wz, self.width, self.height)
            cx_ndc = sx / self.width  * 2 - 1
            cy_ndc = 1 - sy / self.height * 2
            r, g, b = _hue_to_rgb_f(fh)
            self._wave_rings.append((t, cx_ndc, cy_ndc, r, g, b))
        self._prev_beat = af.beat
        # Cull expired rings
        self._wave_rings = [(bt, x, y, r, g, b) for bt, x, y, r, g, b in self._wave_rings
                            if t - bt < 2.5]

        # Awake — driven by fiducial count only; no audio floor (no people = dark)
        target_awake = float(np.clip(len(fid_world) / 4.0, 0.0, 1.0))
        if target_awake > self._awake:
            self._awake = self._awake * 0.967 + target_awake * 0.033
        else:
            self._awake = self._awake * 0.996 + target_awake * 0.004

        # Per-fiducial energies
        fid_energies = [
            float(np.clip(math.hypot(*self._vel_smooth.get(fid.index, (0.0, 0.0)))
                          / 0.009, 0.0, 1.0))
            for fid in frame.fiducials
        ]

        # Activity
        if fid_energies:
            target_activity = float(np.clip(np.mean(fid_energies) * 2.5, 0.0, 1.0))
        else:
            target_activity = 0.0
        self._activity = self._activity * 0.92 + target_activity * 0.08

        # Organic blend weight — only kick/bass energy shifts toward INTERFERENCE
        if not self._constellation:
            mood         = _mood(t)
            blend_target = float(np.clip(
                self._activity * 0.3 + (mood - 0.5) * 0.2
                + self._energy_state * 0.45, 0.0, 1.0))
            self._blend_weight = self._blend_weight * 0.988 + blend_target * 0.012
            self._blend_weight = min(1.0, self._blend_weight + af.beat * 0.15)
            # Calm = drift firmly back to FLOW
            if self._energy_state < 0.20:
                self._blend_weight = self._blend_weight * 0.987

        blend_weight = self._blend_weight

        # Brightness — calm resting state, beat/bass lifts it
        a              = self._awake
        e              = self._energy_state
        # All brightness is proportional to presence — fades to near-zero with no people
        bloom_live      = a * 0.28 + e * 0.18 + pulse * a * 0.18 + af.beat * a * 0.15
        particle_bright = 0.05 + a * 0.75 + e * 0.25 + af.beat * a * 0.18

        # Simulation — energy_state scales motion; calm = dreamily slow
        e              = self._energy_state
        speed_scale    = 0.35 + e * 0.65          # 0.35 at silence → long dreamy streaks; 1.0 at party
        trail_decay    = TRAIL_DECAY + (1.0 - e) * 0.050   # calmer = longer persistent trails
        self._flow_field.update(t, fid_grid)
        self._particles.update(self._flow_field, fid_world, pulse=pulse,
                               speed_scale=speed_scale)

        # Energy map
        fid_world_energies = [(wx, wy, wz, e)
                              for (wx, wy, wz, _), e in zip(fid_world, fid_energies)]
        self._renderer.update_energy_map(fid_world_energies)

        # Render
        if self._constellation:
            self._renderer.clear_interf()
            self._renderer.render_constellation(fid_world, self._particles, global_hue)
            self._renderer.composite(0.0, bloom_live, constellation=True)
        else:
            if blend_weight < 0.98:
                self._renderer.splat_particles(self._particles, particle_bright,
                                               trail_decay=trail_decay)
            else:
                self._renderer.decay_flow()

            if blend_weight > 0.02:
                t_interf = t * (1.0 + af.mid * 0.55)
                self._renderer.render_interference(
                    fid_world, t_interf, global_hue, fid_energies, a)
            else:
                self._renderer.decay_interf()

            self._renderer.composite(blend_weight, bloom_live)

        # Ghost constellation collage — only in flow mode, not when user forced constellation
        if not self._constellation:
            if self._ghost_active and t >= self._ghost_end_t:
                self._ghost_active = False
                self._ghost_next_t = t + float(np.random.uniform(20.0, 45.0))
            elif not self._ghost_active and t >= self._ghost_next_t:
                self._ghost_active = True
                dur = float(np.random.uniform(0.8, 1.3))
                self._ghost_end_t  = t + dur
                self._ghost_rect   = self._pick_ghost_rect()
            self._ghost_alpha = 0.55 if self._ghost_active else 0.0
            if self._ghost_alpha > 0.0:
                self._renderer.render_ghost_constellation(
                    fid_world, self._particles, global_hue)
                self._renderer.draw_ghost_constellation(
                    self._ghost_alpha, self._ghost_rect)

        # NYC image ghost — fades in/out over tens of seconds
        self._images.update(t)
        if self._images.ready:
            rgba, iw, ih = self._images.image
            self._renderer.draw_image_overlay(rgba, iw, ih,
                                              self._images.alpha, global_hue)

        # Overlays (rendered additively on top of composite)
        if self._show_overlays:
            self._renderer.draw_fiducial_spheres(fid_world, sub_bass=sub_bass, beat=af.beat)
            self._renderer.draw_velocity_bars(fid_world, fid_vels, self.width, self.height)
            self._renderer.draw_aurora_ribbons(fid_world, t, self.width, self.height)
            self._renderer.draw_halos(fid_world, t, pulse, self.width, self.height,
                                      sub_bass=sub_bass, beat=af.beat)
            self._renderer.draw_wave_rings(self._wave_rings, t, self.width, self.height)

        # HUD
        self._draw_hud(fid_world, fps, t, blend_weight, af)

        pygame.display.flip()
        self._clock.tick(0)
        self._frame_count += 1
        return True

    def _pick_ghost_rect(self) -> tuple:
        """Return a random (x, y_gl, w, h) scissor rect in OpenGL pixel coords.

        OpenGL y=0 is at the bottom, so 'top half of screen' = y_gl = h//2.
        """
        W, H = self.width, self.height
        options = [
            (0,      0,      W // 2, H),          # left half
            (W // 2, 0,      W // 2, H),          # right half
            (0,      H // 2, W,      H // 2),     # top half
            (0,      0,      W,      H // 2),     # bottom half
            (0,      0,      W // 3, H),          # left third
            (W*2//3, 0,      W // 3, H),          # right third
            (W // 4, H // 4, W // 2, H // 2),    # centre quadrant
            (0,      H // 3, W*2//3, H*2//3),    # upper-left two-thirds
        ]
        return options[int(np.random.randint(0, len(options)))]

    def _draw_hud(self, fid_world, fps: float, t: float,
                  blend_weight: float, af: AudioFeatures) -> None:
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

        audio_str = ""
        if self._audio:
            sb_bar = "|" * int(af.sub_bass * 6) + "." * (6 - int(af.sub_bass * 6))
            h_bar  = "|" * int(af.high    * 6) + "." * (6 - int(af.high    * 6))
            e_bar  = "|" * int(self._energy_state * 8) + "." * (8 - int(self._energy_state * 8))
            beat_s = "*BEAT*" if af.beat > 0.3 else "  -  "
            audio_str = f"  kick[{sb_bar}] hi[{h_bar}] nrg[{e_bar}] {beat_s}"

        lines = [
            f"{mode_str}  |  fids: {len(fid_world)}  |  fps: {fps:.0f}{audio_str}",
            "M=constellation   G=overlays   F=fullscreen   Q=quit",
        ]

        # Floating 3-D coordinate labels — randomly on ~25% of fiducials,
        # refreshed every 4 s so they drift from one fiducial to the next.
        fid_labels = []
        if fid_world:
            slot = int(t / 4.0)
            rng  = np.random.default_rng(slot)
            n    = max(1, len(fid_world) // 4)
            idxs = rng.choice(len(fid_world), size=min(n, len(fid_world)),
                              replace=False)
            for i in idxs:
                wx, wy, wz, fh = fid_world[i]
                sx, sy = world_to_screen(wx, wy, wz, self.width, self.height)
                r, g, b = _hue_to_rgb_f(fh)
                # Convert normalised world → mm (inverse of _normalize_fid)
                xmm = wx * 700.0
                ymm = wy / WY * 400.0
                zmm = wz / WZ * 1700.0 + 300.0
                fid_labels.append((sx, sy, xmm, ymm, zmm, r, g, b))

        self._renderer.draw_hud_surface(lines, fid_labels=fid_labels)

    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if ev.key == pygame.K_m:
                    self._constellation = not self._constellation
                    self._renderer.clear_flow()
                    self._renderer.clear_interf()
                    self._renderer.invalidate_voronoi()
                    label = "CONSTELLATION" if self._constellation else "FLOW<->INTERFERENCE"
                    print(f"[Visualizer] mode -> {label}")
                if ev.key == pygame.K_UP and self.on_person_add:
                    self.on_person_add()
                if ev.key == pygame.K_DOWN and self.on_person_remove:
                    self.on_person_remove()
                if ev.key == pygame.K_g:
                    self._show_overlays = not self._show_overlays
                if ev.key == pygame.K_f:
                    self._fullscreen = not self._fullscreen
                    flags = (pygame.OPENGL | pygame.DOUBLEBUF |
                             (pygame.FULLSCREEN if self._fullscreen else pygame.RESIZABLE))
                    self._screen = pygame.display.set_mode(
                        (self.width, self.height), flags)
            if ev.type == pygame.VIDEORESIZE:
                self.width, self.height = ev.w, ev.h
        return True
