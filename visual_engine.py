"""visual_engine.py — Luminous Flow visual engine for fidart.

Three full-screen immersive modes driven by fiducial positions:

  FLOW          — 12 000 particles ride an animated sinusoidal vector field.
                  Each fiducial bends the field into a luminous vortex.
                  Additive accumulation buffer + bloom → every pixel glows
                  like a light-painting long-exposure.

  INTERFERENCE  — Circular waves radiate from each fiducial. Constructive /
                  destructive interference paints the whole screen in shimmering
                  moiré colour — like stones dropped in liquid light.

  CONSTELLATION — Voronoi colour zones own the background. Delaunay edges
                  become animated aurora ribbons. Orbital particle halos pulse
                  at each person's position → living neural-network.

Pipeline per frame
──────────────────
1. Update flow field (sinusoidal octaves + fiducial vortices)
2. Update 12 000 particles (numpy vectorised bilinear field sample)
3. Accumulate particles into float32 HDR buffer (additive light painting)
4. Apply bloom  (threshold → downsample → blur → upsample → additive)
5. Display via surfarray (zero-copy)
6. Draw overlays: aurora ribbons, pulsing halos, minimal HUD
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage
import pygame

from tracker.base import Frame

# ── constants ──────────────────────────────────────────────────────────────────
N_PARTICLES    = 6_000
GRID_W, GRID_H = 80, 45          # flow-field resolution
TRAIL_DECAY    = 0.935            # HDR buffer × this per frame  (↑ = longer trails)
PARTICLE_SPD   = 2.8             # max pixels/frame
DRAG           = 0.96
FIELD_GAIN     = 0.28            # acceleration from field sample
MAX_AGE        = 500             # frames before particle respawns

BLOOM_DS       = 4               # bloom downsample factor
BLOOM_SIGMA    = 7.0             # blur kernel (pixels at 1/4 res)
BLOOM_THRESH   = 45.0            # only bloom above this brightness
BLOOM_STR      = 0.55            # bloom additive strength

# Sinusoidal wave superposition for base flow field
# (amplitude, spatial-kx, spatial-ky, temporal-omega, phase)
_WAVES = [
    (1.00,  1.10,  0.70, 0.050, 0.000),
    (0.80, -0.85,  1.30, 0.070, 1.100),
    (0.60,  1.70, -0.95, 0.040, 2.300),
    (0.40, -1.45,  1.80, 0.090, 4.100),
    (0.30,  2.30,  0.55, 0.060, 0.700),
    (0.20, -0.65, -2.10, 0.110, 3.200),
    (0.15,  3.10, -1.45, 0.080, 5.500),
    (0.10,  0.90,  3.00, 0.130, 1.900),
]


# ── colour helpers ─────────────────────────────────────────────────────────────

def _hsv_rgb_vec(h: np.ndarray, s: float, v: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised HSV→RGB.  h,v: shape (N,) ∈ [0,1], s: scalar.
    Returns r, g, b each (N,) ∈ [0, 255] float32.
    Uses advanced indexing (faster than np.select for large N)."""
    h6 = np.mod(h, 1.0) * 6.0
    i  = h6.astype(np.int32) % 6
    f  = (h6 - np.floor(h6)).astype(np.float32)
    p  = (v * (1.0 - s)).astype(np.float32)
    q  = (v * (1.0 - s * f)).astype(np.float32)
    t_ = (v * (1.0 - s * (1.0 - f))).astype(np.float32)
    vv = v.astype(np.float32) if hasattr(v, 'astype') else np.full_like(p, v)
    # Stack each channel's 6 possible values, then pick by sector index
    arange = np.arange(len(i))
    r = np.stack([vv,  q,  p,  p, t_, vv])[i, arange]
    g = np.stack([t_, vv, vv,  q,  p,  p])[i, arange]
    b = np.stack([ p,  p, t_, vv, vv,  q])[i, arange]
    return r * 255.0, g * 255.0, b * 255.0


def _hue_to_rgb(h: float) -> Tuple[int, int, int]:
    """Scalar HSV→RGB (s=0.90, v=1.0)."""
    s, v = 0.90, 1.0
    h6 = (h % 1.0) * 6.0
    i  = int(h6) % 6
    f  = h6 - int(h6)
    p, q, t_ = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
    rgb = [(v,t_,p),(q,v,p),(p,v,t_),(p,q,v),(t_,p,v),(v,p,q)][i]
    return (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


# ── Flow Field ─────────────────────────────────────────────────────────────────

class FlowField:
    """Animated 2-D vector field: sinusoidal octaves + fiducial vortices.

    The base field uses 8 overlapping sine waves with different spatial
    frequencies and temporal rates — coherent, continuously evolving, never
    repeating.  Each fiducial adds a counter-clockwise vortex + inward spiral.
    """

    def __init__(self, grid_w: int = GRID_W, grid_h: int = GRID_H) -> None:
        self.W, self.H = grid_w, grid_h
        # Spatial coordinates in [0, 2π] — used by sinusoidal waves
        x = np.linspace(0, 2*math.pi, grid_w, dtype=np.float32)
        y = np.linspace(0, 2*math.pi, grid_h, dtype=np.float32)
        self._GX, self._GY = np.meshgrid(x, y)   # (H, W)
        # Grid-index coordinates — used by vortex math
        xi = np.arange(grid_w, dtype=np.float32)
        yi = np.arange(grid_h, dtype=np.float32)
        self._IX, self._IY = np.meshgrid(xi, yi)  # (H, W)
        self.field = np.zeros((grid_h, grid_w, 2), dtype=np.float32)

    def update(self, t: float, fiducials_grid: List[Tuple]) -> None:
        """Recompute field.

        fiducials_grid entries: (gx, gy, strength)
                             or (gx, gy, strength, vdx, vdy, energy, spin)
        vdx/vdy  — normalised velocity direction of the fiducial
        energy   — ∈[0,1], 0=sitting, 1=dancing
        spin     — +1 CCW vortex, -1 CW vortex (alternate per person)

        Behaviour blends with energy:
          low  → classic rotating vortex  (orbital, circular)
          high → directional wake         (particles stream along movement)
        """
        angle = np.zeros((self.H, self.W), dtype=np.float32)
        for amp, kx, ky, omega, phase in _WAVES:
            angle += amp * np.sin(kx*self._GX + ky*self._GY + omega*t + phase)

        self.field[:, :, 0] = np.cos(angle)
        self.field[:, :, 1] = np.sin(angle)

        for entry in fiducials_grid:
            gx, gy, strength = entry[0], entry[1], entry[2]
            vdx  = entry[3] if len(entry) > 3 else 0.0
            vdy  = entry[4] if len(entry) > 3 else 0.0
            energy = entry[5] if len(entry) > 3 else 0.0
            spin = entry[6] if len(entry) > 3 else 1.0

            dx = self._IX - gx
            dy = self._IY - gy
            dist2 = dx*dx + dy*dy + 0.15

            # Vortex: strong when still, fades as energy rises.
            # Spin ±1 → alternate CW/CCW so nearby people interact richly.
            vortex_mix = max(0.10, 1.0 - energy * 0.85)
            vs = strength * 9.0 * vortex_mix / dist2
            self.field[:, :, 0] -= dy * vs * spin
            self.field[:, :, 1] += dx * vs * spin

            # Directional wake: particles stream in movement direction.
            # Gaussian envelope keeps it local to the person.
            if energy > 0.05:
                local = np.exp(-dist2 * 0.22)
                dir_str = energy * 1.3 * strength
                self.field[:, :, 0] += vdx * dir_str * local * 7.0
                self.field[:, :, 1] += vdy * dir_str * local * 7.0

            # Inward spiral — reduced for high-energy (dancer sweeps outward)
            ps = strength * 2.5 * (1.0 - energy * 0.6) / (dist2 + 1.5)
            self.field[:, :, 0] -= dx * ps
            self.field[:, :, 1] -= dy * ps

        # Re-normalise so all vectors are unit length
        mag = np.hypot(self.field[:, :, 0], self.field[:, :, 1]) + 1e-6
        self.field[:, :, 0] /= mag
        self.field[:, :, 1] /= mag


# ── Particle System ────────────────────────────────────────────────────────────

class ParticleSystem:
    """12 000 particles as parallel numpy arrays — fully vectorised update.

    Colour model: each particle slowly drifts its hue toward the nearest
    fiducial's hue.  Speed sets brightness.  Together this creates organic
    colour-zone boundaries that bleed into each other as people move.
    """

    def __init__(self, n: int = N_PARTICLES, W: int = 1280, H: int = 720) -> None:
        self.n, self.W, self.H = n, W, H
        rng = np.random.default_rng(1)
        self.px  = rng.uniform(0, W, n).astype(np.float32)
        self.py  = rng.uniform(0, H, n).astype(np.float32)
        self.pvx = np.zeros(n, np.float32)
        self.pvy = np.zeros(n, np.float32)
        self.hue  = rng.uniform(0, 1, n).astype(np.float32)
        self.bri  = rng.uniform(0.5, 1.0, n).astype(np.float32)
        self.age  = rng.integers(0, MAX_AGE, n).astype(np.int32)
        # Visual size: 1=tiny(50%), 2=medium(35%), 3=large(15%)
        self.scale = rng.choice([1, 1, 1, 2, 2, 3], n).astype(np.int8)
        self._rng = np.random.default_rng()

    def update(self, field: FlowField,
               fid_screen: List[Tuple[float, float, float]],  # (sx, sy, hue)
               pulse: float) -> None:
        # ── bilinear sample of flow field ──────────────────────────────────
        gx = np.clip(self.px / self.W * (field.W - 1), 0, field.W - 1.001)
        gy = np.clip(self.py / self.H * (field.H - 1), 0, field.H - 1.001)
        gxi = gx.astype(np.int32);  tx = (gx - gxi).astype(np.float32)
        gyi = gy.astype(np.int32);  ty = (gy - gyi).astype(np.float32)
        gxi1 = np.minimum(gxi + 1, field.W - 1)
        gyi1 = np.minimum(gyi + 1, field.H - 1)

        for dim, pv in enumerate([self.pvx, self.pvy]):
            f00 = field.field[gyi,  gxi,  dim]
            f10 = field.field[gyi,  gxi1, dim]
            f01 = field.field[gyi1, gxi,  dim]
            f11 = field.field[gyi1, gxi1, dim]
            sampled = (f00*(1-tx)*(1-ty) + f10*tx*(1-ty) +
                       f01*(1-tx)*ty     + f11*tx*ty)
            accel = FIELD_GAIN * (1.0 + pulse * 0.4)
            if dim == 0:
                self.pvx = self.pvx * DRAG + sampled * accel
            else:
                self.pvy = self.pvy * DRAG + sampled * accel

        # Speed clamp
        speed = np.hypot(self.pvx, self.pvy)
        cap   = PARTICLE_SPD * (1.0 + pulse * 0.25)
        over  = speed > cap
        self.pvx[over] = self.pvx[over] / speed[over] * cap
        self.pvy[over] = self.pvy[over] / speed[over] * cap

        self.px  += self.pvx
        self.py  += self.pvy
        self.age += 1

        # ── colour: drift hue toward nearest fiducial ──────────────────────
        if fid_screen:
            fpos  = np.array([[sx, sy] for sx, sy, _ in fid_screen], np.float32)
            fhues = np.array([fh for _, _, fh in fid_screen], np.float32)
            diff  = np.stack([self.px, self.py], 1)[:, None, :] - fpos[None, :, :]
            dist2 = diff[:, :, 0]**2 + diff[:, :, 1]**2
            near  = np.argmin(dist2, axis=1)
            target = fhues[near]
            delta  = (target - self.hue + 0.5) % 1.0 - 0.5
            self.hue = (self.hue + delta * 0.04) % 1.0

        # Brightness from speed + pulse
        self.bri = np.clip(0.55 + speed / PARTICLE_SPD * 0.45 + pulse * 0.15, 0.2, 1.0)

        # ── respawn dead particles ──────────────────────────────────────────
        dead = ((self.px < 0) | (self.px >= self.W) |
                (self.py < 0) | (self.py >= self.H) |
                (self.age > MAX_AGE))
        nd = int(dead.sum())
        if nd:
            if fid_screen:
                # 70 % respawn near a random fiducial; 30 % random anywhere.
                # This concentrates particles in active areas over time.
                n_near = int(nd * 0.70)
                n_rand = nd - n_near
                fpos = np.array([[sx, sy] for sx, sy, _ in fid_screen], np.float32)
                choice = self._rng.integers(0, len(fpos), n_near)
                spread_x = self.W * 0.18
                spread_y = self.H * 0.18
                nx = np.clip(fpos[choice, 0] +
                             self._rng.normal(0, spread_x, n_near), 0, self.W-1)
                ny = np.clip(fpos[choice, 1] +
                             self._rng.normal(0, spread_y, n_near), 0, self.H-1)
                dead_idx = np.where(dead)[0]
                self.px[dead_idx[:n_near]] = nx.astype(np.float32)
                self.py[dead_idx[:n_near]] = ny.astype(np.float32)
                if n_rand:
                    self.px[dead_idx[n_near:]] = self._rng.uniform(0, self.W, n_rand)
                    self.py[dead_idx[n_near:]] = self._rng.uniform(0, self.H, n_rand)
            else:
                self.px[dead] = self._rng.uniform(0, self.W, nd).astype(np.float32)
                self.py[dead] = self._rng.uniform(0, self.H, nd).astype(np.float32)
            self.pvx[dead] = self.pvy[dead] = 0
            self.age[dead] = 0
            # Reassign random scale on respawn for variety
            self.scale[dead] = self._rng.choice([1, 1, 1, 2, 2, 3], nd).astype(np.int8)


# ── Luminous Renderer ──────────────────────────────────────────────────────────

class LuminousRenderer:
    """Full-screen renderer with three modes, bloom, and overlay drawing.

    The accumulation buffer (`_buf`, float32 HDR) stores light energy that
    decays each frame — identical to how long-exposure photography works.
    Additive particle splatting via `np.bincount` fills it efficiently.
    Bloom extracts bright regions, blurs at 1/BLOOM_DS resolution, and
    composites additively — turning every bright point into a real glow.
    """

    MODES = ["FLOW", "INTERFERENCE", "CONSTELLATION"]

    def __init__(self, W: int = 1280, H: int = 720) -> None:
        self.W, self.H = W, H
        self.mode_idx  = 0
        # Two separate HDR buffers — both always live, blended at display time
        self._buf_flow   = np.zeros((H, W, 3), np.float32)
        self._buf_interf = np.zeros((H, W, 3), np.float32)
        self._blend_work = np.zeros((H, W, 3), np.float32)  # reusable scratch
        # ── bloom cache: recompute every BLOOM_SKIP frames ─────────────────
        self._bloom_skip   = 3
        self._bloom_count  = 0
        self._bloom_cache  = np.zeros((H, W, 3), np.float32)
        self._blend_temp   = np.zeros((H, W, 3), np.float32)  # second scratch
        # ── low-res pixel grid: 1/8 res for speed ──────────────────────────
        lh, lw = H // 8, W // 8
        self._LH, self._LW = lh, lw
        yi, xi  = np.mgrid[0:lh, 0:lw]
        self._PX = (xi * 8).astype(np.float32)  # screen-space x  (lh, lw)
        self._PY = (yi * 8).astype(np.float32)  # screen-space y  (lh, lw)
        # ── structural plane-wave buffer (H/V bands, velocity-driven) ──────
        # Stored at 1/8 resolution; slow decay so bands linger 3-5 seconds
        self._buf_struct  = np.zeros((lh, lw, 3), np.float32)
        self._struct_decay = 0.955   # ~3 s half-life at 30 fps
        # ── spatial energy map (per-pixel presence/energy of people) ────────
        # Gaussian sum weighted by each fiducial's velocity energy.
        # Drives local brightness amplification: hot near dancers, dim elsewhere.
        self._energy_map  = np.zeros((lh, lw), np.float32)
        # ── constellation Voronoi cache ─────────────────────────────────────
        self._voronoi_cache: Optional[np.ndarray] = None  # full-size (H, W, 3)
        self._voronoi_fids:  Optional[np.ndarray] = None  # last fiducial positions
        self._ribbon_surf: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None

    def init_surfaces(self) -> None:
        """Call once after pygame.init()."""
        self._ribbon_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self._ribbon_surf.fill((0, 0, 0, 0))
        self._font = pygame.font.SysFont("consolas", 15)

    @property
    def mode(self) -> str:
        return self.MODES[self.mode_idx]

    def next_mode(self) -> None:
        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
        self._buf_flow[:]   = 0
        self._buf_interf[:] = 0
        self._voronoi_cache = None

    # ── FLOW mode ──────────────────────────────────────────────────────────────

    def splat_particles(self, particles: ParticleSystem,
                        brightness_scale: float = 3.0) -> None:
        """Additively accumulate particle light into the flow HDR buffer."""
        self._buf_flow *= TRAIL_DECAY

        px  = np.clip(particles.px.astype(np.int32), 0, self.W - 1)
        py  = np.clip(particles.py.astype(np.int32), 0, self.H - 1)
        idx = py * self.W + px
        r, g, b = _hsv_rgb_vec(particles.hue, 0.88, particles.bri)
        NP = self.W * self.H
        s  = brightness_scale
        self._buf_flow[:, :, 0] += np.bincount(idx, r*s, NP).reshape(self.H, self.W)
        self._buf_flow[:, :, 1] += np.bincount(idx, g*s, NP).reshape(self.H, self.W)
        self._buf_flow[:, :, 2] += np.bincount(idx, b*s, NP).reshape(self.H, self.W)

    # ── INTERFERENCE mode ──────────────────────────────────────────────────────

    def update_energy_map(self,
                          fid_screen_energies: List[Tuple[float, float, float]]
                          ) -> None:
        """Recompute the spatial energy map from current fiducial states.

        fid_screen_energies: list of (sx, sy, energy) where energy ∈ [0,1].
        energy = 0 → sitting still, energy = 1 → full-energy dancing.
        The map is a sum of Gaussians; clipped to [0, 1].
        """
        self._energy_map[:] = 0.0
        if not fid_screen_energies:
            return
        # Cap to 12 most energetic fiducials — beyond that, visual gain is zero
        if len(fid_screen_energies) > 12:
            fid_screen_energies = sorted(fid_screen_energies,
                                         key=lambda e: e[2], reverse=True)[:12]
        X, Y = self._PX, self._PY
        sig_x = self.W * 0.14
        sig_y = self.H * 0.14
        for sx, sy, energy in fid_screen_energies:
            if energy < 0.02:
                continue
            dx = (X - sx) / sig_x
            dy = (Y - sy) / sig_y
            self._energy_map += np.exp(-0.5 * (dx*dx + dy*dy)) * float(energy)
        np.clip(self._energy_map, 0.0, 1.0, out=self._energy_map)

    def render_interference(self, fid_screen: List[Tuple[float, float, float]],
                            t: float, global_hue: float,
                            fid_energies: Optional[List[float]] = None) -> None:
        """Full-screen wave interference — writes into the interference HDR buffer.

        fid_energies: optional per-fiducial energy ∈ [0,1].  High-energy
        (dancing) fiducials cast stronger waves; sitting fiducials barely ripple.
        """
        self._buf_interf *= 0.72   # faster fade — pattern redraws itself each frame

        X, Y = self._PX, self._PY   # (LH, LW)

        if fid_screen:
            # Cap fiducials: beyond ~12, interference adds no visual richness
            # but the (LH, LW, F) tensor cost scales linearly with F
            _MAX_INTERF_FIDS = 12
            if len(fid_screen) > _MAX_INTERF_FIDS:
                if fid_energies is not None:
                    top = sorted(range(len(fid_energies)),
                                 key=lambda i: fid_energies[i], reverse=True
                                 )[:_MAX_INTERF_FIDS]
                    fid_screen  = [fid_screen[i]  for i in top]
                    fid_energies = [fid_energies[i] for i in top]
                else:
                    fid_screen = fid_screen[:_MAX_INTERF_FIDS]
            # Vectorised over all fiducials at once: dist shape → (LH, LW, F)
            sx_a = np.array([s[0] for s in fid_screen], np.float32)
            sy_a = np.array([s[1] for s in fid_screen], np.float32)
            dx   = X[:, :, None] - sx_a          # (LH, LW, F)
            dy   = Y[:, :, None] - sy_a
            dist = np.hypot(dx, dy)              # (LH, LW, F)
            raw  = np.sin(dist * 0.028 - t * 2.8) * np.exp(-dist * 0.0007)
            if fid_energies is not None and len(fid_energies) == len(fid_screen):
                w = np.array(fid_energies, np.float32) * 1.4 + 0.25
                w /= w.mean()
                wave = (raw * w).mean(axis=2).astype(np.float32)
            else:
                wave = raw.mean(axis=2).astype(np.float32)
        else:
            d1   = np.hypot(X - self.W*0.35, Y - self.H*0.5)
            d2   = np.hypot(X - self.W*0.65, Y - self.H*0.5)
            wave = (np.sin(d1 * 0.028 - t*2.8) +
                    np.sin(d2 * 0.028 - t*2.4)).astype(np.float32) * 0.5

        hue  = (wave * 0.30 + global_hue) % 1.0
        bri  = np.abs(wave) * 0.85 + 0.12
        r, g, b = _hsv_rgb_vec(hue.ravel(), 0.92, bri.ravel())
        small  = np.stack([r, g, b], -1).reshape(self._LH, self._LW, 3)
        big    = small.repeat(8, 0).repeat(8, 1)[:self.H, :self.W]
        self._buf_interf += big

    # ── CONSTELLATION mode ─────────────────────────────────────────────────────

    def render_constellation(self,
                             fid_screen: List[Tuple[float, float, float]],
                             particles: ParticleSystem,
                             global_hue: float) -> None:
        """Voronoi background + orbital particle halos.

        The Voronoi background is cached and only recomputed when fiducials
        have moved more than a threshold — constellation zones evolve slowly.
        """
        # Constellation uses _buf_flow as its single working buffer
        self._buf_flow *= 0.88

        X, Y = self._PX, self._PY
        lh, lw = self._LH, self._LW

        if fid_screen:
            fpos  = np.array([[sx, sy] for sx, sy, _ in fid_screen], np.float32)
            fhues = np.array([fh for _, _, fh in fid_screen], np.float32)
            recompute = True
            if self._voronoi_fids is not None and self._voronoi_fids.shape == fpos.shape:
                max_move = float(np.max(np.abs(fpos - self._voronoi_fids)))
                recompute = max_move > 8.0
            if recompute:
                pix   = np.stack([X.ravel(), Y.ravel()], 1)
                diff  = pix[:, None, :] - fpos[None, :, :]
                dist2 = diff[:, :, 0]**2 + diff[:, :, 1]**2
                near  = np.argmin(dist2, 1)
                min_d = np.sqrt(dist2[np.arange(len(dist2)), near])
                phue  = fhues[near]
                pbri  = np.clip(0.50 - min_d / (self.W * 0.7), 0.04, 0.42)
                r, g, b = _hsv_rgb_vec(phue, 1.0, pbri.astype(np.float32))
                small  = np.stack([r, g, b], -1).reshape(lh, lw, 3)
                self._voronoi_cache = small.repeat(8, 0).repeat(8, 1)[:self.H, :self.W]
                self._voronoi_fids  = fpos.copy()
        else:
            if self._voronoi_cache is None:
                phue = (X.ravel() / self.W + global_hue) % 1.0
                pbri = np.full(lh * lw, 0.15, np.float32)
                r, g, b = _hsv_rgb_vec(phue, 1.0, pbri)
                small  = np.stack([r, g, b], -1).reshape(lh, lw, 3)
                self._voronoi_cache = small.repeat(8, 0).repeat(8, 1)[:self.H, :self.W]

        self._buf_flow += self._voronoi_cache

        px = np.clip(particles.px.astype(np.int32), 0, self.W - 1)
        py = np.clip(particles.py.astype(np.int32), 0, self.H - 1)
        idx = py * self.W + px
        r2, g2, b2 = _hsv_rgb_vec(particles.hue, 0.7, particles.bri * 0.6)
        NP = self.W * self.H
        self._buf_flow[:, :, 0] += np.bincount(idx, r2*1.8, NP).reshape(self.H, self.W)
        self._buf_flow[:, :, 1] += np.bincount(idx, g2*1.8, NP).reshape(self.H, self.W)
        self._buf_flow[:, :, 2] += np.bincount(idx, b2*1.8, NP).reshape(self.H, self.W)

    # ── Structural plane waves (H/V bands driven by fiducial velocity) ────────

    def update_planes(self,
                      fid_screen_vels: List[Tuple[Tuple[float,float,float],
                                                  Tuple[float,float]]],
                      t: float) -> None:
        """Accumulate velocity-driven horizontal/vertical plane waves.

        fid_screen_vels: list of ((sx, sy, hue), (vx, vy)) per fiducial.
        X-motion  → horizontal stripes that slowly drift up/down.
        Y-motion  → vertical stripes that slowly drift left/right.
        Each stripe field is spatially localised around the person.
        Contributions persist in _buf_struct with slow decay (~3 s half-life).
        """
        self._buf_struct *= self._struct_decay

        X, Y = self._PX, self._PY   # (LH, LW) screen-space

        for (sx, sy, hue), (vx, vy) in fid_screen_vels:
            speed = math.hypot(vx, vy)
            if speed < 1.5:
                continue

            r, g, b_c = _hue_to_rgb(hue)
            col = np.array([r, g, b_c], np.float32)  # [0..255]

            # Spatial Gaussian envelopes (wide — extends well beyond the person)
            sig_x = self.W * 0.28
            sig_y = self.H * 0.28
            env_x = np.exp(-0.5 * ((X - sx) / sig_x) ** 2)   # (LH, LW)
            env_y = np.exp(-0.5 * ((Y - sy) / sig_y) ** 2)

            # Strength scales with speed, capped
            h_str = min(abs(vx) / 45.0, 1.2)   # horizontal bands from X-motion
            v_str = min(abs(vy) / 45.0, 1.2)   # vertical bands from Y-motion

            # Horizontal bands: sin varies in Y → stripes span full width
            # Localised by env_x (centred on person's X column)
            if h_str > 0.04:
                wave_h = np.sin(Y / self.H * 11.0 - t * 2.3 + sx * 0.009)
                bright = np.maximum(wave_h * env_x, 0).astype(np.float32)
                bright *= h_str * 18.0
                self._buf_struct += bright[:, :, None] * col

            # Vertical bands: sin varies in X → stripes span full height
            # Localised by env_y (centred on person's Y row)
            if v_str > 0.04:
                wave_v = np.sin(X / self.W * 11.0 - t * 2.0 + sy * 0.009)
                bright = np.maximum(wave_v * env_y, 0).astype(np.float32)
                bright *= v_str * 18.0
                self._buf_struct += bright[:, :, None] * col

        # Keep buffer in a displayable range so it never blows out
        np.clip(self._buf_struct, 0, 220, out=self._buf_struct)

    def update_audio_planes(self, bass: float, high: float, t: float) -> None:
        """Add global audio-reactive structural waves to _buf_struct.

        Bass  → slow horizontal undulating bands across the full screen.
                Warm orange/amber colour — feels like the room breathing.
        High  → fast thin vertical shimmer lines, cool blue/cyan.
                Responds to cymbals and hi-hats.
        Both  layers accumulate into the same _buf_struct as the
        person-driven bands, so they blend naturally.
        """
        if bass < 0.04 and high < 0.04:
            return

        X, Y = self._PX, self._PY

        if bass > 0.04:
            wave_h = np.sin(Y / self.H * 7.0 - t * 1.6 + bass * 4.0)
            bright = np.maximum(wave_h, 0).astype(np.float32) * bass * 35.0
            self._buf_struct[:, :, 0] += bright * 0.88   # warm orange-red
            self._buf_struct[:, :, 1] += bright * 0.38
            self._buf_struct[:, :, 2] += bright * 0.05

        if high > 0.04:
            wave_v = np.sin(X / self.W * 16.0 - t * 5.5 + high * 7.0)
            bright = np.maximum(wave_v, 0).astype(np.float32) * high * 28.0
            self._buf_struct[:, :, 0] += bright * 0.30   # cool blue-cyan
            self._buf_struct[:, :, 1] += bright * 0.72
            self._buf_struct[:, :, 2] += bright * 0.95

        np.clip(self._buf_struct, 0, 220, out=self._buf_struct)

    # ── Bloom ──────────────────────────────────────────────────────────────────

    def get_display_array(self, blend_weight: float = 0.0,
                          bloom_str: float = BLOOM_STR) -> np.ndarray:
        """Blend the two HDR buffers + structural layer, apply bloom.

        blend_weight: 0.0 = pure FLOW, 1.0 = pure INTERFERENCE, 0…1 = crossfade.
        _buf_struct (plane waves) is always composited on top — it exists at all
        blend weights, bridging the two modes with spatial structure.
        """
        # Weighted blend of the two live buffers into scratch space
        if blend_weight <= 0.0:
            buf = self._buf_flow
        elif blend_weight >= 1.0:
            buf = self._buf_interf
        else:
            # Two-temp lerp: zero allocations, two memory-parallel multiplies
            #   blend_work = buf_flow*(1-w)
            #   blend_temp = buf_interf*w
            #   blend_work += blend_temp
            w = float(blend_weight)
            np.multiply(self._buf_flow,   1.0 - w, out=self._blend_work)
            np.multiply(self._buf_interf, w,        out=self._blend_temp)
            np.add(self._blend_work, self._blend_temp, out=self._blend_work)
            buf = self._blend_work

        np.minimum(buf, 255.0, out=buf)

        # Recompute bloom on schedule
        self._bloom_count += 1
        if self._bloom_count >= self._bloom_skip:
            self._bloom_count = 0
            bright = buf - BLOOM_THRESH
            np.maximum(bright, 0, out=bright)
            small   = bright[::BLOOM_DS, ::BLOOM_DS]
            blurred = scipy.ndimage.uniform_filter(
                small, size=[BLOOM_SIGMA, BLOOM_SIGMA, 1], mode='reflect')
            big = blurred.repeat(BLOOM_DS, 0).repeat(BLOOM_DS, 1)[:self.H, :self.W]
            np.multiply(big, bloom_str, out=big)
            self._bloom_cache = big

        # Upscale structural plane-wave buffer (1/8 → full res) and composite
        struct_big  = self._buf_struct.repeat(8, 0).repeat(8, 1)[:self.H, :self.W]
        struct_str  = 0.10 + blend_weight * 0.12   # subtle structural layer
        result = buf + self._bloom_cache + struct_big * struct_str

        # Spatial energy: gentle multiply so dense crowds don't blow out,
        # plus a small additive glow to lift dark areas near people.
        # Range: 0.60× far from everyone → 1.15× near a dancer.
        energy_full = self._energy_map.repeat(8, 0).repeat(8, 1)[:self.H, :self.W, None]
        result *= (0.60 + energy_full * 0.55)
        result += energy_full * 18.0   # soft additive glow, max +18 units

        np.clip(result, 0, 255, out=result)
        return result.astype(np.uint8)

    # ── Overlay: aurora ribbons ────────────────────────────────────────────────

    def draw_aurora_ribbons(self, screen: pygame.Surface,
                            fid_screen: List[Tuple[float, float, float]],
                            t: float) -> None:
        """Animated bezier ribbons connecting nearby fiducials."""
        if len(fid_screen) < 2 or self._ribbon_surf is None:
            return

        # Determine pairs to connect (Delaunay if ≥3, else all pairs)
        pairs: List[Tuple[int, int]] = []
        try:
            if len(fid_screen) >= 3:
                from scipy.spatial import Delaunay
                pts = np.array([[sx, sy] for sx, sy, _ in fid_screen])
                tri = Delaunay(pts)
                seen: set = set()
                for simp in tri.simplices:
                    for a, b in [(0,1),(1,2),(0,2)]:
                        e = (min(simp[a],simp[b]), max(simp[a],simp[b]))
                        if e not in seen:
                            seen.add(e); pairs.append(e)
            else:
                pairs = [(i,j) for i in range(len(fid_screen))
                         for j in range(i+1, len(fid_screen))]
        except Exception:
            pairs = [(i,j) for i in range(len(fid_screen))
                     for j in range(i+1, len(fid_screen))]

        self._ribbon_surf.fill((0, 0, 0, 0))

        for a, b in pairs:
            sx0, sy0, h0 = fid_screen[a]
            sx1, sy1, h1 = fid_screen[b]
            # Animated midpoint — creates a living wave between people
            seed = sx0*0.013 + sy0*0.019 + sx1*0.011 + sy1*0.017
            mid_x = (sx0+sx1)*0.5 + math.sin(t*0.7 + seed)*70
            mid_y = (sy0+sy1)*0.5 + math.cos(t*0.5 + seed*1.3)*45

            # Draw 3 passes: wide dim, medium, thin bright → depth/glow
            for width, alpha_scale in [(4, 0.30), (2, 0.55), (1, 0.90)]:
                pts_bezier = []
                n = 28
                for step in range(n + 1):
                    tt = step / n
                    bx = (1-tt)**2*sx0 + 2*(1-tt)*tt*mid_x + tt**2*sx1
                    by = (1-tt)**2*sy0 + 2*(1-tt)*tt*mid_y + tt**2*sy1
                    pts_bezier.append((int(bx), int(by)))

                for i in range(len(pts_bezier) - 1):
                    tt   = i / n
                    hue  = (h0*(1-tt) + h1*tt) % 1.0
                    # Alpha: 0 at endpoints, peak in middle
                    env  = math.sin(tt * math.pi)
                    alp  = int(90 * alpha_scale * env)
                    if alp < 4:
                        continue
                    r, g, b_ = _hue_to_rgb(hue)
                    pygame.draw.line(self._ribbon_surf,
                                     (r, g, b_, alp),
                                     pts_bezier[i], pts_bezier[i+1], width)

        screen.blit(self._ribbon_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    # ── Overlay: fiducial halos ────────────────────────────────────────────────

    def draw_halos(self, screen: pygame.Surface,
                   fid_screen: List[Tuple[float, float, float]],
                   t: float, pulse: float) -> None:
        """Multi-ring pulsing halos at fiducial positions."""
        for sx, sy, fhue in fid_screen:
            r, g, b = _hue_to_rgb(fhue)
            cx, cy = int(sx), int(sy)

            # 5 glow rings, each at a slightly different phase
            base_r = int(14 + pulse * 10)
            for layer in range(5, 0, -1):
                ring_r = base_r + layer * 9
                phase  = math.sin(t * 3.5 + layer * 0.6) * 0.5 + 0.5
                alpha  = int(30 * (layer / 5) * phase + 10)
                s = pygame.Surface((ring_r*2+4, ring_r*2+4), pygame.SRCALPHA)
                pygame.draw.circle(s, (r, g, b, alpha),
                                   (ring_r+2, ring_r+2), ring_r)
                screen.blit(s, (cx-ring_r-2, cy-ring_r-2),
                            special_flags=pygame.BLEND_ADD)

            # Solid bright core
            pygame.draw.circle(screen, (r, g, b), (cx, cy), base_r)
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy),
                               max(base_r - 5, 3))

    # ── HUD ────────────────────────────────────────────────────────────────────

    def draw_hud(self, screen: pygame.Surface,
                 frame: Frame, fps: float, t: float) -> None:
        """Minimal translucent HUD — unobtrusive."""
        if self._font is None:
            return
        mode = self.mode
        lines = [
            f"MODE: {mode}  |  fiducials: {len(frame.fiducials)}"
            f"  |  fps: {fps:.0f}  |  t: {t:.0f}s",
            "M=mode   G=glow   F=fullscreen   Q=quit",
        ]
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, (90, 90, 90))
            screen.blit(surf, (12, 10 + i * 20))
