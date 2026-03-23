"""Simulated fiducial tracker — no hardware required.

Party simulation model
──────────────────────
Each "person" has 2–3 fiducials (e.g. chest + wrists) that move together
with small body-part offsets.  Every person has a behavior mode that changes
over time:

  SITTING  — tiny jitter, essentially static.  amp×0.06, freq×0.2.
  WALKING  — gentle wandering, slow direction changes.  amp×0.4, freq×0.6.
  DANCING  — large fast sweeping motion + frequent burst kicks.  amp×1.6, freq×1.8.

Crowd dynamics
──────────────
"Scenes" set a crowd-size target.  One person arrives/leaves every 0.5 s until
the target is reached.  Scenes are short (~15 s average), so the crowd size
swings frequently and dramatically — ideal for previewing all visual states.
"""
import math
import time
from typing import List, Optional, Tuple

import numpy as np

from .base import Fiducial3D, Frame, TrackerBase

# ── working volume — spryTrack 300 ─────────────────────────────────────────────
# Baseline ~300 mm, max depth ~3000 mm, ~60° H-FOV → ±1500 mm at max depth
Z_MIN, Z_MAX = 300.0, 3000.0
X_HALF = 1500.0
Y_HALF = 850.0

# ── behavior mode parameters ───────────────────────────────────────────────────
#           amp_scale  freq_scale  burst_prob  min_dur  max_dur (seconds)
_BEHAVIOR_PARAMS = {
    'sitting': (0.04,  0.12, 0.000, 60, 240),   # nearly still, very long
    'walking': (0.22,  0.35, 0.000, 30, 120),   # slow drift, no bursts
    'dancing': (0.80,  0.90, 0.004, 20,  70),   # moderate movement
}
# How often each behavior is chosen (weights for sitting/walking/dancing)
_BEHAVIOR_WEIGHTS = [0.40, 0.40, 0.20]

# ── crowd dynamics ─────────────────────────────────────────────────────────────
_SCENE_MAX_PEOPLE    = 25
_SCENE_MEAN_DURATION = 45.0   # longer scenes — crowd stays put
_CROWD_CHANGE_INTERVAL = 2.0  # seconds — slower arrivals/departures


class _Person:
    """One party-goer with 2–3 fiducials and a changing behavior mode."""

    def __init__(self, base_index: int, n_fids: int,
                 rng: np.random.Generator) -> None:
        self.base_index = base_index
        self.n_fids = n_fids
        self.rng = rng

        # Unique fiducial indices: base_index, base_index+1, …
        self.fid_indices = list(range(base_index, base_index + n_fids))

        # Small body-part offsets in mm (chest=0,0,0 + wrists/head)
        offsets_pool = [
            (0.0, 0.0, 0.0),
            (float(rng.uniform(60, 120)),  float(rng.uniform(-30, 80)),  float(rng.uniform(-40, 40))),
            (float(rng.uniform(-120, -60)), float(rng.uniform(-30, 80)), float(rng.uniform(-40, 40))),
            (0.0, float(rng.uniform(100, 160)), 0.0),
        ]
        self._fid_offsets: List[Tuple[float, float, float]] = offsets_pool[:n_fids]

        # Base sinusoidal motion (phases, frequencies, amplitudes)
        self._phases = rng.uniform(0, 2*math.pi, size=(2, 6))
        self._base_freqs  = np.array([0.02, 0.04, 0.06, 0.015, 0.035, 0.025])  # very slow
        self._base_amps_x = rng.uniform(15, 60, size=6).astype(np.float32)     # small range
        self._base_amps_y = rng.uniform(10, 35, size=6).astype(np.float32)
        self._base_amps_z = rng.uniform(15, 60, size=6).astype(np.float32)

        # Home position
        self._cx = float(rng.uniform(-X_HALF*0.65, X_HALF*0.65))
        self._cy = float(rng.uniform(-Y_HALF*0.65, Y_HALF*0.65))
        self._cz = float(rng.uniform(500.0, 2500.0))

        # Behavior state
        self._behavior     = 'walking'
        self._behavior_end = 0.0   # simulation time when behavior changes
        self._burst_t      = 0.0
        self._burst_dur    = 0.0
        self._burst_dx     = 0.0
        self._burst_dy     = 0.0

    # ── behavior ───────────────────────────────────────────────────────────────

    def _update_behavior(self, t: float) -> None:
        """Pick a new behavior mode and schedule the next switch."""
        choices = list(_BEHAVIOR_PARAMS.keys())
        self._behavior = str(self.rng.choice(choices, p=_BEHAVIOR_WEIGHTS))
        _, _, _, mn, mx = _BEHAVIOR_PARAMS[self._behavior]
        self._behavior_end = t + float(self.rng.uniform(mn, mx))

    # ── position ───────────────────────────────────────────────────────────────

    def _base_position(self, t: float) -> Tuple[float, float, float]:
        """Center-of-body position at time t."""
        if t >= self._behavior_end:
            self._update_behavior(t)

        amp_s, freq_s, burst_p, *_ = _BEHAVIOR_PARAMS[self._behavior]
        freqs  = self._base_freqs * freq_s
        angles = 2*math.pi * freqs * t + self._phases[0]

        x = self._cx + float(np.dot(self._base_amps_x * amp_s, np.sin(angles)))
        y = self._cy + float(np.dot(self._base_amps_y * amp_s, np.cos(angles)))
        z = self._cz + float(np.dot(self._base_amps_z * amp_s,
                                    np.sin(angles + self._phases[1])))

        # Dance burst
        if t > self._burst_t + self._burst_dur:
            if self.rng.random() < burst_p:
                self._burst_t   = t
                self._burst_dur = float(self.rng.uniform(0.3, 1.4))
                self._burst_dx  = float(self.rng.uniform(-320, 320))
                self._burst_dy  = float(self.rng.uniform(-220, 220))

        if t < self._burst_t + self._burst_dur:
            phase = (t - self._burst_t) / self._burst_dur * math.pi
            s = math.sin(phase)
            x += self._burst_dx * s
            y += self._burst_dy * s

        x = float(np.clip(x, -X_HALF, X_HALF))
        y = float(np.clip(y, -Y_HALF, Y_HALF))
        z = float(np.clip(z, Z_MIN, Z_MAX))
        return x, y, z

    def positions(self, t: float) -> List[Tuple[float, float, float, int]]:
        """Return (x, y, z, index) for each fiducial on this person."""
        bx, by, bz = self._base_position(t)
        result = []
        for i, (ox, oy, oz) in enumerate(self._fid_offsets):
            fx = float(np.clip(bx + ox, -X_HALF, X_HALF))
            fy = float(np.clip(by + oy, -Y_HALF, Y_HALF))
            fz = float(np.clip(bz + oz, Z_MIN, Z_MAX))
            result.append((fx, fy, fz, self.fid_indices[i]))
        return result

    @property
    def behavior(self) -> str:
        return self._behavior


class SimulatedTracker(TrackerBase):
    """Drop-in simulator for the real spryTrack 300.

    Parameters
    ----------
    n_fiducials:
        Starting fiducial count (people × ~2.5 fids each).
        0 = start with empty room and let the crowd arrive naturally.
    fps:
        Simulated frame rate.
    seed:
        RNG seed (None = random).
    """

    def __init__(self, n_fiducials: int = 6, fps: float = 30.0,
                 seed: Optional[int] = None) -> None:
        self._initial_n = n_fiducials
        self._fps  = fps
        self._dt   = 1.0 / fps
        self._seed = seed
        self._people: List[_Person] = []
        self._next_index  = 0
        self._t0          = 0.0
        self._last_tick   = 0.0
        self._frame_idx   = 0
        self._rng = np.random.default_rng(seed)

        self._scene_target  = 0
        self._scene_end_t   = 0.0
        self._last_change_t = 0.0

    # ── crowd management ───────────────────────────────────────────────────────

    def _new_person(self) -> _Person:
        n_fids = int(self._rng.choice([2, 2, 3]))   # 2/3 → 2 fids, 1/3 → 3 fids
        p = _Person(self._next_index, n_fids,
                    np.random.default_rng(self._rng.integers(0, 2**31)))
        self._next_index += n_fids
        return p

    def _initial_people(self, n_fiducials: int) -> List[_Person]:
        """Spawn enough people to cover approximately n_fiducials fiducials."""
        people = []
        count = 0
        while count < n_fiducials:
            p = self._new_person()
            people.append(p)
            count += p.n_fids
        return people

    def _next_scene(self, now: float) -> None:
        """Pick new target crowd size + schedule next scene change."""
        r = self._rng.random()
        if r < 0.15:
            # Empty room
            target_people = 0
        elif r < 0.35:
            # Small group (1–3 people)
            target_people = int(self._rng.integers(1, 4))
        elif r < 0.65:
            # Medium (4–10)
            target_people = int(self._rng.integers(4, 11))
        elif r < 0.85:
            # Busy (11–18)
            target_people = int(self._rng.integers(11, 19))
        else:
            # Packed (19–25)
            target_people = int(self._rng.integers(19, _SCENE_MAX_PEOPLE + 1))

        self._scene_target = target_people
        dur = max(float(self._rng.exponential(_SCENE_MEAN_DURATION)), 6.0)
        self._scene_end_t  = now + dur

        n_fids_approx = target_people * 2  # rough estimate for logging
        behaviors = [p.behavior for p in self._people]
        print(f"[SimulatedTracker] scene → {target_people} people "
              f"(~{n_fids_approx} fids)  for {dur:.0f}s")

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._next_index = 0
        self._people = self._initial_people(self._initial_n)
        self._scene_target = len(self._people)

        now = time.perf_counter()
        self._t0 = now
        self._last_tick     = now
        self._last_change_t = now
        self._scene_end_t   = now + float(self._rng.uniform(10.0, 20.0))

        n_fids = sum(p.n_fids for p in self._people)
        print(f"[SimulatedTracker] opened — {len(self._people)} people / "
              f"{n_fids} fiducials at {self._fps} fps  (crowd evolves dynamically)")

    def get_frame(self) -> Frame:
        # Pace to target fps
        now = time.perf_counter()
        sleep_for = self._dt - (now - self._last_tick)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_tick = time.perf_counter()
        now = self._last_tick

        # ── crowd dynamics ─────────────────────────────────────────────────
        if now >= self._scene_end_t:
            self._next_scene(now)

        if now - self._last_change_t >= _CROWD_CHANGE_INTERVAL:
            n = len(self._people)
            if n < self._scene_target:
                p = self._new_person()
                self._people.append(p)
                n_fids = sum(pp.n_fids for pp in self._people)
                print(f"[SimulatedTracker] +1 arrived ({p.behavior:7s} "
                      f"{p.n_fids}fids) → {len(self._people)} people / {n_fids} fids")
                self._last_change_t = now
            elif n > self._scene_target:
                gone = self._people.pop(int(self._rng.integers(0, n)))
                n_fids = sum(pp.n_fids for pp in self._people)
                print(f"[SimulatedTracker] -1 left                      "
                      f"→ {len(self._people)} people / {n_fids} fids")
                self._last_change_t = now

        # ── emit fiducials ─────────────────────────────────────────────────
        t = now - self._t0
        timestamp_us = int(t * 1_000_000)

        fiducials: List[Fiducial3D] = []
        for person in self._people:
            for fx, fy, fz, fidx in person.positions(t):
                tri_err = float(self._rng.uniform(0.05, 0.4))
                epi_err = float(self._rng.uniform(0.3, 1.5))
                prob    = float(self._rng.uniform(0.85, 1.0))
                fiducials.append(Fiducial3D(
                    x=fx, y=fy, z=fz,
                    index=fidx,
                    probability=prob,
                    epipolar_error_px=epi_err,
                    triangulation_error_mm=tri_err,
                ))

        self._frame_idx += 1
        return Frame(fiducials=fiducials, timestamp_us=timestamp_us,
                     frame_index=self._frame_idx)

    def add_person(self) -> None:
        """Manually add one person and lock the scene target."""
        p = self._new_person()
        self._people.append(p)
        self._scene_target = len(self._people)
        n_fids = sum(pp.n_fids for pp in self._people)
        print(f"[SimulatedTracker] +1 manual ({p.behavior:7s} {p.n_fids}fids)"
              f" -> {len(self._people)} people / {n_fids} fids")

    def remove_person(self) -> None:
        """Manually remove one person and lock the scene target."""
        if not self._people:
            return
        gone = self._people.pop(int(self._rng.integers(0, len(self._people))))
        self._scene_target = len(self._people)
        n_fids = sum(pp.n_fids for pp in self._people)
        print(f"[SimulatedTracker] -1 manual"
              f" -> {len(self._people)} people / {n_fids} fids")

    def close(self) -> None:
        print("[SimulatedTracker] closed")
