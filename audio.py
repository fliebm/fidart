"""audio.py — Real-time microphone/line-in analysis for fidart.

Runs in a background thread.  Exposes AudioFeatures with smoothed,
normalised [0,1] values ready to plug into the visual engine.

Feature design
──────────────
  rms      — overall loudness (energy across all frequencies)
  bass     — 20–250 Hz  (kick drum, bass notes)
  mid      — 250–4 kHz  (instruments, vocals)
  high     — 4–20 kHz   (cymbals, hi-hats, air)
  beat     — transient strength; spikes on each beat, decays between hits
  centroid — spectral centroid ∈ [0,1]: 0=all bass, 1=all treble

Normalisation
─────────────
Each band tracks a slow running-maximum (half-life ≈ 3 min) so the
values auto-calibrate to the actual room level within a few seconds.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

# ── tunable constants ──────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
CHUNK         = 1024          # ~46 ms per callback at 22050 Hz

# Beat detection
BEAT_LOOKBACK = 40            # chunks of history (~1.9 s)
BEAT_THRESH   = 1.50          # current energy must exceed mean × this
BEAT_COOLDOWN = 0.18          # s — minimum gap between recognised beats
BEAT_DECAY    = 7.0           # beat value decays at this rate per second

# Running-max normalisation: slow decay keeps it relevant for hours
# half-life ≈ 3 min at 22 callbacks/s → decay ≈ 0.9998
_NORM_DECAY   = 0.9998
_NORM_FLOOR   = 1e-4           # avoids division by zero in silence


@dataclass
class AudioFeatures:
    rms:      float = 0.0
    bass:     float = 0.0
    mid:      float = 0.0
    high:     float = 0.0
    beat:     float = 0.0
    centroid: float = 0.5

    @classmethod
    def silent(cls) -> "AudioFeatures":
        return cls()


class AudioProcessor:
    """Background mic capture + feature extraction.

    Usage::
        ap = AudioProcessor()
        ap.open()          # starts background thread
        af = ap.features   # call each frame
        ap.close()
    """

    def __init__(self, device=None) -> None:
        self._device  = device
        self._stream  = None
        self._active  = False
        self._lock    = threading.Lock()

        # Public feature snapshot (written from audio thread, read from main)
        self._snap = AudioFeatures.silent()

        # Beat detection: circular energy history
        self._energy_hist = np.zeros(BEAT_LOOKBACK, np.float32)
        self._hist_ptr    = 0
        self._beat_val    = 0.0
        self._last_beat   = 0.0
        self._last_read   = time.perf_counter()

        # EMA state
        self._rms_ema  = 0.0
        self._bass_ema = 0.0
        self._mid_ema  = 0.0
        self._high_ema = 0.0

        # Running max for adaptive normalisation
        self._rms_max  = _NORM_FLOOR
        self._bass_max = _NORM_FLOOR
        self._mid_max  = _NORM_FLOOR
        self._high_max = _NORM_FLOOR

        # Frequency bin masks (computed once)
        freqs              = np.fft.rfftfreq(CHUNK, 1.0 / SAMPLE_RATE)
        self._freqs        = freqs.astype(np.float32)
        self._bass_mask    = (freqs >= 20)   & (freqs < 250)
        self._mid_mask     = (freqs >= 250)  & (freqs < 4000)
        self._high_mask    = (freqs >= 4000) & (freqs < 20000)
        # Hann window (computed once, reused in callback)
        self._window       = np.hanning(CHUNK).astype(np.float32)

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> bool:
        if not _SD_AVAILABLE:
            print("[Audio] sounddevice not installed — audio disabled.\n"
                  "        Install with:  pip install sounddevice")
            return False
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK,
                device=self._device,
                callback=self._callback,
            )
            self._stream.start()
            self._active = True
            dev = self._device or "default"
            print(f"[Audio] microphone active  (device: {dev})")
            return True
        except Exception as exc:
            print(f"[Audio] could not open microphone: {exc}")
            return False

    def close(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._active = False

    # ── feature access ─────────────────────────────────────────────────────────

    @property
    def features(self) -> AudioFeatures:
        """Return the latest AudioFeatures snapshot.

        Beat value is decayed in real time so callers always get the right
        value regardless of when the audio callback last fired.
        """
        now = time.perf_counter()
        dt  = now - self._last_read
        self._last_read = now

        with self._lock:
            # Decay beat outside the audio callback so it feels instantaneous
            self._beat_val = max(0.0, self._beat_val - dt * BEAT_DECAY)
            return AudioFeatures(
                rms      = self._snap.rms,
                bass     = self._snap.bass,
                mid      = self._snap.mid,
                high     = self._snap.high,
                beat     = float(self._beat_val),
                centroid = self._snap.centroid,
            )

    # ── audio callback (runs on sounddevice thread) ────────────────────────────

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa
        mono = indata[:, 0]

        # RMS
        rms_raw = float(np.sqrt(np.mean(mono * mono)))

        # FFT with Hann window
        spectrum = np.abs(np.fft.rfft(mono * self._window)) / frames

        # Band energies
        bass_e = float(spectrum[self._bass_mask].mean()) if self._bass_mask.any() else 0.0
        mid_e  = float(spectrum[self._mid_mask].mean())  if self._mid_mask.any()  else 0.0
        high_e = float(spectrum[self._high_mask].mean()) if self._high_mask.any() else 0.0

        # Spectral centroid (0=bass-only, 1=all-treble)
        total = float(spectrum.sum())
        if total > 1e-8:
            centroid_hz   = float((self._freqs[:len(spectrum)] * spectrum).sum() / total)
            centroid_norm = float(np.clip(centroid_hz / 8000.0, 0.0, 1.0))
        else:
            centroid_norm = 0.5

        # EMA smoothing — α=0.30 → τ ≈ 3 chunks ≈ 140 ms
        a = 0.30
        self._rms_ema  = self._rms_ema  * (1-a) + rms_raw * a
        self._bass_ema = self._bass_ema * (1-a) + bass_e  * a
        self._mid_ema  = self._mid_ema  * (1-a) + mid_e   * a
        self._high_ema = self._high_ema * (1-a) + high_e  * a

        # Adaptive normalisation: running max with slow decay
        self._rms_max  = max(self._rms_max  * _NORM_DECAY, self._rms_ema,  _NORM_FLOOR)
        self._bass_max = max(self._bass_max * _NORM_DECAY, self._bass_ema, _NORM_FLOOR)
        self._mid_max  = max(self._mid_max  * _NORM_DECAY, self._mid_ema,  _NORM_FLOOR)
        self._high_max = max(self._high_max * _NORM_DECAY, self._high_ema, _NORM_FLOOR)

        rms_n  = float(np.clip(self._rms_ema  / self._rms_max,  0.0, 1.0))
        bass_n = float(np.clip(self._bass_ema / self._bass_max, 0.0, 1.0))
        mid_n  = float(np.clip(self._mid_ema  / self._mid_max,  0.0, 1.0))
        high_n = float(np.clip(self._high_ema / self._high_max, 0.0, 1.0))

        # Beat detection: current energy vs recent history
        energy = rms_raw * rms_raw
        mean_e = float(self._energy_hist.mean())
        now    = time.perf_counter()
        if (mean_e > 1e-7
                and energy > BEAT_THRESH * mean_e
                and now - self._last_beat > BEAT_COOLDOWN):
            strength = float(np.clip(energy / (mean_e * BEAT_THRESH), 0.5, 2.0))
            self._beat_val  = min(1.0, self._beat_val + strength * 0.65)
            self._last_beat = now

        self._energy_hist[self._hist_ptr] = energy
        self._hist_ptr = (self._hist_ptr + 1) % BEAT_LOOKBACK

        with self._lock:
            self._snap = AudioFeatures(
                rms      = rms_n,
                bass     = bass_n,
                mid      = mid_n,
                high     = high_n,
                beat     = float(self._beat_val),
                centroid = centroid_norm,
            )
