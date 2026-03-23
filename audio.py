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

Backends
────────
  Microphone   — sounddevice (cross-platform)
  System audio — pyaudiowpatch (Windows WASAPI loopback)
                 Install: pip install pyaudiowpatch
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

try:
    import pyaudiowpatch as pyaudio
    _PAWP_AVAILABLE = True
except ImportError:
    _PAWP_AVAILABLE = False


def find_loopback_device() -> Optional[bool]:
    """Return True if system-audio loopback is available, None if not.

    Actual device selection happens inside AudioProcessor.open() using
    pyaudiowpatch, which enumerates WASAPI loopback devices directly.
    """
    if _PAWP_AVAILABLE:
        return True
    return None


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
    sub_bass: float = 0.0
    mid:      float = 0.0
    high:     float = 0.0
    beat:     float = 0.0
    onset:    float = 0.0
    centroid: float = 0.5
    energy:   float = 0.0

    @classmethod
    def silent(cls) -> "AudioFeatures":
        return cls()


class AudioProcessor:
    """Background audio capture + feature extraction.

    Usage::
        ap = AudioProcessor()
        ap.open()          # starts background thread
        af = ap.features   # call each frame
        ap.close()
    """

    def __init__(self, device=None, loopback: bool = False) -> None:
        self._device     = device
        self._loopback   = loopback
        self._stream     = None   # sounddevice stream (mic) or pyaudio stream (loopback)
        self._pa         = None   # pyaudiowpatch PyAudio instance (loopback only)
        self._active     = False
        self._n_channels = 1
        self._dbg_calls  = 0
        self._lock       = threading.Lock()

        # Public feature snapshot (written from audio thread, read from main)
        self._snap = AudioFeatures.silent()

        # Beat detection: circular energy history
        self._energy_hist = np.zeros(BEAT_LOOKBACK, np.float32)
        self._hist_ptr    = 0
        self._beat_val    = 0.0
        self._last_beat   = 0.0
        self._last_read   = time.perf_counter()

        # EMA state
        self._rms_ema      = 0.0
        self._bass_ema     = 0.0
        self._mid_ema      = 0.0
        self._high_ema     = 0.0
        self._sub_bass_ema = 0.0
        self._flux_ema     = 0.0
        self._energy_ema   = 0.0

        # Running max for adaptive normalisation
        self._rms_max      = _NORM_FLOOR
        self._bass_max     = _NORM_FLOOR
        self._mid_max      = _NORM_FLOOR
        self._high_max     = _NORM_FLOOR
        self._sub_bass_max = _NORM_FLOOR
        self._flux_max     = _NORM_FLOOR

        # Spectral flux state
        self._prev_spectrum   = None
        self._sub_energy_hist = np.zeros(BEAT_LOOKBACK, np.float32)

        # Frequency bin masks — recomputed in open() once sample rate is known
        self._freqs        = np.array([], np.float32)
        self._bass_mask    = np.array([], bool)
        self._mid_mask     = np.array([], bool)
        self._high_mask    = np.array([], bool)
        self._sub_bass_mask = np.array([], bool)
        self._window       = np.hanning(CHUNK).astype(np.float32)
        self._actual_rate  = SAMPLE_RATE

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> bool:
        if self._loopback:
            return self._open_loopback()
        return self._open_mic()

    def _open_mic(self) -> bool:
        if not _SD_AVAILABLE:
            print("[Audio] sounddevice not installed — audio disabled.\n"
                  "        Install with:  pip install sounddevice")
            return False
        try:
            dev_info    = sd.query_devices(self._device or sd.default.device[0])
            use_rate    = int(dev_info.get('default_samplerate', SAMPLE_RATE))
            self._build_masks(use_rate)
            self._n_channels = 1

            self._stream = sd.InputStream(
                samplerate=use_rate,
                channels=1,
                dtype="float32",
                blocksize=CHUNK,
                device=self._device,
                callback=self._sd_callback,
            )
            self._stream.start()
            self._active = True
            dev = self._device if self._device is not None else "default"
            print(f"[Audio] microphone active  (device: {dev!r}  rate: {use_rate} Hz)")
            return True
        except Exception as exc:
            print(f"[Audio] could not open microphone: {exc}")
            return False

    def _open_loopback(self) -> bool:
        if not _PAWP_AVAILABLE:
            print("[Audio] pyaudiowpatch not installed — cannot capture system audio.\n"
                  "        Install with:  pip install pyaudiowpatch")
            return False
        try:
            pa = pyaudio.PyAudio()
            try:
                wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            except OSError:
                print("[Audio] WASAPI not available")
                pa.terminate()
                return False

            default_out  = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            default_name = default_out["name"]

            # Find the loopback mirror of the default output device
            loopback_dev = None
            print("[Audio] searching for WASAPI loopback devices …")
            for dev in pa.get_loopback_device_info_generator():
                print(f"  found loopback: [{dev['index']}] {dev['name']}")
                if loopback_dev is None or default_name in dev["name"]:
                    loopback_dev = dev

            if loopback_dev is None:
                print("[Audio] no loopback device found via pyaudiowpatch")
                pa.terminate()
                return False

            use_rate = int(loopback_dev["defaultSampleRate"])
            n_ch     = int(loopback_dev["maxInputChannels"])
            self._build_masks(use_rate)
            self._n_channels = n_ch
            self._pa = pa

            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=n_ch,
                rate=use_rate,
                input=True,
                input_device_index=int(loopback_dev["index"]),
                frames_per_buffer=CHUNK,
                stream_callback=self._pa_callback,
            )
            stream.start_stream()
            self._stream = stream
            self._active = True
            print(f"[Audio] system audio (loopback) active  "
                  f"'{loopback_dev['name']}'  rate: {use_rate} Hz  ch: {n_ch}")
            return True
        except Exception as exc:
            print(f"[Audio] could not open loopback: {exc}")
            if self._pa:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None
            return False

    def _build_masks(self, rate: int) -> None:
        freqs = np.fft.rfftfreq(CHUNK, 1.0 / rate)
        self._freqs         = freqs.astype(np.float32)
        self._sub_bass_mask = (freqs >= 20)   & (freqs < 80)
        self._bass_mask     = (freqs >= 20)   & (freqs < 250)
        self._mid_mask      = (freqs >= 250)  & (freqs < 4000)
        self._high_mask     = (freqs >= 4000) & (freqs < 20000)
        self._actual_rate   = rate

    def close(self) -> None:
        if self._stream:
            try:
                if self._pa:                    # pyaudiowpatch stream
                    self._stream.stop_stream()
                    self._stream.close()
                else:                           # sounddevice stream
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
        self._active = False

    # ── feature access ─────────────────────────────────────────────────────────

    @property
    def features(self) -> AudioFeatures:
        """Return the latest AudioFeatures snapshot."""
        now = time.perf_counter()
        dt  = now - self._last_read
        self._last_read = now

        with self._lock:
            self._beat_val = max(0.0, self._beat_val - dt * BEAT_DECAY)
            return AudioFeatures(
                rms      = self._snap.rms,
                bass     = self._snap.bass,
                sub_bass = self._snap.sub_bass,
                mid      = self._snap.mid,
                high     = self._snap.high,
                beat     = float(self._beat_val),
                onset    = self._snap.onset,
                centroid = self._snap.centroid,
                energy   = self._snap.energy,
            )

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _sd_callback(self, indata, frames, time_info, status) -> None:  # noqa
        mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
        self._process_chunk(mono, frames)

    def _pa_callback(self, in_data, frame_count, time_info, status):
        mono = np.frombuffer(in_data, dtype=np.float32)
        if self._n_channels > 1:
            mono = mono.reshape(-1, self._n_channels).mean(axis=1)
        self._process_chunk(mono, frame_count)
        return (None, pyaudio.paContinue)

    def _process_chunk(self, mono: np.ndarray, frames: int) -> None:
        # Debug: print raw RMS every ~3 s so silence is obvious in console
        self._dbg_calls += 1
        if self._dbg_calls % 150 == 1:
            raw = float(np.sqrt(np.mean(mono * mono)))
            print(f"[Audio] raw RMS={raw:.5f}  len={len(mono)}"
                  + ("  *** SILENCE ***" if raw < 1e-5 else ""))

        # RMS
        rms_raw = float(np.sqrt(np.mean(mono * mono)))

        # FFT with Hann window
        spectrum = np.abs(np.fft.rfft(mono * self._window)) / frames

        # Band energies
        bass_e     = float(spectrum[self._bass_mask].mean())     if self._bass_mask.any()     else 0.0
        mid_e      = float(spectrum[self._mid_mask].mean())      if self._mid_mask.any()      else 0.0
        high_e     = float(spectrum[self._high_mask].mean())     if self._high_mask.any()     else 0.0
        sub_bass_e = float(spectrum[self._sub_bass_mask].mean()) if self._sub_bass_mask.any() else 0.0

        # Spectral centroid (0=bass-only, 1=all-treble)
        total = float(spectrum.sum())
        if total > 1e-8:
            centroid_hz   = float((self._freqs[:len(spectrum)] * spectrum).sum() / total)
            centroid_norm = float(np.clip(centroid_hz / 8000.0, 0.0, 1.0))
        else:
            centroid_norm = 0.5

        # Half-wave spectral flux (onset detection)
        if self._prev_spectrum is not None and len(self._prev_spectrum) == len(spectrum):
            flux_raw = float(np.maximum(spectrum - self._prev_spectrum, 0).sum())
        else:
            flux_raw = 0.0
        self._prev_spectrum = spectrum.copy()

        # EMA smoothing — α=0.30 → τ ≈ 3 chunks ≈ 140 ms
        a = 0.30
        self._rms_ema  = self._rms_ema  * (1-a) + rms_raw * a
        self._bass_ema = self._bass_ema * (1-a) + bass_e  * a
        self._mid_ema  = self._mid_ema  * (1-a) + mid_e   * a
        self._high_ema = self._high_ema * (1-a) + high_e  * a

        # Sub-bass EMA + normalise
        self._sub_bass_ema = self._sub_bass_ema * (1 - a) + sub_bass_e * a
        self._sub_bass_max = max(self._sub_bass_max * _NORM_DECAY, self._sub_bass_ema, _NORM_FLOOR)
        sub_bass_n = float(np.clip(self._sub_bass_ema / self._sub_bass_max, 0.0, 1.0))

        # Flux EMA + normalise
        self._flux_ema = self._flux_ema * 0.75 + flux_raw * 0.25
        self._flux_max = max(self._flux_max * _NORM_DECAY, self._flux_ema, _NORM_FLOOR)
        onset_n = float(np.clip(self._flux_ema / self._flux_max, 0.0, 1.0))

        # Adaptive normalisation: running max with slow decay
        self._rms_max  = max(self._rms_max  * _NORM_DECAY, self._rms_ema,  _NORM_FLOOR)
        self._bass_max = max(self._bass_max * _NORM_DECAY, self._bass_ema, _NORM_FLOOR)
        self._mid_max  = max(self._mid_max  * _NORM_DECAY, self._mid_ema,  _NORM_FLOOR)
        self._high_max = max(self._high_max * _NORM_DECAY, self._high_ema, _NORM_FLOOR)

        rms_n  = float(np.clip(self._rms_ema  / self._rms_max,  0.0, 1.0))
        bass_n = float(np.clip(self._bass_ema / self._bass_max, 0.0, 1.0))
        mid_n  = float(np.clip(self._mid_ema  / self._mid_max,  0.0, 1.0))
        high_n = float(np.clip(self._high_ema / self._high_max, 0.0, 1.0))

        # Scene energy: bass + sub_bass drive it; rises fast, decays very slowly
        target_energy = float(np.clip(bass_n * 0.5 + sub_bass_n * 0.5, 0.0, 1.0))
        if target_energy > self._energy_ema:
            self._energy_ema = self._energy_ema * 0.80 + target_energy * 0.20   # rises fast
        else:
            self._energy_ema = self._energy_ema * 0.997 + target_energy * 0.003  # decays slowly
        energy_n = float(np.clip(self._energy_ema, 0.0, 1.0))

        # Beat detection: sub-bass energy spikes (kick drum detection)
        sub_energy = sub_bass_e * sub_bass_e
        mean_sub_e = float(self._sub_energy_hist.mean())
        now    = time.perf_counter()
        if (mean_sub_e > 1e-8
                and sub_energy > BEAT_THRESH * mean_sub_e
                and now - self._last_beat > BEAT_COOLDOWN):
            strength = float(np.clip(sub_energy / (mean_sub_e * BEAT_THRESH), 0.5, 2.0))
            self._beat_val  = min(1.0, self._beat_val + strength * 0.65)
            self._last_beat = now
        # also keep overall RMS history for fallback
        self._energy_hist[self._hist_ptr]     = rms_raw * rms_raw
        self._sub_energy_hist[self._hist_ptr] = sub_energy
        self._hist_ptr = (self._hist_ptr + 1) % BEAT_LOOKBACK

        with self._lock:
            self._snap = AudioFeatures(
                rms      = rms_n,
                bass     = bass_n,
                sub_bass = sub_bass_n,
                mid      = mid_n,
                high     = high_n,
                beat     = float(self._beat_val),
                onset    = onset_n,
                centroid = centroid_norm,
                energy   = energy_n,
            )
