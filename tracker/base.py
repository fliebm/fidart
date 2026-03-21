"""Abstract base class for all tracker backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class Fiducial3D:
    """A single tracked IR fiducial in 3D space.

    Coordinate system (matches spryTrack 300):
      - Origin at camera
      - Z forward (depth, mm)
      - X right (mm)
      - Y up (mm)
    Typical working volume:
      Z: 300–2000 mm
      X: ±700 mm
      Y: ±400 mm
    """
    x: float          # mm
    y: float          # mm
    z: float          # mm
    index: int = 0    # track identity (stable across frames)
    probability: float = 1.0          # 0..1, SDK confidence
    epipolar_error_px: float = 0.0    # pixels, SDK quality metric
    triangulation_error_mm: float = 0.0  # mm, SDK quality metric


@dataclass
class Frame:
    """One tracking frame."""
    fiducials: List[Fiducial3D] = field(default_factory=list)
    timestamp_us: int = 0   # microseconds, matches SDK ftkImageHeader.timestampUS
    frame_index: int = 0


class TrackerBase(ABC):
    """Shared interface for real SDK and simulator."""

    @abstractmethod
    def open(self) -> None:
        """Initialize hardware or simulation."""

    @abstractmethod
    def get_frame(self) -> Frame:
        """Return the latest frame (blocks until data ready)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    # Context manager support
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
