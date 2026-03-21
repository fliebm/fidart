"""Real spryTrack 300 tracker via the Atracsys Python SDK.

Install the SDK wheel first:
    pip install "C:/Program Files/Atracsys/spryTrack SDK x64/python/atracsys-*.tar.gz"

The SDK exposes:
    import atracsys.stk as tracking_sdk
    tracker = tracking_sdk.TrackingSystem()
    tracker.initialise()
    tracker.enumerate_devices()
    frame = tracking_sdk.FrameData()
    tracker.create_frame(False, 10, 20, 20, 10)
    tracker.get_last_frame(frame)
    for fid in frame.fiducials:
        x, y, z = fid.position[0], fid.position[1], fid.position[2]

References:
  - ftkInterface.h  → ftk3DFiducial  (positionMM.x/y/z in mm)
  - doc/python/index.rst.txt
  - samples/stereo4_AcquisitionRawData.cpp
"""
import time
from typing import Any

from .base import Fiducial3D, Frame, TrackerBase

# Number of 3D fiducial slots to pre-allocate for ftkSetFrameOptions
_MAX_FIDUCIALS = 64
_FRAME_TIMEOUT_MS = 100


class SDKTracker(TrackerBase):
    """Live spryTrack 300 tracker.

    Parameters
    ----------
    serial_number:
        Camera serial number (int).  Pass None to use the first device found.
    """

    def __init__(self, serial_number: int | None = None):
        self._sn = serial_number
        self._sdk: Any = None
        self._frame_data: Any = None
        self._frame_idx = 0

    def open(self) -> None:
        try:
            import atracsys.stk as tracking_sdk  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "atracsys Python package not found.\n"
                "Install with:\n"
                '  pip install "C:/Program Files/Atracsys/spryTrack SDK x64/python/atracsys-*.tar.gz"'
            ) from exc

        self._sdk = tracking_sdk.TrackingSystem()
        self._sdk.initialise()

        # Enumerate devices; picks up first device or the requested serial
        devices = self._sdk.enumerate_devices()
        if not devices:
            raise RuntimeError("No spryTrack device found. Check USB connection.")

        if self._sn is None:
            self._sn = devices[0]
            print(f"[SDKTracker] Using first device: SN={self._sn}")
        elif self._sn not in devices:
            raise RuntimeError(f"Device SN={self._sn} not found. Available: {devices}")

        # Allocate frame buffer:
        # create_frame(pixels, n_events, left_raw, right_raw, n_3d_fiducials, n_markers)
        self._sdk.create_frame(
            False,            # no raw images
            0,                # no events
            _MAX_FIDUCIALS,   # left raw blobs
            _MAX_FIDUCIALS,   # right raw blobs
            _MAX_FIDUCIALS,   # 3D fiducials  ← what we care about
            0,                # no rigid body markers
        )
        self._frame_data = tracking_sdk.FrameData()
        print(f"[SDKTracker] opened — SN={self._sn}")

    def get_frame(self) -> Frame:
        err = self._sdk.get_last_frame(self._frame_data, _FRAME_TIMEOUT_MS)

        # FTK_OK == 0; FTK_WAR_NO_FRAME == -1 (no new frame yet, return empty)
        if err != 0:
            return Frame(frame_index=self._frame_idx)

        fiducials = []
        for i, fid in enumerate(self._frame_data.fiducials):
            pos = fid.position   # [x, y, z] in mm
            fiducials.append(Fiducial3D(
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]),
                index=i,
                probability=float(getattr(fid, 'probability', 1.0)),
                epipolar_error_px=float(getattr(fid, 'epipolarError', 0.0)),
                triangulation_error_mm=float(getattr(fid, 'triangulationError', 0.0)),
            ))

        # Timestamp from image header (microseconds)
        ts = getattr(self._frame_data, 'timestamp_us', int(time.perf_counter() * 1e6))

        self._frame_idx += 1
        return Frame(fiducials=fiducials, timestamp_us=ts, frame_index=self._frame_idx)

    def close(self) -> None:
        if self._sdk is not None:
            try:
                self._sdk.close()
            except Exception:
                pass
            self._sdk = None
        print("[SDKTracker] closed")
