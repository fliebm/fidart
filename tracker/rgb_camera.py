"""RGB camera tracker — multi-person body tracking via YOLO pose estimation.

Uses YOLO11n-pose (downloads automatically on first run, ~6 MB) to detect up to
25+ people from any connected webcam and maps body keypoints to Fiducial3D objects
in the same coordinate system as the spryTrack 300.

Install dependencies:
    pip install ultralytics opencv-python

Emitted fiducials per person (up to 4, skipped if keypoint confidence too low):
    slot 0 — head        (nose keypoint)
    slot 1 — chest       (midpoint of shoulders)
    slot 2 — left wrist
    slot 3 — right wrist

Coordinate mapping:
    X: (pixel_x / frame_width  - 0.5) × 3000 mm   →  ±1500 mm
    Y: (pixel_y / frame_height - 0.5) × 1700 mm   →  ±850 mm  (inverted — up is positive)
    Z: DEPTH_SCALE / bounding_box_height_px         →  300–3000 mm  (tune per room)

DEPTH_SCALE default 250 000 gives ~1250 mm for a 200 px-tall person at 720 p.
Increase for a larger room / farther camera; decrease if people are very close.
"""
import math
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import Fiducial3D, Frame, TrackerBase

# ── COCO-17 keypoint indices ────────────────────────────────────────────────────
_NOSE       = 0
_L_EAR      = 3
_R_EAR      = 4
_L_SHOULDER = 5
_R_SHOULDER = 6

# ── tunable constants ───────────────────────────────────────────────────────────
_KP_CONF_MIN  = 0.40          # skip keypoints below this confidence
_DET_CONF     = 0.35          # YOLO person detection threshold
_Z_MIN, _Z_MAX = 300.0, 3000.0
_X_HALF, _Y_HALF = 1500.0, 850.0
_DEPTH_SCALE  = 250_000.0     # Z ≈ DEPTH_SCALE / box_height_px  (tune per room)
_MAX_MATCH_DIST = 0.20        # normalised centroid distance for ID re-use


class RGBCameraTracker(TrackerBase):
    """Multi-person body tracker via webcam + YOLO11-pose.

    Parameters
    ----------
    camera_index : int
        OpenCV camera index (0 = first webcam, 1 = second, …).
    model_name : str
        YOLO model name without extension.
        'yolo11n-pose' — fastest, good for CPU.
        'yolo11s-pose' — slightly slower, more accurate.
        'yolov8n-pose'  — fallback if YOLO11 unavailable.
    fps : float
        Target frame rate.  0 = run as fast as inference allows.
    depth_scale : float
        Tune this per environment.  Larger value = people appear farther away.
    """

    def __init__(self, camera_index: int = 0,
                 model_name: str = 'yolo11n-pose',
                 fps: float = 30.0,
                 depth_scale: float = _DEPTH_SCALE,
                 debug: bool = False) -> None:
        self._cam_idx     = camera_index
        self._model_name  = model_name
        self._fps         = fps
        self._dt          = 1.0 / fps if fps > 0 else 0.0
        self._depth_scale = depth_scale
        self._debug       = debug
        self._cap         = None
        self._model       = None
        self._frame_idx   = 0
        self._last_tick   = 0.0
        self._frame_w     = 640
        self._frame_h     = 480

        # Person identity: pid → last normalised centroid (cx, cy)
        self._tracked: Dict[int, Tuple[float, float]] = {}
        self._next_pid = 0

        # Background inference thread state
        self._latest_frame: Frame = Frame()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python not found.\n"
                "Install with:  pip install opencv-python"
            ) from exc
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics not found.\n"
                "Install with:  pip install ultralytics"
            ) from exc

        self._cap = cv2.VideoCapture(self._cam_idx)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self._cam_idx}.  "
                "Check that a webcam is connected and not in use by another app."
            )

        self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[RGBCameraTracker] camera {self._cam_idx} — "
              f"{self._frame_w}×{self._frame_h}")

        print(f"[RGBCameraTracker] loading '{self._model_name}' "
              f"(auto-downloads ~6 MB on first run)…")
        self._model = YOLO(self._model_name + '.pt')

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        print(f"[RGBCameraTracker] ready  conf={_DET_CONF}  "
              f"fps_target={self._fps}  depth_scale={self._depth_scale:.0f}")

    def get_frame(self) -> Frame:
        """Return the latest frame produced by the background inference thread."""
        with self._lock:
            return self._latest_frame

    def _inference_loop(self) -> None:
        """Background thread: capture → YOLO → update latest frame."""
        last_tick = time.perf_counter()
        while not self._stop_event.is_set():
            # Pace to target fps
            if self._dt > 0:
                wait = self._dt - (time.perf_counter() - last_tick)
                if wait > 0:
                    time.sleep(wait)
            last_tick = time.perf_counter()

            ret, img = self._cap.read()
            if not ret:
                continue

            # YOLO runs on the original unflipped frame for best accuracy.
            # After detection, Y coordinates are flipped for presentation.
            results   = self._model(img, conf=_DET_CONF, verbose=False)
            raw_frame = img

            # ── collect detections (Y-flip applied to coordinates) ────────────
            detections: List[Tuple] = []
            for result in results:
                if result.keypoints is None or result.boxes is None:
                    continue
                boxes    = result.boxes.xyxy.cpu().numpy()
                kps      = result.keypoints.xy.cpu().numpy().copy()
                kp_confs = result.keypoints.conf
                if kp_confs is not None:
                    kp_confs = kp_confs.cpu().numpy()
                else:
                    kp_confs = np.ones((len(boxes), 17), dtype=np.float32)

                # Flip Y axis of all keypoints for presentation
                kps[:, :, 1] = self._frame_h - kps[:, :, 1]

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    box_h = max(float(y2 - y1), 1.0)
                    cx    = float((x1 + x2) / 2) / self._frame_w
                    cy    = 1.0 - float((y1 + y2) / 2) / self._frame_h   # flip Y
                    detections.append((cx, cy, kps[i], kp_confs[i], box_h))

            # ── match detections to existing person IDs ───────────────────────
            assigned: Dict[int, int] = {}
            used: set = set()

            for pid, (prev_cx, prev_cy) in list(self._tracked.items()):
                best_d, best_i = float('inf'), -1
                for i, (cx, cy, *_) in enumerate(detections):
                    if i in used:
                        continue
                    d = math.hypot(cx - prev_cx, cy - prev_cy)
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i >= 0 and best_d <= _MAX_MATCH_DIST:
                    assigned[pid] = best_i
                    used.add(best_i)

            for i in range(len(detections)):
                if i not in used:
                    pid = self._next_pid
                    self._next_pid += 1
                    assigned[pid] = i

            for pid in list(self._tracked):
                if pid not in assigned:
                    del self._tracked[pid]

            # ── emit fiducials ────────────────────────────────────────────────
            fiducials: List[Fiducial3D] = []
            for pid, det_i in assigned.items():
                cx, cy, kp, kp_conf, box_h = detections[det_i]
                self._tracked[pid] = (cx, cy)

                z_mm = float(np.clip(self._depth_scale / box_h, _Z_MIN, _Z_MAX))

                if kp_conf[_NOSE] >= _KP_CONF_MIN:
                    self._emit(fiducials, kp, kp_conf, _NOSE, pid, 0, z_mm)
                else:
                    for ear_kp in (_L_EAR, _R_EAR):
                        if kp_conf[ear_kp] >= _KP_CONF_MIN:
                            self._emit(fiducials, kp, kp_conf, ear_kp, pid, 0, z_mm)
                            break

                self._emit(fiducials, kp, kp_conf, _L_SHOULDER, pid, 1, z_mm)
                self._emit(fiducials, kp, kp_conf, _R_SHOULDER, pid, 2, z_mm)

            if self._debug:
                self._show_debug(results, raw_frame, assigned, detections)

            ts = int(time.perf_counter() * 1_000_000)
            self._frame_idx += 1
            frame = Frame(fiducials=fiducials, timestamp_us=ts,
                          frame_index=self._frame_idx)
            with self._lock:
                self._latest_frame = frame

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._debug:
            import cv2
            cv2.destroyWindow("fidart — RGB debug")
        print("[RGBCameraTracker] closed")

    # ── helpers ─────────────────────────────────────────────────────────────────

    def _to_mm(self, px: float, py: float) -> Tuple[float, float]:
        """Pixel coords → world mm.  Y is inverted (up = positive)."""
        x_mm = (px / self._frame_w - 0.5) * 2 * _X_HALF
        y_mm = -(py / self._frame_h - 0.5) * 2 * _Y_HALF
        return (float(np.clip(x_mm, -_X_HALF, _X_HALF)),
                float(np.clip(y_mm, -_Y_HALF, _Y_HALF)))

    def _show_debug(self, results, raw_frame, assigned: Dict, detections: List) -> None:
        """Draw skeleton + per-person info in a separate OpenCV window."""
        import cv2

        # YOLO's built-in skeleton + keypoint overlay
        vis = results[0].plot(conf=False, labels=False)

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1

        for pid, det_i in assigned.items():
            cx, cy, kp, kp_conf, box_h = detections[det_i]
            z_mm = float(np.clip(self._depth_scale / box_h, _Z_MIN, _Z_MAX))

            # Bounding box top in pixels
            bx = int(cx * self._frame_w)
            by = int(cy * self._frame_h) - int(box_h / 2) - 8

            # Which slots are active?
            slots = []
            head_visible = (kp_conf[_NOSE] >= _KP_CONF_MIN or
                            kp_conf[_L_EAR] >= _KP_CONF_MIN or
                            kp_conf[_R_EAR] >= _KP_CONF_MIN)
            if head_visible:
                slots.append('head')
            if kp_conf[_L_SHOULDER] >= _KP_CONF_MIN:
                slots.append('L.shoulder')
            if kp_conf[_R_SHOULDER] >= _KP_CONF_MIN:
                slots.append('R.shoulder')

            label = f"P{pid}  Z={z_mm:.0f}mm  [{', '.join(slots)}]"
            # Shadow then white text for readability
            cv2.putText(vis, label, (bx - 60, max(by, 14)),
                        font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(vis, label, (bx - 60, max(by, 14)),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Header bar
        n_people = len(assigned)
        n_fids   = sum(
            (1 if (kp_conf[_NOSE] >= _KP_CONF_MIN or
                   kp_conf[_L_EAR] >= _KP_CONF_MIN or
                   kp_conf[_R_EAR] >= _KP_CONF_MIN) else 0) +
            (1 if kp_conf[_L_SHOULDER] >= _KP_CONF_MIN else 0) +
            (1 if kp_conf[_R_SHOULDER] >= _KP_CONF_MIN else 0)
            for _, det_i in assigned.items()
            for _, _, _, kp_conf, _ in [detections[det_i]]
        )
        header = f"people: {n_people}   fiducials: {n_fids}   depth_scale: {self._depth_scale:.0f}"
        cv2.rectangle(vis, (0, 0), (self._frame_w, 22), (30, 30, 30), -1)
        cv2.putText(vis, header, (6, 15),
                    font, font_scale, (180, 255, 180), thickness, cv2.LINE_AA)

        cv2.imshow("fidart — RGB debug", vis)
        cv2.waitKey(1)

    def _emit(self, fiducials: List[Fiducial3D],
              kp: np.ndarray, kp_conf: np.ndarray,
              kp_idx: int, pid: int, slot: int, z_mm: float) -> None:
        """Append a Fiducial3D if keypoint confidence meets threshold."""
        if float(kp_conf[kp_idx]) < _KP_CONF_MIN:
            return
        xm, ym = self._to_mm(float(kp[kp_idx][0]), float(kp[kp_idx][1]))
        fiducials.append(Fiducial3D(
            x=xm, y=ym, z=z_mm,
            index=pid * 3 + slot,
            probability=float(kp_conf[kp_idx]),
        ))
