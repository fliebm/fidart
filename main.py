"""fidart — spryTrack 300 party visualizer.

Usage:
  python main.py                    # simulated crowd (default)
  python main.py -n 10              # 10 starting people
  python main.py --live             # real spryTrack 300 IR camera
  python main.py --rgb              # webcam + YOLO pose tracking
  python main.py --rgb --camera 1   # second webcam
  python main.py --constellation    # start in CONSTELLATION mode
  python main.py --fullscreen       # start fullscreen (ideal for projection)
  python main.py --fps 60           # simulation frame rate
  python main.py --seed 42          # reproducible simulation
  python main.py --serial 12345678  # specific spryTrack serial number
"""
import argparse

from tracker.simulator import SimulatedTracker
from tracker.sdk import SDKTracker
from tracker.rgb_camera import RGBCameraTracker
from visualizer import Visualizer
from audio import find_loopback_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="fidart — IR fiducial party visuals (spryTrack 300)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--live",          action="store_true",
                   help="Use real spryTrack 300 IR camera")
    p.add_argument("--rgb",           action="store_true",
                   help="Use webcam + YOLO pose tracking (pip install ultralytics opencv-python)")
    p.add_argument("--camera",        type=int, default=0, metavar="IDX",
                   help="Webcam index for --rgb mode (default: 0)")
    p.add_argument("--depth-scale",   type=float, default=250000.0, metavar="D",
                   help="Depth tuning for --rgb: Z ≈ D / box_height_px (default: 250000)")
    p.add_argument("--debug",         action="store_true",
                   help="Show separate OpenCV debug window (--rgb only)")
    p.add_argument("-n", "--n-fiducials", type=int, default=6, metavar="N",
                   help="Simulated party-goers (default: 6)")
    p.add_argument("--constellation", action="store_true",
                   help="Start in CONSTELLATION mode (M toggles live)")
    p.add_argument("--fps",           type=float, default=30.0)
    p.add_argument("--seed",          type=int, default=None)
    p.add_argument("--serial",        type=int, default=None, metavar="SN")
    p.add_argument("--fullscreen",    action="store_true")
    p.add_argument("--maximize",      action="store_true",
                   help="Launch as maximized window (default on)")
    p.add_argument("--width",         type=int, default=1280)
    p.add_argument("--height",        type=int, default=720)
    p.add_argument("--no-overlays",   action="store_true",
                   help="Disable halos + aurora ribbons")
    p.add_argument("--no-audio",      action="store_true",
                   help="Disable audio input")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.live:
        tracker  = SDKTracker(serial_number=args.serial)
        mode_str = f"LIVE  SN={args.serial or 'auto'}"
    elif args.rgb:
        tracker  = RGBCameraTracker(camera_index=args.camera,
                                    fps=args.fps,
                                    depth_scale=args.depth_scale,
                                    debug=args.debug)
        mode_str = f"RGB  camera={args.camera}  fps={args.fps}"
    else:
        tracker  = SimulatedTracker(n_fiducials=args.n_fiducials,
                                    fps=args.fps, seed=args.seed)
        mode_str = f"SIM  n={args.n_fiducials}  fps={args.fps}"

    audio_device = None
    use_loopback = False
    if not args.no_audio:
        audio_device = find_loopback_device()
        if audio_device is not None:
            use_loopback = True
        else:
            print("[Audio] pyaudiowpatch not installed — falling back to microphone\n"
                  "        Install with:  pip install pyaudiowpatch")

    vis = Visualizer(
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        maximize=args.maximize,
        show_overlays=not args.no_overlays,
        audio=not args.no_audio,
        audio_device=audio_device,
        audio_loopback=use_loopback,
    )
    if args.constellation:
        vis._constellation = True

    if not args.live and not args.rgb:
        vis.on_person_add    = tracker.add_person
        vis.on_person_remove = tracker.remove_person

    print(f"fidart  |  {mode_str}")
    print("Controls: M=constellation   G=overlays   F=fullscreen   UP/DOWN=people   Q=quit")

    try:
        with tracker:
            vis.open()
            while True:
                frame = tracker.get_frame()
                if not vis.update(frame):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        vis.close()


if __name__ == "__main__":
    main()
