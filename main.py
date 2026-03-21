"""fidart — spryTrack 300 party visualizer.

Usage:
  python main.py                    # 6 simulated people
  python main.py -n 10              # 10 people
  python main.py --constellation    # start in CONSTELLATION mode
  python main.py --fullscreen       # start fullscreen (ideal for projection)
  python main.py --fps 60           # simulation frame rate
  python main.py --seed 42          # reproducible simulation
  python main.py --live             # real spryTrack 300 camera
  python main.py --serial 12345678  # specific camera serial number
"""
import argparse

from tracker.simulator import SimulatedTracker
from tracker.sdk import SDKTracker
from visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="fidart — IR fiducial party visuals (spryTrack 300)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--live",          action="store_true",
                   help="Use real spryTrack 300 camera")
    p.add_argument("-n", "--n-fiducials", type=int, default=6, metavar="N",
                   help="Simulated party-goers (default: 6)")
    p.add_argument("--constellation", action="store_true",
                   help="Start in CONSTELLATION mode (M toggles live)")
    p.add_argument("--fps",           type=float, default=30.0)
    p.add_argument("--seed",          type=int, default=None)
    p.add_argument("--serial",        type=int, default=None, metavar="SN")
    p.add_argument("--fullscreen",    action="store_true")
    p.add_argument("--width",         type=int, default=1280)
    p.add_argument("--height",        type=int, default=720)
    p.add_argument("--no-overlays",   action="store_true",
                   help="Disable halos + aurora ribbons")
    p.add_argument("--no-audio",      action="store_true",
                   help="Disable microphone input")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.live:
        tracker  = SDKTracker(serial_number=args.serial)
        mode_str = f"LIVE  SN={args.serial or 'auto'}"
    else:
        tracker  = SimulatedTracker(n_fiducials=args.n_fiducials,
                                    fps=args.fps, seed=args.seed)
        mode_str = f"SIM  n={args.n_fiducials}  fps={args.fps}"

    vis = Visualizer(
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        show_overlays=not args.no_overlays,
        audio=not args.no_audio,
    )
    if args.constellation:
        vis._constellation = True

    print(f"fidart  |  {mode_str}")
    print("Controls: M=constellation   G=overlays   F=fullscreen   Q=quit")

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
