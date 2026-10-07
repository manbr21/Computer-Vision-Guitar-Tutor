"""Command line entry point: python -m chordar [--camera N] [--debug]."""
import argparse
import logging
import sys

import cv2

from chordar.app import ChordARApp


def main():
    parser = argparse.ArgumentParser(prog="chordar", description="Monitor-based AR guitar tutor.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--debug", action="store_true", help="log what the marker detector sees")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="[%(name)s] %(message)s")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    ChordARApp(capture).run()


if __name__ == "__main__":
    main()
