"""Command line entry point: python -m chordar [--camera N] [--no-audio] [--debug]."""
import argparse
import logging
import sys

import cv2

from chordar import config
from chordar.app import ChordARApp
from chordar.audio import StrumListener


def main():
    parser = argparse.ArgumentParser(prog="chordar", description="Monitor-based AR guitar tutor.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--no-audio", action="store_true", help="do not listen to the microphone for strums")
    parser.add_argument("--debug", action="store_true", help="log what the marker detector sees")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="[%(name)s] %(message)s")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    audio = None if args.no_audio or not config.AUDIO_ENABLED else StrumListener()
    ChordARApp(capture, audio).run()


if __name__ == "__main__":
    main()
