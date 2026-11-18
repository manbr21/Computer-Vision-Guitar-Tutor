"""Command line entry point: python -m chordar [--camera N] [--no-audio] [--record FOLDER] [--reference FILE] [--markers]."""
import argparse
import logging
import sys

import cv2

from chordar import config
from chordar.app import ChordARApp
from chordar.audio import StrumListener
from chordar.markerless import MarklessTracker, NeckReference
from chordar.recorder import SessionRecorder
from chordar.reference import adopt_placement


def main():
    parser = argparse.ArgumentParser(prog="chordar", description="Monitor-based AR guitar tutor. With no options it "
                                     "takes a photo of your guitar when it starts and finds the neck from it.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--no-audio", action="store_true", help="do not listen to the microphone for strums")
    parser.add_argument("--reference", metavar="FILE", help="skip the photo and use a saved one (references/last.npz is "
                        "saved at every start)")
    parser.add_argument("--markers", action="store_true", help="track the neck with four ArUco markers instead of a photo")
    parser.add_argument("--record", metavar="FOLDER", help="save the camera video and the detected neck corners of this session under FOLDER")
    parser.add_argument("--window-scale", type=float, default=config.WINDOW_SCALE, metavar="X",
                        help="size of the windows relative to the camera picture (default %(default)s; 1.0 for a small screen)")
    parser.add_argument("--debug", action="store_true", help="log what the detectors see")
    args = parser.parse_args()
    config.WINDOW_SCALE = args.window_scale
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="[%(name)s] %(message)s")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    audio = None if args.no_audio or not config.AUDIO_ENABLED else StrumListener()
    recorder = SessionRecorder(args.record) if args.record else None
    tracker = None
    if args.reference and not args.markers:
        reference = NeckReference.load(args.reference)
        height, width = reference.image.shape[:2]
        adopt_placement([(width - x, y) for x, y in reference.corners], (width, height))
        tracker = MarklessTracker(reference)
    ChordARApp(capture, audio, recorder, tracker, markerless=not args.markers).run()


if __name__ == "__main__":
    main()
