"""Command line entry point: python -m chordar [--camera N] [--no-audio] [--record FOLDER] [--reference FILE] [--debug]."""
import argparse
import logging
import sys

import cv2

from chordar import config
from chordar.app import ChordARApp
from chordar.audio import StrumListener
from chordar.markerless import MarklessTracker, NeckReference
from chordar.recorder import SessionRecorder


def main():
    parser = argparse.ArgumentParser(prog="chordar", description="Monitor-based AR guitar tutor.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--no-audio", action="store_true", help="do not listen to the microphone for strums")
    parser.add_argument("--reference", metavar="FILE", help="track the neck without markers, from a reference made with python -m chordar.reference")
    parser.add_argument("--record", metavar="FOLDER", help="save the camera video and the detected neck corners of this session under FOLDER")
    parser.add_argument("--debug", action="store_true", help="log what the marker detector sees")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="[%(name)s] %(message)s")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    audio = None if args.no_audio or not config.AUDIO_ENABLED else StrumListener()
    recorder = SessionRecorder(args.record) if args.record else None
    tracker = None
    if args.reference:
        tracker = MarklessTracker(NeckReference.load(args.reference))
    ChordARApp(capture, audio, recorder, tracker).run()


if __name__ == "__main__":
    main()
