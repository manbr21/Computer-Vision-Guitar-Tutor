"""The ChordAR application: per-frame processing, overlay and keyboard control."""
import logging
import time

import cv2

from chordar.calibration import run_calibration
from chordar.chords import fretted_notes
from chordar.hands import FingertipTracker
from chordar.markers import MarkerTracker
from chordar.overlay import (draw_chord_diagram, draw_chord_instructions, draw_chord_targets, draw_fingertips,
                             draw_neck, put_text_bg)
from chordar.placement import PlacementMonitor
from chordar.scoring import chord_accuracy

log = logging.getLogger(__name__)

WINDOW = "ChordAR"
CHORD_KEYS = {ord(key): chord for key, chord in zip("12345678", ["A", "Am", "C", "D", "Dm", "E", "Em", "G"])}
KEY_HELP = "Press 1-8 to change chords, P parallax fix, X swap pose, C recalibrate, ESC to exit"
WHITE, GREEN, ORANGE, RED, GRAY = (255, 255, 255), (0, 255, 0), (0, 165, 255), (0, 0, 255), (200, 200, 200)


class ChordARApp:
    def __init__(self, capture):
        self.capture = capture
        self.markers = MarkerTracker()
        self.hands = FingertipTracker()
        self.monitor = PlacementMonitor()
        self.chord = "C"
        self.parallax_on = True  # P toggles the fingertip parallax correction
        self.pose_variant = 0    # X swaps between the two possible camera-pose solutions

    def run(self):
        """Calibrates, then shows the video with the overlay until ESC. Releases the camera at the end."""
        try:
            if not run_calibration(self.capture, self.markers, self.hands, WINDOW):
                return
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    log.warning("Camera stopped delivering frames")
                    break
                cv2.imshow(WINDOW, self.process_frame(frame, time.time()))
                if not self.handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            self.capture.release()
            self.hands.close()
            cv2.destroyAllWindows()

    def process_frame(self, frame, now):
        """Detects the neck and fingers in a camera frame and returns the image to show. now: frame start time."""
        display, neck = self.markers.track(frame, self.pose_variant)
        fingertips = self.hands.track(frame, neck.to_image(0.5, 0.2) if neck is not None else None)
        size = (display.shape[1], display.shape[0])

        score = None
        if neck is not None:
            draw_neck(display, neck)
            draw_chord_targets(display, neck, self.chord)
            draw_fingertips(display, neck, fingertips, self.parallax_on)
            score = chord_accuracy(fretted_notes(self.chord), fingertips, neck, self.parallax_on)

        self._draw_status(display, neck, self.monitor.update(neck, size, now))
        self._draw_score(display, score)
        self._draw_chord_panel(display)
        put_text_bg(display, KEY_HELP, (20, size[1] - 20), 0.5, WHITE, thickness=1)
        fps = 1.0 / (time.time() - now) if time.time() != now else 0.0
        put_text_bg(display, f"FPS: {fps:.1f}", (size[0] - 130, size[1] - 20), 0.7, GREEN)
        return display

    def handle_key(self, key):
        """Applies a key press; returns False to quit."""
        if key == 27:  # ESC
            return False
        if key in CHORD_KEYS:
            self.chord = CHORD_KEYS[key]
        elif key == ord("p"):
            self.parallax_on = not self.parallax_on
        elif key == ord("x"):
            self.pose_variant = 1 - self.pose_variant
        elif key == ord("c"):
            run_calibration(self.capture, self.markers, self.hands, WINDOW)  # ESC there comes back to playing
            self.monitor.reset()
        return True

    def _draw_status(self, display, neck, drift_hint):
        if neck is None:
            put_text_bg(display, "ArUco markers not detected (IDs 0-3)", (20, 90), 0.6, RED)
            return
        status = f"Parallax fix: {'ON' if self.parallax_on else 'OFF'} (pose {'AB'[self.pose_variant]})"
        angles = neck.camera_angles()
        if angles is not None:
            status += f" | camera angle {angles[0]:.0f} deg"
        put_text_bg(display, status, (20, 120), 0.5, WHITE, thickness=1)
        if drift_hint is not None:
            put_text_bg(display, f"{drift_hint} - press C to recalibrate", (20, 150), 0.55, ORANGE, thickness=1)

    def _draw_score(self, display, score):
        put_text_bg(display, f"Current Chord: {self.chord}", (20, 30), 0.7, WHITE)
        if score is None:  # no neck, or a chord with no fretted notes
            put_text_bg(display, "Accuracy: N/A", (20, 60), 0.8, GRAY)
        else:
            put_text_bg(display, f"Accuracy: {score}%", (20, 60), 0.8, GREEN if score == 100 else ORANGE)

    def _draw_chord_panel(self, display):
        x, y, width, height = display.shape[1] - 220, 20, 200, 190
        draw_chord_diagram(display, x, y, width, height, self.chord)
        draw_chord_instructions(display, x, y + height + 10, width, self.chord)
