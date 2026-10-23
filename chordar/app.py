"""The ChordAR application: per-frame processing, overlay and keyboard control."""
import logging
import time

import cv2

from chordar.calibration import run_calibration
from chordar.chords import MAX_CAPO, fretted_notes
from chordar.hands import FingertipTracker
from chordar.markers import MarkerTracker
from chordar.coaching import coach
from chordar.overlay import (draw_arrow, draw_banner, draw_capo, draw_chord_diagram, draw_chord_targets,
                             draw_fingertips, draw_gauge, draw_neck, put_text_bg)
from chordar.placement import PlacementMonitor
from chordar.scoring import chord_accuracy
from chordar.window import open_window, show

log = logging.getLogger(__name__)

WINDOW = "ChordAR"
CHORD_KEYS = {ord(key): chord for key, chord in zip("12345678", ["A", "Am", "C", "D", "Dm", "E", "Em", "G"])}
HELP_LINES = ["1-8   choose the chord", "[ ]   capo up / down (0 removes it)", "A     auto-advance on a perfect strum",
              "C     recalibrate", "P     parallax fix on/off", "X     swap camera pose",
              "D     technical info", "H     this help", "ESC   quit"]
STRUM_BANNER_S = 2.0  # how long the result of a strum stays on screen
WHITE, GREEN, ORANGE, RED, GRAY = (255, 255, 255), (0, 255, 0), (0, 165, 255), (0, 0, 255), (200, 200, 200)


class ChordARApp:
    def __init__(self, capture, audio=None):
        """audio: a StrumListener (or anything with start/poll/level/stop), or None to play without sound."""
        self.capture = capture
        self.audio = audio
        self.markers = MarkerTracker()
        self.hands = FingertipTracker()
        self.monitor = PlacementMonitor()
        self.chord = "C"
        self.capo = 0            # capo fret: ] raises it, [ lowers it, 0 removes it
        self.parallax_on = True  # P toggles the fingertip parallax correction
        self.pose_variant = 0    # X swaps between the two possible camera-pose solutions
        self.auto_advance = False  # A: a correct strum moves on to the next chord
        self.last_strum = None   # (time, chord, capo, score or None) of the latest strum
        self.show_help = False   # H: list of the keys
        self.debug_info = False  # D: technical details (pose, parallax, fingertip labels, FPS, mic level)

    def run(self):
        """Calibrates, then shows the video with the overlay until ESC. Releases the camera at the end."""
        if self.audio is not None:
            self.audio.start()
        open_window(WINDOW)
        try:
            if not run_calibration(self.capture, self.markers, self.hands, WINDOW):
                return
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    log.warning("Camera stopped delivering frames")
                    break
                show(WINDOW, self.process_frame(frame, time.time()))
                if not self.handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            if self.audio is not None:
                self.audio.stop()
            self.capture.release()
            self.hands.close()
            cv2.destroyAllWindows()

    def process_frame(self, frame, now):
        """Detects the neck and fingers in a camera frame and returns the image to show. now: frame start time."""
        display, neck = self.markers.track(frame, self.pose_variant)
        fingertips = self.hands.track(frame, neck.to_image(0.5, 0.2) if neck is not None else None)
        size = (display.shape[1], display.shape[0])

        score, coaching = None, None
        if neck is not None:
            notes = fretted_notes(self.chord, self.capo)
            coaching = coach(notes, fingertips, neck, self.parallax_on)
            draw_neck(display, neck)
            if self.capo:
                draw_capo(display, neck, self.capo)
            draw_chord_targets(display, neck, notes, coaching.done)
            if coaching.fix is not None:
                note, pixel = coaching.fix
                draw_arrow(display, pixel, neck.target(note[0], note[1]))
            draw_fingertips(display, neck, fingertips, self.parallax_on, self.debug_info)
            score = chord_accuracy(notes, fingertips, neck, self.parallax_on)

        self._handle_strums(score, now)
        drift_hint = self.monitor.update(neck, size, now)
        self._draw_header(display, score)
        self._draw_message(display, neck, coaching, drift_hint)
        self._draw_chord_panel(display)
        self._draw_strum(display, now)
        if self.debug_info:
            self._draw_debug(display, neck, now)
        if self.show_help:
            self._draw_help(display)
        else:
            put_text_bg(display, "H: help", (20, size[1] - 20), 0.5, GRAY, thickness=1)
        return display

    def handle_key(self, key):
        """Applies a key press; returns False to quit."""
        if key == 27:  # ESC
            return False
        if key in CHORD_KEYS:
            self.chord = CHORD_KEYS[key]
        elif key == ord("]"):
            self.capo = min(self.capo + 1, MAX_CAPO)
        elif key == ord("["):
            self.capo = max(self.capo - 1, 0)
        elif key == ord("0"):
            self.capo = 0
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("d"):
            self.debug_info = not self.debug_info
        elif key == ord("a"):
            self.auto_advance = not self.auto_advance
        elif key == ord("p"):
            self.parallax_on = not self.parallax_on
        elif key == ord("x"):
            self.pose_variant = 1 - self.pose_variant
        elif key == ord("c"):
            run_calibration(self.capture, self.markers, self.hands, WINDOW)  # ESC there comes back to playing
            self.monitor.reset()
        return True

    def _handle_strums(self, score, now):
        """A strum judges the chord at that moment; with auto-advance, a perfect one moves to the next chord."""
        if self.audio is None or not self.audio.poll():
            return
        self.last_strum = (now, self.chord, self.capo, score)
        if self.auto_advance and score == 100:
            chords = list(CHORD_KEYS.values())
            self.chord = chords[(chords.index(self.chord) + 1) % len(chords)]

    def _draw_strum(self, display, now):
        """The result of the latest strum, just under the header."""
        if self.audio is None or self.last_strum is None or now - self.last_strum[0] >= STRUM_BANNER_S:
            return
        _, chord, capo, score = self.last_strum
        if score is None:
            put_text_bg(display, "Strum heard (neck not seen)", (20, 120), 0.7, GRAY)
        else:
            put_text_bg(display, f"Strum! {score}%" + ("  correct" if score == 100 else ""), (20, 120), 0.7,
                        GREEN if score == 100 else ORANGE)

    def _draw_header(self, display, score):
        """Accuracy ring, big chord name, and a small AUTO tag when a perfect strum moves on by itself."""
        draw_gauge(display, (52, 52), 34, score)
        name = self.chord + (f"  capo {self.capo}" if self.capo else "")
        put_text_bg(display, name, (100, 62), 1.2, WHITE, thickness=3)
        if self.audio is not None and self.auto_advance:
            put_text_bg(display, "AUTO", (100, 92), 0.5, GREEN, thickness=1)

    def _draw_message(self, display, neck, coaching, drift_hint):
        """The one thing to do now, at the bottom."""
        if neck is None:
            draw_banner(display, "Show all 4 markers to the camera", RED)
        elif drift_hint is not None:
            draw_banner(display, f"{drift_hint} (C: recalibrate)", ORANGE)
        elif coaching is not None:
            text = coaching.message
            if text == "Perfect!" and self.audio is not None and self.audio.active:
                text += " Now strum it"
            draw_banner(display, text, GREEN if coaching.perfect else WHITE)

    def _draw_chord_panel(self, display):
        x, y, width, height = display.shape[1] - 220, 20, 200, 190
        draw_chord_diagram(display, x, y, width, height, self.chord, self.capo)

    def _draw_help(self, display):
        pad, line_h = 12, 24
        width = 330
        height = len(HELP_LINES) * line_h + 2 * pad
        x, y = 20, display.shape[0] - height - 20
        shade = display.copy()
        cv2.rectangle(shade, (x, y), (x + width, y + height), (30, 30, 30), -1)
        cv2.addWeighted(shade, 0.8, display, 0.2, 0, dst=display)
        for i, line in enumerate(HELP_LINES):
            cv2.putText(display, line, (x + pad, y + pad + 16 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    def _draw_debug(self, display, neck, now):
        """Technical details for tuning: parallax and pose, camera angle, FPS and the microphone level."""
        size = (display.shape[1], display.shape[0])
        if neck is not None:
            status = f"Parallax fix: {'ON' if self.parallax_on else 'OFF'} (pose {'AB'[self.pose_variant]})"
            angles = neck.camera_angles()
            if angles is not None:
                status += f" | camera angle {angles[0]:.0f} deg"
            put_text_bg(display, status, (20, 160), 0.5, WHITE, thickness=1)
        fps = 1.0 / (time.time() - now) if time.time() != now else 0.0
        put_text_bg(display, f"FPS: {fps:.1f}", (size[0] - 130, size[1] - 20), 0.7, GREEN)
        if self.audio is not None:
            bar = int(min(self.audio.level / 0.2, 1.0) * 120)  # full bar at an RMS of 0.2
            cv2.rectangle(display, (20, 175), (140, 185), (60, 60, 60), -1)
            cv2.rectangle(display, (20, 175), (20 + bar, 185), GREEN, -1)
            put_text_bg(display, "mic" if self.audio.active else "mic off", (150, 185), 0.45, GRAY, thickness=1, pad=3)
