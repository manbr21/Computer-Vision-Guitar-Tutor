"""The ChordAR application: per-frame processing, overlay and keyboard control."""
import logging
import time

import cv2

from chordar.calibration import run_calibration
from chordar.chords import MAX_CAPO, fretted_notes
from chordar.hands import FingertipTracker
from chordar.markers import MarkerTracker
from chordar.overlay import (draw_capo, draw_chord_diagram, draw_chord_instructions, draw_chord_targets,
                             draw_fingertips, draw_neck, put_text_bg)
from chordar.placement import PlacementMonitor
from chordar.scoring import chord_accuracy

log = logging.getLogger(__name__)

WINDOW = "ChordAR"
CHORD_KEYS = {ord(key): chord for key, chord in zip("12345678", ["A", "Am", "C", "D", "Dm", "E", "Em", "G"])}
KEY_HELP = ("1-8 chord, [ ] capo, A auto-advance on strum,", "P parallax, X pose, C recalibrate, ESC exit")
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

    def run(self):
        """Calibrates, then shows the video with the overlay until ESC. Releases the camera at the end."""
        if self.audio is not None:
            self.audio.start()
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

        score = None
        if neck is not None:
            notes = fretted_notes(self.chord, self.capo)
            draw_neck(display, neck)
            if self.capo:
                draw_capo(display, neck, self.capo)
            draw_chord_targets(display, neck, notes)
            draw_fingertips(display, neck, fingertips, self.parallax_on)
            score = chord_accuracy(notes, fingertips, neck, self.parallax_on)

        self._handle_strums(score, now)
        self._draw_status(display, neck, self.monitor.update(neck, size, now))
        self._draw_score(display, score)
        self._draw_strum(display, now)
        self._draw_chord_panel(display)
        for i, line in enumerate(reversed(KEY_HELP)):  # two lines so it fits narrow webcam frames
            put_text_bg(display, line, (20, size[1] - 20 - 26 * i), 0.5, WHITE, thickness=1)
        fps = 1.0 / (time.time() - now) if time.time() != now else 0.0
        put_text_bg(display, f"FPS: {fps:.1f}", (size[0] - 130, size[1] - 20), 0.7, GREEN)
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
        """The result of the latest strum, and a small microphone level bar."""
        if self.audio is None:
            return
        if self.last_strum is not None and now - self.last_strum[0] < STRUM_BANNER_S:
            _, chord, capo, score = self.last_strum
            name = chord + (f" capo {capo}" if capo else "")
            if score is None:
                put_text_bg(display, f"Strum! {name}: neck not seen", (20, 190), 0.7, GRAY)
            else:
                put_text_bg(display, f"Strum! {name}: {score}%" + ("  correct" if score == 100 else ""),
                            (20, 190), 0.7, GREEN if score == 100 else ORANGE)
        if self.auto_advance:
            put_text_bg(display, "Auto-advance ON: strum a perfect chord to move on", (20, 245), 0.5, GREEN, thickness=1)
        bar = int(min(self.audio.level / 0.2, 1.0) * 120)  # full bar at an RMS of 0.2
        cv2.rectangle(display, (20, 205), (140, 215), (60, 60, 60), -1)
        cv2.rectangle(display, (20, 205), (20 + bar, 215), GREEN, -1)
        put_text_bg(display, "mic" if self.audio.active else "mic off", (150, 215), 0.45, GRAY, thickness=1, pad=3)

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
        title = f"Current Chord: {self.chord}" + (f" | Capo {self.capo}" if self.capo else "")
        if self.audio is not None:
            title += " | AUTO" if self.auto_advance else ""
        put_text_bg(display, title, (20, 30), 0.7, WHITE)
        if score is None:  # no neck, or a chord with no fretted notes
            put_text_bg(display, "Accuracy: N/A", (20, 60), 0.8, GRAY)
        else:
            put_text_bg(display, f"Accuracy: {score}%", (20, 60), 0.8, GREEN if score == 100 else ORANGE)

    def _draw_chord_panel(self, display):
        x, y, width, height = display.shape[1] - 220, 20, 200, 190
        draw_chord_diagram(display, x, y, width, height, self.chord, self.capo)
        draw_chord_instructions(display, x, y + height + 10, width, self.chord, self.capo)
