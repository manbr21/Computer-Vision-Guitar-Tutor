"""The ChordAR application: per-frame processing, overlay and keyboard control."""
import logging
import time

import cv2

from chordar.calibration import run_calibration
from chordar.chords import MAX_CAPO, fretted_notes, reading_bias
from chordar.hands import FingertipTracker, fretting_hand_region
from chordar.markerless import MarklessTracker
from chordar.markers import MarkerTracker
from chordar.coaching import coach
from chordar.overlay import (draw_arrow, draw_banner, draw_capo, draw_chord_diagram, draw_chord_targets, target_radius,
                             draw_fingertips, draw_gauge, draw_neck, put_text_bg)
from chordar.placement import PlacementMonitor
from chordar.reference import capture_reference
from chordar.scoring import NoteStabilizer, corrected_points
from chordar import ui
from chordar.zoom import NeckZoom
from chordar.window import open_window, show

log = logging.getLogger(__name__)

WINDOW = "ChordAR"
CHORD_KEYS = {ord(key): chord for key, chord in zip("12345678", ["A", "Am", "C", "D", "Dm", "E", "Em", "G"])}
CHORD_NAMES = ["A", "Am", "C", "D", "Dm", "E", "Em", "G"]
# the key bar always on the screen, and the sections of the help panel (H): (title, [(key, what it does)])
KEY_BAR = [("1-8", "Chord"), ("[ ]", "Capo"), ("Z", "Zoom"), ("H", "All keys")]
HELP_SECTIONS = [("CAPO", [("[", "capo down"), ("]", "capo up (max %d)" % MAX_CAPO), ("0", "no capo")]),
                 ("VIEW", [("Z", "zoom on the neck"), ("O", "neck lines on/off"), ("D", "technical info")]),
                 ("PLAY", [("A", "auto-advance on strum")]),
                 ("SET UP", [("C", "new photo"), ("H", "this help"), ("ESC", "quit")])]
# keys that are not listed in the help, for tuning and debugging: D shows the technical panel, which lists P and X
LAST_REFERENCE = "references/last.npz"  # the photo of the latest set-up, to reuse with --reference
STRUM_BANNER_S = 2.0  # how long the result of a strum stays on screen
WHITE, GREEN, ORANGE, RED, GRAY = (255, 255, 255), (0, 255, 0), (0, 165, 255), (0, 0, 255), (200, 200, 200)


class ChordARApp:
    def __init__(self, capture, audio=None, recorder=None, tracker=None, markerless=False):
        """
        audio: a StrumListener (or anything with start/poll/level/stop), or None to play without sound.
        recorder: a SessionRecorder to save the session, or None. tracker: the neck tracker (default: ArUco markers).
        markerless: no markers: the neck is found from a photo taken when the app starts (and again with C), unless
        a tracker built from a saved photo is given.
        """
        self.capture = capture
        self.audio = audio
        self.recorder = recorder
        self.markerless = markerless or isinstance(tracker, MarklessTracker)
        self.markers = tracker or (None if markerless else MarkerTracker())
        self.hands = FingertipTracker()
        self.stabilizer = NoteStabilizer()   # steadies which notes are shown as covered
        self.monitor = PlacementMonitor()
        self.chord = "C"
        self.capo = 0            # capo fret: ] raises it, [ lowers it, 0 removes it
        self.parallax_on = False  # P toggles the fingertip parallax correction (off: its sign was wrong at some camera heights)
        self.pose_variant = 0    # X swaps between the two possible camera-pose solutions
        self.auto_advance = False  # A: a correct strum moves on to the next chord
        self.last_score = None   # the accuracy shown in the latest frame, for the recorder
        self.last_strum = None   # (time, chord, capo, score or None) of the latest strum
        self.show_neck_overlay = True  # O: the markers, outline, frets and strings; off leaves only fingers and targets
        self.zoom_view = False   # Z: the screen shows the neck and the hand enlarged (the detection still sees the whole frame)
        self.zoom = NeckZoom()
        self.show_help = False   # H: list of the keys
        self.debug_info = False  # D: technical details (pose, parallax, fingertip labels, FPS, mic level)

    def run(self):
        """Calibrates, then shows the video with the overlay until ESC. Releases the camera at the end."""
        if self.audio is not None:
            self.audio.start()
        open_window(WINDOW)
        try:
            if not self._set_up():
                return
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    log.warning("Camera stopped delivering frames")
                    break
                now = time.time()
                show(WINDOW, self.process_frame(frame, now))
                if self.recorder is not None:
                    self.recorder.write(frame, self.markers, now,
                                        {"chord": self.chord, "capo": self.capo, "score": self.last_score})
                if not self.handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            if self.audio is not None:
                self.audio.stop()
            if self.recorder is not None:
                self.recorder.close()
            self.capture.release()
            self.hands.close()
            cv2.destroyAllWindows()

    def _set_up(self):
        """The calibration: the guided photo (marker-free) or the calibration screen (markers). False if cancelled."""
        if not self.markerless:
            return run_calibration(self.capture, self.markers, self.hands, WINDOW)
        if self.markers is not None:   # built from a saved reference: no photo
            return True
        self.markers = capture_reference(self.capture, save_to=LAST_REFERENCE)
        return self.markers is not None

    def process_frame(self, frame, now):
        """Detects the neck and fingers in a camera frame and returns the image to show. now: frame start time."""
        display, neck = self.markers.track(frame, self.pose_variant, draw=self.show_neck_overlay)
        region = fretting_hand_region(neck, (display.shape[1], display.shape[0]), self.capo) if neck is not None else None
        fingertips = self.hands.track(frame, neck.to_image(0.5, 0.2) if neck is not None else None, region)
        size = (display.shape[1], display.shape[0])

        score, coaching = None, None
        if neck is not None:
            notes = fretted_notes(self.chord, self.capo)
            coaching = coach(notes, fingertips, neck, self.parallax_on, self.stabilizer, reading_bias(self.chord, self.capo))
            if self.show_neck_overlay:
                draw_neck(display, neck)
            if self.capo:
                draw_capo(display, neck, self.capo)
            highlight = notes.index(coaching.fix[0]) if coaching.fix is not None else None
            draw_chord_targets(display, neck, notes, coaching.done, highlight, now)
            if coaching.fix is not None:
                note, pixel = coaching.fix
                draw_arrow(display, pixel, neck.target(note[0], note[1]), stop_short=target_radius(neck) + 6, now=now)
            touch_points = corrected_points(notes, fingertips, neck, self.parallax_on, reading_bias(self.chord, self.capo))
            draw_fingertips(display, neck, fingertips, self.parallax_on, self.debug_info, touch_points)
            score = int(100 * sum(coaching.done) / len(notes)) if notes else None
        else:
            self.stabilizer.reset()

        if self.zoom_view and neck is not None:
            display = self.zoom.apply(display, neck, self.capo)   # after the drawing in the neck's coordinates, before the text on the screen
        self.last_score = score
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
            self._draw_key_bar(display)
        return display

    def handle_key(self, key):
        """Applies a key press; returns False to quit."""
        if key == 27:  # ESC
            return False
        if key in CHORD_KEYS:
            self.chord = CHORD_KEYS[key]
            self.stabilizer.reset()
        elif key == ord("]"):
            self.capo = min(self.capo + 1, MAX_CAPO)
            self.stabilizer.reset()
        elif key == ord("["):
            self.capo = max(self.capo - 1, 0)
            self.stabilizer.reset()
        elif key == ord("0"):
            self.capo = 0
            self.stabilizer.reset()
        elif key == ord("o"):
            self.show_neck_overlay = not self.show_neck_overlay
        elif key == ord("z"):
            self.zoom_view = not self.zoom_view
            self.zoom.reset()
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
            if self.markerless:   # a new photo; cancelling it keeps the current one
                self.markers = capture_reference(self.capture, save_to=LAST_REFERENCE) or self.markers
            else:
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
            self.stabilizer.reset()

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
            draw_banner(display, "Neck not found: hold it in the outline of the photo (C: new photo)" if self.markerless
                        else "Show all 4 markers to the camera", RED)
        elif self.markerless and not getattr(self.markers, "frets_visible", True):
            draw_banner(display, "Frets hard to see: put more light on the neck (C: new photo)", ORANGE)
        elif drift_hint is not None:
            draw_banner(display, f"{drift_hint} (C: " + ("new photo)" if self.markerless else "recalibrate)"), ORANGE)
        elif coaching is not None:
            text = coaching.message
            if text == "Perfect!" and self.audio is not None and self.audio.active:
                text += " Now strum it"
            draw_banner(display, text, GREEN if coaching.perfect else WHITE)

    def _draw_chord_panel(self, display):
        x, y, width, height = display.shape[1] - 220, 20, 200, 190
        draw_chord_diagram(display, x, y, width, height, self.chord, self.capo)

    def _draw_help(self, display):
        """The keys, by what they do, in a panel at the bottom left where the key bar is (H closes it): the chords in a row, then one column per topic."""
        h = display.shape[0]
        width, row = 1000, 29
        height = 34 + 50 + 22 + 3 * row + 12
        x, y = 12, h - 10 - height
        ui.panel(display, x, y, x + width, y + height, alpha=0.86, radius=14, border=ui.ACCENT_DIM)
        ui.text(display, "KEYS", (x + 18, y + 26), 0.6, ui.ACCENT, 1, shadow=False)
        ui.text(display, "Press H to close", (x + width - 150, y + 26), 0.45, ui.MUTED, 1, shadow=False)
        ui.text(display, "CHORD", (x + 18, y + 54), 0.45, ui.MUTED, 1, shadow=False)
        for i, name in enumerate(CHORD_NAMES):          # 1 A  2 Am  3 C ...
            cx = x + 90 + i * 110
            end = ui.key_cap(display, str(i + 1), cx, y + 36, 0.5)
            ui.text(display, name, (end + 8, y + 56), 0.55, ui.TEXT, 1, shadow=False)
        for column, (title, items) in enumerate(HELP_SECTIONS):
            cx, cy = x + 18 + column * 245, y + 34 + 50
            ui.text(display, title, (cx, cy + 4), 0.45, ui.MUTED, 1, shadow=False)
            for k, (key, what) in enumerate(items):
                end = ui.key_cap(display, key, cx, cy + 12 + k * row, 0.5)
                ui.text(display, what, (end + 10, cy + 12 + k * row + 19), 0.5, ui.TEXT, 1, shadow=False)

    def _draw_key_bar(self, display):
        """The few keys that matter, always at the bottom left."""
        h = display.shape[0]
        ui.panel(display, 12, h - 56, 12 + ui.row_width(KEY_BAR) + 24, h - 10, alpha=0.62, radius=10)
        ui.key_row(display, KEY_BAR, 24, h - 48)

    def _draw_debug(self, display, neck, now):
        """Technical details for tuning: parallax and pose, camera angle, FPS and the microphone level."""
        size = (display.shape[1], display.shape[0])
        if neck is not None:
            status = f"Parallax fix: {'ON' if self.parallax_on else 'OFF'} (pose {'AB'[self.pose_variant]})"
            angles = neck.camera_angles()
            if angles is not None:
                status += f" | camera angle {angles[0]:.0f} deg"
            put_text_bg(display, status, (20, 160), 0.5, WHITE, thickness=1)
        put_text_bg(display, "P parallax fix on/off, X swap pose", (20, 212), 0.5, GRAY, thickness=1)
        fps = 1.0 / (time.time() - now) if time.time() != now else 0.0
        put_text_bg(display, f"FPS: {fps:.1f}", (size[0] - 130, size[1] - 20), 0.7, GREEN)
        if self.audio is not None:
            bar = int(min(self.audio.level / 0.2, 1.0) * 120)  # full bar at an RMS of 0.2
            cv2.rectangle(display, (20, 175), (140, 185), (60, 60, 60), -1)
            cv2.rectangle(display, (20, 175), (20 + bar, 185), GREEN, -1)
            put_text_bg(display, "mic" if self.audio.active else "mic off", (150, 185), 0.45, GRAY, thickness=1, pad=3)
