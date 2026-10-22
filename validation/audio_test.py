"""
Check of the strum detector (chordar/audio.py) and its use by the app, with synthetic sound and no microphone.

A strum is simulated with plucked-string (Karplus-Strong) notes a few milliseconds apart; the room is white noise.
Run from the repository root:  python validation/audio_test.py
Exits with status 1 on any mismatch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_smoke_test as smoke
from chordar import config
from chordar.app import ChordARApp
from chordar.audio import BLOCK, SAMPLE_RATE, StrumDetector

RNG = np.random.default_rng(0)
OPEN_STRINGS_HZ = [82.4, 110.0, 146.8, 196.0, 246.9, 329.6]


def pluck(freq, seconds, amplitude):
    """Karplus-Strong plucked string."""
    n = int(SAMPLE_RATE * seconds)
    delay = int(SAMPLE_RATE / freq)
    buf = RNG.uniform(-1, 1, delay)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % delay]
        buf[i % delay] = 0.996 * 0.5 * (buf[i % delay] + buf[(i + 1) % delay])
    return amplitude * out


def strum(amplitude, seconds=1.0):
    """Six strings, 12 ms apart."""
    n = int(SAMPLE_RATE * seconds)
    sound = np.zeros(n)
    for k, freq in enumerate(OPEN_STRINGS_HZ):
        start = int(0.012 * k * SAMPLE_RATE)
        note = pluck(freq, seconds - 0.012 * k, amplitude)
        end = min(n, start + len(note))
        sound[start:end] += note[: end - start]
    return sound


def run(signal, detector):
    """Feeds the signal block by block (time = position in the signal); returns the strum times in seconds."""
    times = []
    for i in range(0, len(signal) - BLOCK + 1, BLOCK):
        now = i / SAMPLE_RATE
        if detector.feed(signal[i:i + BLOCK], now):
            times.append(round(now, 2))
    return times


def scene(events, noise, seconds):
    """Background noise with strums (start second, amplitude) mixed in."""
    signal = RNG.normal(0, noise, int(SAMPLE_RATE * seconds))
    for start, amplitude in events:
        s = strum(amplitude)
        a = int(start * SAMPLE_RATE)
        signal[a:a + len(s)] += s[: len(signal) - a]
    return signal


class FakeAudio:
    """Stands in for the StrumListener: strums are queued by the test."""
    active, level = True, 0.05

    def __init__(self):
        self.pending = 0

    def poll(self):
        n, self.pending = self.pending, 0
        return [0.0] * n

    def start(self):
        pass

    def stop(self):
        pass


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    quiet = run(scene([(1.0, 0.3), (3.0, 0.3), (5.0, 0.15)], 0.002, 7), StrumDetector())
    check(f"three strums in a quiet room are found, once each (at {quiet} s)",
          len(quiet) == 3 and all(abs(t - e) < 0.15 for t, e in zip(quiet, (1.0, 3.0, 5.0))))
    check("nothing is heard in a room with only noise", run(scene([], 0.01, 6), StrumDetector()) == [])
    noisy = run(scene([(2.0, 0.4), (4.5, 0.4)], 0.03, 7), StrumDetector())
    check(f"strums over a noisy room (noise RMS 0.03) are still found (at {noisy} s)", len(noisy) == 2)
    soft = run(scene([(1.0, 0.02)], 0.002, 3), StrumDetector())
    check("a very soft touch below the minimum level is ignored", soft == [])
    check("a strum that rings on does not count twice (refractory)",
          len(run(scene([(1.0, 0.3), (1.3, 0.3)], 0.002, 4), StrumDetector())) == 1)
    swell = np.concatenate([np.zeros(SAMPLE_RATE), np.sin(2 * np.pi * 220 * np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE)
                            * np.linspace(0.001, 0.3, 3 * SAMPLE_RATE)])
    check("a slow swell of volume is not a strum", run(swell, StrumDetector()) == [])

    # --- the app ---
    config.NECK_CORNER.clear()
    config.NECK_CORNER.update(smoke.MARKER_CORNERS)
    audio = FakeAudio()
    app = ChordARApp(smoke.FakeCapture(), audio)
    frame = smoke.synthetic_frame()
    app.process_frame(frame, now=10.0)
    check("no strum, no result", app.last_strum is None)
    audio.pending = 1
    app.process_frame(frame, now=11.0)
    check("a strum records the chord and the score at that moment (no hand: score 0 or none)",
          app.last_strum is not None and app.last_strum[1] == "C" and app.last_strum[3] in (0, None))
    start_chord = app.chord
    audio.pending = 1
    app.process_frame(frame, now=12.0)
    check("without auto-advance the chord stays", app.chord == start_chord)
    app.handle_key(ord("a"))
    check("A turns auto-advance on", app.auto_advance)
    app.last_strum = None
    audio.pending = 1
    app.process_frame(frame, now=13.0)
    check("auto-advance keeps the chord when the strum is not perfect", app.chord == start_chord)
    app.chord = "G"
    # no hand is in the synthetic frame, so give the perfect score directly
    audio.pending = 1
    app._handle_strums(100, 14.0)
    check("auto-advance moves to the next chord on a perfect strum (G wraps to A)", app.chord == "A")
    shown = app.process_frame(frame, now=14.5)
    check("the strum banner and mic bar are drawn", shown is not None)
    app.hands.close()

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: strums are detected in synthetic sound and drive the app")


if __name__ == "__main__":
    main()
