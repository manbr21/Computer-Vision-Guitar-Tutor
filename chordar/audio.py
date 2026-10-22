"""Strum detection from the microphone: a sharp rise in loudness is read as the strings being strummed."""
import logging
import threading
import time
from collections import deque

import numpy as np

from chordar import config

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
BLOCK = 1024          # samples per analysis block (~23 ms)
RISE = 1.5            # a block must be this many times louder than the previous one: a sharp attack, not a swell
FLOOR_ALPHA = 0.02    # how fast the background level follows the sound between strums


class StrumDetector:
    """
    Loudness-onset detector. Feed it consecutive blocks of samples; feed() returns True on the block where a
    strum starts. The background level is a slow average that does not learn the strums themselves, so a
    quiet room and a noisy one both work.
    """

    def __init__(self, min_rms=None, ratio=None, refractory_s=None):
        self.min_rms = config.STRUM_MIN_RMS if min_rms is None else min_rms
        self.ratio = config.STRUM_RATIO if ratio is None else ratio
        self.refractory_s = config.STRUM_REFRACTORY_S if refractory_s is None else refractory_s
        self.level = 0.0       # RMS of the last block, for the on-screen meter
        self._floor = None
        self._previous = 0.0
        self._last_strum = -1e9

    def feed(self, block, now):
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))) if len(block) else 0.0
        self.level = rms
        if self._floor is None:
            self._floor = rms
        louder = rms > max(self.min_rms, self.ratio * self._floor)
        strum = louder and rms > RISE * self._previous and now - self._last_strum >= self.refractory_s
        if strum:
            self._last_strum = now
        elif rms < 2 * max(self._floor, 1e-6):
            self._floor += FLOOR_ALPHA * (rms - self._floor)
        self._previous = rms
        return strum


class StrumListener:
    """Opens the default microphone and collects the strum times. Does nothing (and says so) if it cannot."""

    def __init__(self, detector=None):
        self.detector = detector or StrumDetector()
        self.active = False
        self._strums = deque()
        self._lock = threading.Lock()
        self._stream = None

    def start(self):
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK, callback=self._on_audio)
            self._stream.start()
            self.active = True
            log.info("Listening for strums on the default microphone")
        except Exception as error:  # no package, no device, or the microphone permission was refused
            log.warning("Strum detection disabled: %s", error)
            self._stream = None
        return self.active

    def _on_audio(self, indata, frames, time_info, status):
        if self.detector.feed(indata[:, 0], time.time()):
            with self._lock:
                self._strums.append(time.time())

    def poll(self):
        """The strums heard since the last call, as a list of times."""
        with self._lock:
            strums = list(self._strums)
            self._strums.clear()
        return strums

    @property
    def level(self):
        return self.detector.level

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.active = False
