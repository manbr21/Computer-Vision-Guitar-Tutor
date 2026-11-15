"""Settings that depend on the guitar, on how the markers are mounted and on the webcam.

Measure QUAD_* and NECK_LENGTH_MM on the real guitar (see the README): the camera-tilt estimate and the
parallax correction degrade when their proportions are off by more than ~10%.
"""

NUM_STRINGS = 6
NUM_FRETS = 12

# Marker ID -> (neck corner it defines, index of the marker corner that sits on that neck corner).
# The index depends on how each sticker is rotated when taped; change it if one is re-taped rotated.
NECK_CORNER = {
    0: ("TL", 1),  # nut, top edge
    1: ("TR", 0),  # nut, bottom edge
    2: ("BR", 3),  # fret 12, bottom edge
    3: ("BL", 2),  # fret 12, top edge
}

# Where the marker corners sit across the neck. Layout and dimensions must match the real mounting:
#   "outer_strings": each corner touches the outer string (low E / high e); the outline is drawn half a
#                    string gap outside. Corner-to-corner distance: ~44 mm at the nut, ~50 mm at fret 12.
#   "neck_edges":    each corner touches the edge of the fretboard; strings are inset. ~52 mm and ~60 mm.
MARKER_PLACEMENT = "outer_strings"
if MARKER_PLACEMENT not in ("outer_strings", "neck_edges"):
    raise ValueError(f"MARKER_PLACEMENT must be 'outer_strings' or 'neck_edges', got {MARKER_PLACEMENT!r}")
_OUTER = MARKER_PLACEMENT == "outer_strings"

# Position of the outer strings across the quad between the marker corners (0 = ID0 side, 1 = ID1 side).
STRING_U_RANGE = (0.0, 1.0) if _OUTER else (1 / 12, 11 / 12)

# Dimensions of the quad between the marker corners (mm).
QUAD_NUT_WIDTH_MM = 44.0 if _OUTER else 52.0     # corner to corner at the nut
QUAD_FRET12_WIDTH_MM = 50.0 if _OUTER else 60.0  # corner to corner at fret 12
NECK_LENGTH_MM = 325.0                           # nut line -> fret 12 line (half of a 650 mm scale)

# Where a finger presses, relative to what MediaPipe reports (measured on recordings of one user, see the README):
CONTACT_BACK = 0.4        # the touch point is this fraction of the way from the fingertip landmark toward its last joint
STRING_TOLERANCE = 0.5    # a finger covers its note if it is within this many string gaps of the string ...
FRET_BEHIND = 0.25        # ... and within the fret space or this fraction of it past its wire (players press behind the wire)
COMMON_SHIFT = 0.5        # the fingers are read with one shared shift of up to this many strings (and none along the neck),
COMMON_FRET_SHIFT = 0.0   # which absorbs what changes with the camera height and the angle of the hand (not along the neck:
                          # a shift there hid a finger one fret off in 40% of the frames)
NOTE_ON_FRAMES = 3        # a note turns green after being covered this many frames in a row ...
NOTE_OFF_FRAMES = 4       # ... and stays green until it has been missed this many frames in a row

MAX_HANDS = 1            # hands MediaPipe may track: only the fretting hand is looked at (see hands.fretting_hand_region)
FINGER_HEIGHT_MM = 10.0   # fingertip height above the fretboard surface when pressing
CAMERA_HFOV_DEG = 65.0    # horizontal field of view of the webcam (laptop webcams: ~60-78)
MAX_PARALLAX_MM = 25.0    # ignore corrections larger than this (degenerate pose)

# Strum detection from the microphone (chordar/audio.py)
AUDIO_ENABLED = True          # --no-audio on the command line turns it off for one run
STRUM_MIN_RMS = 0.02          # a strum must be at least this loud (full scale = 1.0)
STRUM_RATIO = 4.0             # ... and this many times louder than the running background level
STRUM_REFRACTORY_S = 0.7      # ignore new strums for this long after one (the chord is still ringing)
