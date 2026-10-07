# ChordAR

Monitor-based augmented-reality guitar tutor. A webcam films you from the front and the screen works as a
mirror: the program draws, over the live image of your guitar, where each finger of a chord goes, tracks your
fretting hand and scores how many notes you got right.

It is a fork of [Chordially / Computer-Vision-Guitar-Tutor](https://github.com/nathanchiu05/Computer-Vision-Guitar-Tutor)
(ArUco markers + MediaPipe Hands) developed for a university project at CIn-UFPE.

## Features

- **Neck tracking with four ArUco markers** at the nut and at fret 12; frets follow the real fret spacing.
- **Fretting-hand tracking** with MediaPipe Hands; each fingertip is labeled with the string and fret it is on (e.g. `D2`).
- **Parallax correction:** a fingertip is ~1 cm above the board, so an oblique camera would read it on the
  neighbouring string. The camera pose is estimated from the markers and the fingertip is shifted to the board point under it.
- **Guided calibration:** a target neck is drawn on the video; distance, horizontal and vertical position,
  level and camera angle are checked with a hint for each. During play, a warning appears if the guitar drifts.
- **Scoring:** percentage of the chord's fretted notes covered, requiring the right string *and* fret.
- **Eight chords** (A, Am, C, D, Dm, E, Em, G) with a chord chart and finger instructions on screen.

## How it works

```
camera frame -> MarkerTracker (ArUco) -> Fretboard (pixels <-> string lane x position along the neck)
             -> FingertipTracker (MediaPipe) -> parallax shift across the neck -> (string, fret) per finger
             -> chord_accuracy -> overlay (neck grid, chord targets, chord chart, score)
```

## Install

Python 3.9+ and a webcam.

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

MediaPipe is pinned below 0.10.30: later releases removed the `mp.solutions` API used here.

## Prepare the guitar

1. Print `assets/arucos/print_sheet_standalone.html` at **100% scale** (each marker is 40 mm). Keep a white margin
   around every marker and stick it on something rigid (card) so it stays flat.
2. Place the markers (IDs as drawn on the sheet): **0** and **1** at the nut (top and bottom edge of the neck),
   **3** and **2** at fret 12 (top and bottom). Each marker's corner facing the neck is the reference.
3. Set `MARKER_PLACEMENT` in `chordar/config.py` to how you mounted them: `"outer_strings"` (corners touch the
   low E and high e strings) or `"neck_edges"` (corners touch the edges of the fretboard).
4. **Measure with a ruler** the distance between the corners at the nut, at fret 12, and from the nut line to the
   fret 12 line, and set `QUAD_NUT_WIDTH_MM`, `QUAD_FRET12_WIDTH_MM` and `NECK_LENGTH_MM`. The camera-angle estimate
   depends on these proportions: a ~18% error breaks the correction, up to ~9% is fine.
5. If you stick a marker on rotated, change its corner index in `NECK_CORNER` (a dot is drawn on each corner in use).

## Run

```
python -m chordar             # or: python main.py
python -m chordar --camera 1 --debug
```

First the calibration screen: move the guitar onto the drawn neck until every check is OK (ENTER starts anyway,
ESC quits). Then the lesson:

| Key | Action |
|-----|--------|
| `1`-`8` | choose the chord (A, Am, C, D, Dm, E, Em, G) |
| `P` | toggle the parallax correction (compare the score with and without it) |
| `X` | swap between the two camera-pose solutions |
| `C` | recalibrate (ESC there goes back to playing) |
| `ESC` | quit |

## Project layout

```
chordar/
  config.py       marker placement, neck dimensions, camera (the settings you edit)
  geometry.py     Fretboard: pixels <-> neck coordinates, fret positions, locating a fingertip
  pose.py         camera pose from the marker corners, parallax shift, camera angles
  markers.py      ArUco detection and smoothing
  hands.py        fingertip tracking (MediaPipe Hands)
  chords.py       chord library
  scoring.py      chord accuracy
  placement.py    neck-placement metrics and checks, drift monitor
  calibration.py  calibration screen with the target neck
  overlay.py      everything drawn over the video
  app.py          per-frame processing and keyboard control
assets/arucos/    marker images and a printable sheet
validation/       simulation-based checks (no camera needed)
legacy/web_demo/  Flask demo from the original project (not used by ChordAR)
```

## Validation

Run from the repository root; each one exits with status 1 if a check fails.

```
python validation/synthetic_parallax_test.py     # virtual camera: tilt, off-center neck, wrong FOV/dimensions
python validation/calibration_alignment_test.py  # placement metrics, hints and the drift monitor
python validation/app_smoke_test.py              # ArUco markers drawn on a synthetic frame, end to end
```

In the simulation, without the parallax correction the right string drops to ~70% at 20 deg of camera tilt and ~5%
at 30 deg; with it, it stays at ~100%.

## Known limitations

- Markers must be rigid and flat on the instrument; a loosely taped marker shakes the whole grid.
- A dark patch is occasionally read as a marker ID 0-3, briefly flickering a phantom neck (a filter is planned).
- The fret coordinate follows the drawn grid, which does not model perspective: with strong tilt along the neck
  or a neck far from the image center the frets drift (the calibration keeps play within the safe range).
- When the pinky's phalanx shows more than its tip, MediaPipe may take the phalanx for the fingertip.
- The target neck assumes the nut on the left of the (mirrored) screen; left-handed layouts are not supported.

## Credits and licenses

- Original project: [nathanchiu05/Computer-Vision-Guitar-Tutor](https://github.com/nathanchiu05/Computer-Vision-Guitar-Tutor).
  Its README states the MIT license but the repository has no `LICENSE` file.
- The U-Net baseline the project was compared with is [abhishekrana/guitar-augmented-reality](https://github.com/abhishekrana/guitar-augmented-reality) (Apache-2.0); it is not included here.
