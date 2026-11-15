# ChordAR

Monitor-based augmented-reality guitar tutor. A webcam films you from the front and the screen works as a
mirror: the program draws, over the live image of your guitar, where each finger of a chord goes, tracks your
fretting hand and scores how many notes you got right. No markers and no special hardware: a laptop is enough.

It started as a fork of [Chordially / Computer-Vision-Guitar-Tutor](https://github.com/nathanchiu05/Computer-Vision-Guitar-Tutor)
(ArUco markers + MediaPipe Hands) and was developed for a university project at CIn-UFPE. The version that tracks the
neck with four ArUco markers is kept in the `aruco-markers` branch.

## Quick start

Python 3.9+ and a webcam.

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m chordar
```

1. **Photo.** Hold the guitar as you will play it, inside the outline drawn on the screen, with your hand off the neck,
   and press SPACE. The screen warns if the light is too dark, too bright or from behind.
2. **Four clicks** on the neck, in the order the screen asks (the nut at the low E and the high e string, then the
   12th fret wire at the high e and the low E string). Click on the nut edge on the fretboard side and on the 12th fret
   wire; the program then finds the real fret wires in the photo and snaps your clicks onto them, so a few pixels of
   error do not matter (U switches back to your raw clicks). ENTER saves; T takes a new photo.
3. **Play.** Choose a chord with the keys 1 to 8. Notes turn green when a finger covers them, one message says what to
   fix first, and a ring shows the accuracy. A strum on the microphone judges the chord at that moment.

The photo is saved to `references/last.npz`; `python -m chordar --reference references/last.npz` starts without
taking a new one (it only works with similar light and a guitar held in a similar way).

| Key | Action |
|-----|--------|
| `1`-`8` | choose the chord (A, Am, C, D, Dm, E, Em, G) |
| `]` / `[` | move the virtual capo up / down a fret (up to fret 7); `0` removes it |
| `A` | auto-advance: a perfect strum moves to the next chord |
| `C` | take a new photo (ESC there keeps the current one) |
| `O` | hide / show the neck markings (outline, frets, strings); the fingers and the chord targets stay |
| `H` | list of the keys |
| `ESC` | quit |

Keys for tuning, not shown in the app's help: `D` technical panel (pose, parallax, fingertip labels, FPS, microphone
level), and while it is on, `P` turns the parallax correction on and off and `X` swaps between the two camera-pose solutions.

Options: `--camera N`, `--no-audio` (without the microphone), `--reference FILE`, `--record FOLDER`,
`--markers` (ArUco markers instead of the photo, see below), `--debug`.

## Where to put the notebook

Put the notebook in front of you on something stable (a table, a stool, a chair) so that **the webcam is at the height of the
neck or a little below it**, about 50-60 cm away, with the screen where you can see it. The finger positions are read from
the hand as the camera sees it, and what the app corrects (the shift that all the fingers share) has a limit: with the camera
10 to 20 degrees below the neck the eight chords were read 100% of the time once the fingers were in place, with the camera
about 20 degrees *above* it only about 60% (the fingers read a whole string off). The photo screen says
"Camera above the neck: lower the notebook" when the neck's edges show it clearly (it cannot always tell, so this is a guide: keep the
notebook at the height of the neck).

## Lighting

The frets must be visible: the program finds the fret wires in the photo and in the live image to place the grid, so the
light has to show them (an even light on the neck, not behind the guitar, not a dark room). The photo screen warns if the light
is too dark, too bright or from behind, and says how many of the 12 frets it can see ("Only 7 of 12 frets are visible: more
light on the neck"); during play a banner says "Frets hard to see: put more light on the neck" if the wires are lost for
about a second.

## Features

- **Marker-free neck tracking from one photo.** Features of the photo's neck region are matched to each frame
  (ORB + RANSAC homography) and the corners are followed between frames by optical flow; the two are blended
  (the flow is steady but drifts, the match does not drift but jumps), and implausible matches (a stretched or
  sheared neck, a jump the neck cannot make) are refused.
- **Frets drawn where the wires really are.** The photo's fret wires are found and the nut, the 12th fret and the
  spacing (with the foreshortening along the neck) are fitted to the fret formula; during play the same fit is repeated
  on the live frames, so the grid stays on the wires when you move closer or tilt the guitar.
- **Fretting-hand tracking** with MediaPipe Hands, only on the part of the frame around the first frets and the neck, so the
  strumming hand is never looked at (and the hand tracker costs ~17 ms instead of ~33 ms per frame).
- **Parallax correction (off by default):** a fingertip is ~1 cm above the board, so an oblique camera reads it shifted
  across the neck. The camera pose is estimated from the neck corners and the touch point is shifted back; `P` turns it
  on (with `D`). It is off because the two possible pose solutions can be mixed up at small tilts and then it makes things worse
  (on a recording with the camera at another height it lowered the notes covered from 88% to 82%); the shared shift of
  the scoring (below) absorbs most of what it was for.
- **Scoring and coaching:** every note of a chord belongs to one finger, as the chord chart shows (the number in each dot),
  and is covered when that finger is on it. A finger is read relative to its note; the shift that all the fingers share
  (it changes with the height of the camera against the guitar and the angle of the hand) is taken out, within 0.5
  strings and none along the neck, and then each finger must be within 0.5 string gaps of its string and in its fret
  space or just behind the wire. The touch point is the fingertip landmark moved 40% toward the last joint, because the
  pad that presses is behind the tip. Notes turn green when covered (steadily: 3 frames in a row to turn on, 4 missed to
  turn off), a ring shows the accuracy, and one message says what to fix first ("Finger 2: 1 string up, 1 fret left"),
  with an arrow from that finger to its note. The constants are in `config.py` and were fitted to recordings of one user
  (see Known limitations).
- **Eight chords** (A, Am, C, D, Dm, E, Em, G) with a chord chart on screen.
- **Virtual capo:** tell the program the fret of a physical capo and the chord shape moves up that many frets.
- **Strum detection:** the microphone listens for the sharp rise in loudness of a strum. Each strum judges the chord at
  that moment ("Strum! C: 100%"), and with auto-advance (`A`) a perfect strum moves on to the next chord. This detects
  *that* you strummed, not *which* chord sounded.
- **Drift warning:** if the guitar moves away from where the photo was taken, a hint says how (C takes a new photo).

## How it works

```
photo (once) -> NeckReference (4 corners snapped to the fret wires + measured fret spacing)
camera frame -> MarklessTracker (ORB match + optical flow + live fret fit) -> Fretboard (pixels <-> string x fret)
             -> FingertipTracker (MediaPipe) -> touch point -> parallax shift -> covered notes per chord
             -> overlay (neck grid, targets, coaching message, chord chart, accuracy ring)
```

## Markers instead of the photo

The original method, kept for comparison and as ground truth for the evaluation tools: `python -m chordar --markers`
(or check out the `aruco-markers` branch).

1. Print `assets/arucos/print_sheet_standalone.html` at **100% scale** (each marker is 40 mm), keep a white margin
   around every marker and stick it on something rigid so it stays flat.
2. Place the markers: **0** and **1** at the nut (top and bottom edge of the neck), **3** and **2** at fret 12.
3. Set `MARKER_PLACEMENT` in `chordar/config.py` (`"outer_strings"` or `"neck_edges"`) and measure with a ruler
   `QUAD_NUT_WIDTH_MM`, `QUAD_FRET12_WIDTH_MM` and `NECK_LENGTH_MM` (a ~18% error breaks the camera-angle estimate).
4. With markers the calibration screen draws a target neck and checks distance, position, level, camera angle and the
   light; ENTER takes your current position and distance as the reference.

Measuring the photo-based tracker against the markers: record sessions with the markers glued on, then evaluate.

```
python -m chordar --markers --record sessions/            # raw video + the corners the markers found (ground truth)
python validation/evaluate_recording.py sessions/<date_time>
python validation/markerless_test.py --photo photo_with_markers.jpg    # random changes to one photo
```

The markers are painted out of the reference and masked in every frame, so the marker-free tracker cannot use them.

## Project layout

```
chordar/
  app.py          per-frame processing, keyboard control and the set-up (photo or marker calibration)
  reference.py    the photo set-up: photo, four clicks, advice, snapping (python -m chordar.reference does it alone)
  markerless.py   marker-free neck tracking from the reference photo
  frets.py        finds the fret wires and fits the nut, the 12th fret and the spacing to them
  geometry.py     Fretboard: pixels <-> neck coordinates, fret positions, locating a fingertip
  pose.py         camera pose from the neck corners, parallax shift, camera angles
  hands.py        fretting-hand tracking (MediaPipe Hands) and the touch point
  scoring.py      which notes a chord has covered (tolerances, one finger per note)
  coaching.py     the one correction to say
  chords.py       chord library
  audio.py        strum detection from the microphone
  placement.py    neck-placement metrics and checks, drift monitor, lighting advice
  overlay.py      everything drawn over the video
  window.py       resizable display window, image scaled without stretching
  recorder.py     records a session (raw video + ground-truth corners)
  markers.py, calibration.py   the ArUco markers and their calibration screen (--markers)
  config.py       the settings you edit
assets/arucos/    marker images and a printable sheet
validation/       checks that need no camera
legacy/web_demo/  Flask demo from the original project (not used by ChordAR)
```

## Validation

Run from the repository root; each one exits with status 1 if a check fails.

```
python validation/setup_flow_test.py             # the photo set-up and the app that uses it
python validation/frets_test.py                  # fret-wire snapping on a synthetic tapered neck
python validation/reference_test.py              # marker-free tracker on a synthetic textured scene
python validation/scoring_test.py                # each finger on its own note, the shared shift, the steadying
python validation/hands_region_test.py           # the region where the fretting hand is looked for
python validation/coaching_test.py               # finger coaching messages
python validation/capo_test.py                   # virtual capo: shifted notes, score, keys and drawing
python validation/audio_test.py                  # strum detector on synthetic sound, and its use by the app
python validation/lighting_test.py               # lighting advice
python validation/recorder_test.py               # the session recorder
python validation/synthetic_parallax_test.py     # virtual camera: tilt, off-center neck, wrong FOV/dimensions
python validation/calibration_alignment_test.py  # marker calibration: placement metrics, hints, drift monitor
python validation/app_smoke_test.py              # ArUco markers drawn on a synthetic frame, end to end
```

In the simulation, without the parallax correction the right string drops to ~70% at 20 deg of camera tilt and ~5%
at 30 deg; with it, it stays at ~100%.

## Known limitations

- The finger constants (`CONTACT_BACK`, `STRING_TOLERANCE`, `FRET_BEHIND`, `COMMON_SHIFT`) were fitted to recordings of one
  user. The chord library follows that user's fingering (Dm with the ring finger on B3, Em with index and middle, G with
  middle on the low E, index on A and pinky on the high e); the app judges each note by its own finger, so a different
  fingering of the same chord is counted as wrong.
- The reading of the fingers is accurate to about half a string (the strings are only ~14-16 px apart at the comfortable
  playing distance), which is the limit of what the app can tell apart. On a recording of the eight chords held still, with
  the camera ~20 degrees from the neck, 88% of the notes were covered by their own finger (74% of the frames had the whole
  chord); on another recording with the camera ~12 degrees from the neck, the chords not used to fit the constants scored
  98.7% with the earlier any-finger rule. In simulation a finger moved a whole string off is still accepted in about a
  quarter of the frames (tolerance 0.5), and one moved a whole fret in 5%. The index finger on frets 1 and 2 is the least
  reliable. A camera at the height of the neck or a little above works best.
- The photo only works with light and distance similar to when it was taken; with much dimmer or backlit light the
  neck may not be found (take a new photo with C).
- Tracking is validated against ArUco markers on six recorded sessions of one user, one guitar and one laptop.
- With strong tilt along the neck or a neck far from the image center the frets can drift.
- Only the hand in the region around the first frets is tracked (`MAX_HANDS = 1`): a left-handed player, or a fretting hand
  that leaves that region (past fret 9 or so), is not seen.
- The target outline assumes the nut on the left of the (mirrored) screen; left-handed layouts are not supported.

## Credits and licenses

- Original project: [nathanchiu05/Computer-Vision-Guitar-Tutor](https://github.com/nathanchiu05/Computer-Vision-Guitar-Tutor).
  Its README states the MIT license but the repository has no `LICENSE` file.
- The U-Net baseline the project was compared with is [abhishekrana/guitar-augmented-reality](https://github.com/abhishekrana/guitar-augmented-reality) (Apache-2.0); it is not included here.
