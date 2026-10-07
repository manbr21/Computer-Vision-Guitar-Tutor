# Web demo (legacy)

Flask demo that came with the original [Computer-Vision-Guitar-Tutor](https://github.com/nathanchiu05/Computer-Vision-Guitar-Tutor)
project. It streams the camera to a browser page and matches finger positions against `GuitarChords.csv`.

It is **not used by ChordAR** and is not maintained here: it has its own copy of the tracking logic and does not
share the calibration, pose correction or scoring of the `chordar` package. It is kept for reference and attribution.

```
pip install -r requirements.txt
python app_web.py        # run from this folder: match_chord.py reads ./GuitarChords.csv
```
