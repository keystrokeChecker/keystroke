# Local Dataset

This directory contains recordings and synchronized key logs used to develop
and evaluate the keystroke-counting pipeline.

The recordings and logs are intentionally ignored by Git because typing audio
and key timing can be sensitive. Do not publish them without reviewing their
contents and obtaining the typist's explicit consent.

Expected files for a session named `example` are:

- `example.wav` — mono WAV recording.
- `example_log.csv` — synchronized keypress timestamps and word boundaries.
- `example_meta.json` — recording and calibration metadata.

Generated YAMNet arrays under `yamnet_dataset/` are also local artifacts and
can be regenerated from approved source sessions.

Dataset quality requirements and the train/validation/test split will be added
during Step 7 of the completion plan.
