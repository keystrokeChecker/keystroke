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

The current sessions are contaminated development data: some were used for
training, parameter tuning, and reported evaluation. They do not form an
untouched test set and cannot support release claims.

The metric and split contract is documented in
`../../docs/EVALUATION_PROTOCOL.md`; the consent, capture, privacy, fixture,
and split procedure is in `../../docs/DATASET_COLLECTION.md`.

New captures must include opaque keyboard, microphone, placement, room,
typist, capture-batch, intended-split, fixture-type, and derived setup IDs in
their metadata. Start a private manifest from
`../../docs/evaluation_manifest.example.json`, then run from the backend:

```powershell
python validate_dataset.py path\to\private_manifest.json
```

Do not add the populated private manifest or its quality report to Git when it
contains sensitive paths or capture information.
