# Dataset Collection Protocol

## Purpose and privacy

Collect only consented recordings made by the typist operating the capture
computer. Audio and key timing are sensitive private data and remain under
`keystroke_backend/data/`, which is ignored by Git. The recorder stores key
identifiers and timing for evaluation, but metadata and manifests must never
store the phrase or reconstructed typed text. Use opaque aliases for people,
rooms, and devices rather than names or serial numbers.

The recordings already present in `keystroke_backend/data/` were used during
development. They are diagnostic-only and must never be reassigned to a
train, validation, or test split.

## Assign the split before recording

Before capture, assign aliases for keyboard, microphone, microphone placement,
room, typist, and capture batch. Together these fields produce an opaque
`setup_id`. Every recording with the same setup must remain in exactly one of
`train`, `validation`, or `test`.

- `train` is available for fitting model parameters.
- `validation` is available for thresholds, configuration, and method choice.
- `test` remains sealed until one method and configuration have been locked.
- `diagnostic` is for contaminated or exploratory recordings only.

Changing the keyboard, microphone, placement, room, typist, or capture batch
creates a new setup. Do not move recordings between splits after observing
model output. If test results cause a model or configuration change, retire
that test set and capture a new untouched setup.

## Required fixtures

Capture typing fixtures with varied word lengths, typing speeds, and pauses.
Also capture silence (room ambience with no deliberate foreground sound) and
non-keyboard negatives such as chair movement, desk contact, mouse clicks, and
approved speech/background activity. Do not manufacture negative fixtures by
cutting them from a typing recording; the whole recording is the manifest
unit.

At minimum, each controlled split needs a typing recording, and both the
validation and untouched test splits need silence and non-keyboard fixtures.
Validation negatives are required to select a safe method without consulting
test results. For meaningful
evidence, use several independent setups and recordings per fixture type; the
validator's minimum coverage checks are a guardrail, not a statistical sample
size claim.

## Capture command

Run from `keystroke_backend/`. The aliases below are examples:

```powershell
python record_dataset.py `
  --name train_typing_001 `
  --duration 20 `
  --keyboard-id keyboard-a `
  --microphone-id microphone-a `
  --placement-id desk-left-20cm `
  --room-id room-a `
  --typist-id typist-a `
  --capture-batch-id batch-2026-08-a `
  --intended-split train `
  --fixture-type typing
```

Use `--fixture-type silence` or `--fixture-type non_keyboard` for negative
fixtures. Do not press keys during negative capture. `--repeat` appends a
stable numbered suffix and keeps the same setup metadata.

The recorder emits a WAV, keylog CSV, and metadata JSON. Immediately review
the recording summary; reject and recapture obvious interruptions, wrong
fixtures, accidental keypresses in a negative recording, or lack of consent.
Never edit timing rows to make predictions look better.

## Manifest and quality gate

Copy `docs/evaluation_manifest.example.json` to a private manifest location,
replace the placeholder paths and IDs, and set `silence` or `non_keyboard` for
negative fixtures. The manifest setup ID must match the generated metadata
setup ID.

Validate before training or evaluation:

```powershell
python validate_dataset.py path\to\private_manifest.json --output dataset_quality.json
```

Exit code 0 means every recording passed the WAV, metadata, keylog, clipping,
timing, and minimum split/fixture coverage checks. Exit code 1 means the JSON
report contains errors. A diagnostic-only manifest is expected to exit 1 on
release coverage even when its individual files are readable.

Keep the quality report private when it contains local paths. After validation,
freeze the manifest and recording bytes; the Step 6 evaluator hashes all of
them and rejects duplicates or later mutation.
