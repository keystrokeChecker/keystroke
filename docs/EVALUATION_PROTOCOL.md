# Production Evaluation Protocol

## Status of earlier results

The JSON files currently under `keystroke_backend/results/` are legacy research
outputs and are not valid release evidence. They truncated true and predicted
sequences to the shorter prefix, counted separator keys as keystrokes in one
ground-truth parser, and evaluated the count model on recordings used to train
it. They must not be used to select or advertise a production method.

## Ground truth

- A non-boundary keylog row is one true keystroke event.
- A boundary row separates words and is never included in a word count.
- A final nonempty word is retained even when the log has no trailing boundary.
- Keylog time is converted to audio time as
  `timestamp_sec + sync_offset_ms / 1000`.
- Per-session offsets must come from capture metadata. Evaluation must not fit a
  separate offset against the session being scored.

## Production parity

Evaluation invokes the same rule, ML, and YAMNet predictor implementations used
by the API. Each predictor exposes a trace containing its final counts, the
onsets it actually counted, and its serving-time word groups. Ground-truth word
windows must never be used to pool production inference features.

Before the untouched test split is opened, validation must lock one candidate
method and its serving configuration. The evaluator emits a
`configuration_lock_sha256` over that method, its serving parameters, and the
fixed metric settings. A test run is not release-valid evidence unless that
hash is supplied and matches. Test evidence also rejects multiple methods,
injected predictors, custom artifact loaders, and noncanonical artifact paths.

## Count-sequence alignment

True and predicted integer count sequences are aligned with deterministic
Levenshtein/Needleman-Wunsch dynamic programming:

- exact pair: cost 0;
- wrong pair (substitution): cost 1;
- missing predicted word (deletion): cost 1;
- extra predicted word (insertion): cost 1.

Ties prefer fewer gap operations, then lower absolute count error, then a
diagonal, deletion, and insertion in that order. Missing and extra words are
compared with zero for count-error reporting.

The primary count metric is aligned exact-word accuracy:

`matches / (matches + substitutions + deletions + insertions)`

Missing and extra words therefore lower the score. Also report:

- gap-penalized MAE across every alignment operation;
- full-sequence exact match;
- match/substitution/deletion/insertion totals;
- predicted-minus-true word delta;
- absolute total-keystroke error.

Primary aggregate values are micro totals across recordings. Macro per-session
means are secondary diagnostics.

## Event and boundary metrics

Keystroke onsets use ordered one-to-one matching at one locked tolerance
(initially 80 ms). Matching maximizes the number of pairs and then minimizes
total absolute timing error. Report micro precision, recall, F1, timing MAE,
and timing p95. A value exactly on the inclusive tolerance is a match. Timing
MAE and p95 are `null`, not zero, when no events match.

A predicted word boundary is the interval between the previous predicted
group's last onset and the next group's first onset. A true separator is a
boundary true-positive only when it matches one unused predicted interval,
with the same documented fixed slack. Report boundary precision, recall, and
F1 separately from count accuracy.

Silence and approved non-keyboard recordings report false onsets per minute,
false words per minute, any-output rate, and empty-sequence exact rate.

## Leakage-safe partitions

The manifest unit is a whole recording. Each recording declares a `setup_id`
covering keyboard, microphone/device, placement, room, typist, and capture
batch. One setup may occur in only one of `train`, `validation`, or `test`.

- `train`: fit model parameters only;
- `validation`: tune detector, segmentation, classifier threshold, one global
  timing shift, and choose the candidate production method;
- `test`: untouched until the method and configuration are locked.

The test split must contain at least one unseen setup. If test results drive a
change, that test set is retired and a new untouched test set must be collected.
Model metadata must list training/tuning recording IDs and the manifest hash;
the evaluator rejects test contamination.

Before inference, the evaluator hashes every referenced WAV, keylog, and
metadata file. Duplicate resolved paths or byte-identical files under different
recording IDs are rejected across the whole manifest. Results contain the
selected recording hashes and a dataset-snapshot hash so later file changes
cannot be mistaken for the evaluated evidence.

The existing six count-model sessions are development-contaminated data. They
may be used to exercise the harness, but their output must be labeled
`valid_for_release: false`. Step 7 creates the quality-controlled manifest and
untouched test partition needed for a release-valid run.

`valid_for_release` means the result is eligible evidence under this protocol;
it does not mean the model passed the accuracy or false-positive targets.
Legacy scripts such as `tune_and_evaluate.py` remain diagnostic research tools
and must not select or score the untouched test split.
