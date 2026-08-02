# Model Training and Selection

## Safety contract

Step 8 consumes a quality-passing private manifest from Step 7. Trainers open
only non-negative typing recordings assigned to `train`. They embed the exact
manifest SHA-256, training recording/setup IDs, validation IDs, fit settings,
and training timestamp in each candidate artifact. Test recordings are never
opened by a trainer or method selector.

Model selection uses the corrected production-parity evaluator on
`validation`. A candidate is ineligible if it produces a detection on a
validation silence fixture or, for a learned method, has incomplete artifact
provenance. Remaining candidates are ranked deterministically by aligned count
accuracy, full-sequence accuracy, boundary F1, event F1, non-keyboard false
onsets, count MAE, and method name.

## Train candidates

Run from `keystroke_backend/` after `validate_dataset.py` exits successfully:

```powershell
python train_model.py `
  --manifest path\to\private_manifest.json `
  --output models\count_predictor_candidate.joblib

python train_yamnet_classifier.py `
  --manifest path\to\private_manifest.json `
  --model-path models\keystroke_classifier_candidate.joblib
```

These commands create candidates and do not overwrite inherited runtime
artifacts.

## Evaluate validation and lock one method

Evaluate every serving path with the candidate artifacts:

```powershell
python evaluate_methods.py `
  --manifest path\to\private_manifest.json `
  --split validation `
  --methods rule ml yamnet `
  --ml-artifact models\count_predictor_candidate.joblib `
  --yamnet-artifact models\keystroke_classifier_candidate.joblib `
  --output-json results\validation_candidates.json

python select_production_method.py `
  results\validation_candidates.json `
  --output models\production_selection.json
```

The selection file contains one method, the frozen serving configuration, a
summary for every candidate, evidence hashes, and a single-method
`configuration_lock_sha256`. The lock includes the selected artifact SHA-256,
so different model bytes cannot be substituted for test evaluation.

## Promote and test once

Inspect `production_selection.json`. If `ml` was selected, promote the exact
selected candidate bytes to `models/count_predictor_new.joblib`. If `yamnet`
was selected, promote them to `models/keystroke_classifier.joblib`. The rule
method has no artifact. Preserve the inherited artifact separately before an
operator intentionally replaces it.

Run the untouched test split with exactly the selected method, frozen serving
arguments, and the emitted lock:

```powershell
python evaluate_methods.py `
  --manifest path\to\private_manifest.json `
  --split test `
  --methods SELECTED_METHOD `
  --locked-config-sha256 LOCK_FROM_SELECTION `
  --output-json results\untouched_test.json
```

Do not pass candidate artifact overrides during the test run: the promoted
runtime artifact must be at its canonical serving path. The evaluator rejects
changed metric settings, a changed artifact hash, multiple methods, custom
dependencies, incomplete provenance, duplicate data, or a mismatched lock.

The test result chooses nothing. Compare it once with the release targets in
`RELEASE_SPEC.md`. If it drives any model, threshold, segmentation, or method
change, retire the test set and collect a new untouched setup before testing
again.
