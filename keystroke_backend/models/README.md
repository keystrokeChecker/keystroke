# Model Artifacts

These files preserve the model state inherited at the start of the version 1
completion work. Joblib files must only be loaded from a trusted source because
the format can execute Python code during deserialization.

## Current artifacts

- `keystroke_classifier.joblib` — YAMNet-plus-local-feature candidate filter
  used by the `yamnet` prediction path.
- `count_predictor_new.joblib` — experimental per-word RandomForest count
  regressor used by the `ml` path. Its embedded metadata says it was pooled by
  ground-truth word boundaries, contains separator-inflated targets, and was
  scored on the same six sessions used for training. It is not approved as a
  production model.
- `candidate_classifier.joblib` — older hand-crafted candidate classifier.
- `yamnet_keystroke_classifier.joblib` — older/larger YAMNet classifier
  artifact.
- `exp_gt_1032.joblib` — experiment artifact matching the current count
  predictor payload.
- `yamnet_class_map.csv` — YAMNet class-name mapping.

No model is designated canonical for version 1 yet. Step 8 now provides
manifest-only trainers and deterministic validation selection. Candidate
artifacts use `count_predictor_candidate.joblib` and
`keystroke_classifier_candidate.joblib`; they must not replace runtime files
until `production_selection.json` locks one method and artifact hash. The
untouched test split verifies the already-selected pipeline and never chooses
it. See `../../docs/MODEL_RELEASE.md`.

`count_predictor_new.joblib` and `exp_gt_1032.joblib` are byte-identical legacy
artifacts. Keeping both temporarily documents the inherited state; neither is
release evidence.
