# Model Artifacts

These files preserve the model state inherited at the start of the version 1
completion work. Joblib files must only be loaded from a trusted source because
the format can execute Python code during deserialization.

## Current artifacts

- `keystroke_classifier.joblib` — YAMNet-plus-local-feature candidate filter
  used by the `yamnet` prediction path.
- `count_predictor_new.joblib` — experimental per-word RandomForest count
  regressor used by the `ml` path. Its embedded metadata says it was pooled by
  ground-truth word boundaries, so it is not yet approved as a production
  model.
- `candidate_classifier.joblib` — older hand-crafted candidate classifier.
- `yamnet_keystroke_classifier.joblib` — older/larger YAMNet classifier
  artifact.
- `exp_gt_1032.joblib` — experiment artifact matching the current count
  predictor payload.
- `yamnet_class_map.csv` — YAMNet class-name mapping.

No model is designated canonical for version 1 yet. Step 8 will produce a
versioned model manifest, archive obsolete experiments, and select one serving
pipeline using untouched test data.
