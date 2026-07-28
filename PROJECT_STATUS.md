# Project Status

## Current state

The repository contains a functional research prototype with:

- A Flutter recording, upload, history, and playback interface.
- A FastAPI `/health` and `/analyze` service.
- Rule-based onset detection and word segmentation.
- YAMNet candidate filtering and an experimental count regressor.
- Recorded development sessions, diagnostic scripts, and model artifacts.

It is not yet a version 1 release. The remaining blockers are model-side:
the existing evaluation is not release-valid, the dataset has no untouched
test partition, one production pipeline has not been selected, and physical
Android acceptance testing and final packaging are still outstanding.

## Completion sequence

1. Establish the repository and release baseline.
2. Restore a clean Flutter/Android build.
3. Add backend automated tests.
4. Harden and simplify the backend.
5. Fix Flutter reliability and settings.
6. Replace the evaluation methodology.
7. Add dataset quality controls and collect more valid sessions.
8. Retrain and select one production pipeline.
9. Complete physical-device end-to-end acceptance testing.
10. Finish packaging, CI, documentation, and release artifacts.

## Step 1 decisions

- Version 1 targets Android only.
- The backend runs on the same trusted local network as the client.
- Raw WAV recordings, key logs, metadata, and derived training arrays are
  private local artifacts and are excluded from Git.
- Existing runtime model artifacts remain versioned for reproducibility until a
  model manifest and artifact-release process replace them.
- The canonical prediction method will be chosen only after the corrected
  held-out evaluation in Steps 6–8.

See `docs/RELEASE_SPEC.md` for the release contract and acceptance criteria.

## Progress

- Step 1 complete: repository baseline and release contract committed.
- Step 2 complete: Flutter analysis and tests pass; Android debug APK builds;
  release-mode signing configuration was verified with a temporary test key.
  A permanent private release key will be supplied during final packaging.
- Step 3 complete: deterministic backend tests cover API routing and cleanup,
  segmentation, formatting, and all three predictor paths.
- Step 4 complete: uploads and request bodies are bounded and validated,
  cancellation-safe cleanup is in place, inference is moved off the event loop
  with bounded concurrency, readiness is explicit, model loads are cached and
  validated, silent prediction fallback is removed, and the backend suite has
  70 passing tests.
- Step 5 complete: the Android client persists validated settings, supports
  rule/ML/YAMNet selection and connection testing, bounds and cancels uploads,
  preserves failed/cancelled/interrupted recordings for retry, filters and
  orders history safely, tracks playback completion, serializes recorder and
  persistence operations, disables conflicting actions, and has 32 passing
  Flutter tests. Flutter analysis and the Android debug APK build pass.
