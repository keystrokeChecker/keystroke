# Project Status

## Current state

The repository contains a functional research prototype with:

- A Flutter recording, upload, history, and playback interface.
- A FastAPI `/health` and `/analyze` service.
- Rule-based onset detection and word segmentation.
- YAMNet candidate filtering and an experimental count regressor.
- Recorded development sessions, diagnostic scripts, and model artifacts.

It is not yet a version 1 release. Confirmed blockers include an invalid
Android Kotlin entry point, a failing Flutter test, missing backend tests,
unhardened upload handling, unresolved production-model selection, and an
insufficient dataset.

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
