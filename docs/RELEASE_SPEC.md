# Version 1 Release Specification

## Product target

Version 1 is an Android-first, local-network application that records the
sound of a consenting user's physical keyboard, sends the WAV recording to a
FastAPI backend on the same trusted network, and displays the predicted number
of keystrokes in each typed word.

Other generated Flutter targets are not part of version 1 until they receive
their own platform configuration and physical-device verification.

## Supported environment

- Client: Android phone or tablet.
- Backend: Python service running on a trusted computer reachable over local
  Wi-Fi.
- Audio input: mono WAV produced by the application.
- Connectivity: a user-configured HTTP or HTTPS backend URL.
- Processing: recordings are analyzed transiently by the backend and deleted
  from its temporary storage after the request.

## Required user workflow

1. Launch the application and grant microphone permission.
2. Configure and test the backend URL.
3. Start and stop a recording.
4. Receive one formatted count sequence, such as `3|7`.
5. Replay, retry, or delete the local recording.
6. Retain settings and history across application restarts.

## Functional release criteria

- Flutter analysis and automated tests pass.
- Android debug and signed release APKs build successfully.
- Backend automated tests pass without downloading a model during the test
  run.
- Invalid, empty, oversized, corrupt, and unsupported uploads return controlled
  client errors and do not leak temporary files.
- Requests have bounded upload size, recording duration, and processing time.
- The selected production method and model version are explicit; there is no
  silent fallback to another prediction method.
- A clean installation completes the end-to-end workflow on a physical Android
  device.

## Model release criteria

The final thresholds will be confirmed after the corrected evaluation harness
is implemented. The initial target is:

- At least 85% exact per-word count accuracy on untouched test sessions.
- Word-boundary precision, recall, and F1 reported separately.
- Full-sequence exact-match accuracy reported.
- Missing and extra predicted words penalized by the primary metric.
- No detections on approved silence fixtures and a documented false-positive
  rate on non-keyboard audio.
- Test recordings include at least one keyboard or microphone setup absent from
  training.

If the data cannot support these targets, the release must be labeled as an
experimental prototype rather than a completed analyzer.

## Privacy and security

- Only record the device owner's typing with explicit knowledge and consent.
- Raw datasets and key logs are private by default and are excluded from Git.
- The backend is intended for a trusted local network; exposure to the public
  internet is outside the version 1 scope.
- Server logs must not include uploaded audio bytes or inferred typed content.

## Artifact policy

- Source code, documentation, tests, model metadata, and the model artifacts
  needed for the local runtime are versioned.
- Raw recordings, key logs, derived training arrays, caches, local credentials,
  signing keys, and build outputs are not versioned.
- If model artifacts grow substantially, they should move to Git LFS or a
  versioned release-artifact store without changing their model manifest.
