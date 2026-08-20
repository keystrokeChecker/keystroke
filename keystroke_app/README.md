# Keystroke Analyzer Android Client

This Flutter application records a local WAV file, sends it to the Keystroke
Analyzer backend on a trusted local network, and displays the predicted count
sequence.

## Run locally

```powershell
flutter pub get
flutter run
```

On first launch, open **Settings** and enter the backend address, for example
`http://192.168.1.20:8000`. Use **Test Connection**, choose `Rule`, `ML`, or
`YAMNet`, then save.

## Reliability behavior

- Settings and newest-first recording history persist across restarts.
- Uploads time out and can be cancelled explicitly.
- Failed, cancelled, and lifecycle-interrupted recordings stay available for
  Retry instead of being deleted.
- Saved recordings can be played, paused, replayed, or swiped away.
- Missing files are removed from restored history.
- Android backups are disabled because recordings may contain sensitive typing
  audio. Cleartext HTTP is enabled only for the documented trusted-LAN use case.

## Verification

```powershell
flutter analyze --no-pub
flutter test --no-pub
flutter build apk --debug --no-pub
```

Release builds require the signing values documented in
`android/key.properties.example`.
