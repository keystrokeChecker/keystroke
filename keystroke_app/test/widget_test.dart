import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:keystroke_app/audio_services.dart';
import 'package:keystroke_app/backend_client.dart';
import 'package:keystroke_app/keystroke_home_page.dart';
import 'package:keystroke_app/local_store.dart';
import 'package:keystroke_app/main.dart';
import 'package:keystroke_app/recording_result.dart';
import 'package:keystroke_app/recording_storage.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('KeystrokeHomePage', () {
    testWidgets('shows setup guidance when the backend is not configured', (
      tester,
    ) async {
      final harness = await _pumpHome(tester);

      expect(find.text('Backend setup required'), findsOneWidget);
      expect(
        find.text('Configure the backend URL in Settings to begin.'),
        findsOneWidget,
      );
      expect(find.text('No saved recordings yet.'), findsOneWidget);

      await tester.tap(find.byKey(const ValueKey<String>('record-button')));
      await tester.pumpAndSettle();

      expect(
        find.text('Configure the backend URL in Settings before recording.'),
        findsOneWidget,
      );
      expect(harness.recorder.permissionChecks, 0);
      expect(harness.recorder.startedPaths, isEmpty);

      await tester.tap(find.text('SET UP'));
      await tester.pumpAndSettle();

      expect(find.text('Settings'), findsOneWidget);
      expect(
        find.byKey(const ValueKey<String>('backend-url-field')),
        findsOneWidget,
      );
      expect(find.byKey(const ValueKey<String>('test-connection-button')), findsOneWidget);
    });

    testWidgets('tests and persists a normalized backend URL', (
      tester,
    ) async {
      late _FakeBackendClient connectionClient;
      final builtUrls = <String>[];
      final harness = await _pumpHome(
        tester,
        backendClientBuilder: (baseUrl) {
          builtUrls.add(baseUrl);
          connectionClient = _FakeBackendClient(baseUrl: baseUrl);
          return connectionClient;
        },
      );

      await tester.tap(find.byTooltip('Settings'));
      await tester.pumpAndSettle();
      await tester.enterText(
        find.byKey(const ValueKey<String>('backend-url-field')),
        ' HTTP://192.168.1.20:8000/ ',
      );

      await tester.tap(
        find.byKey(const ValueKey<String>('test-connection-button')),
      );
      await tester.pumpAndSettle();

      expect(find.text('Connection successful.'), findsOneWidget);
      expect(builtUrls, <String>['http://192.168.1.20:8000']);
      expect(connectionClient.connectionTests, 1);

      await tester.tap(
        find.byKey(const ValueKey<String>('save-settings-button')),
      );
      await tester.pumpAndSettle();

      final settings = await harness.store.loadSettings();
      expect(settings.backendUrl, 'http://192.168.1.20:8000');
      expect(find.text('Backend setup required'), findsNothing);
      expect(find.text('Settings saved. Ready to record.'), findsOneWidget);

      await tester.tap(find.byTooltip('Settings'));
      await tester.pumpAndSettle();
      final field = tester.widget<TextField>(
        find.byKey(const ValueKey<String>('backend-url-field')),
      );
      expect(field.controller!.text, 'http://192.168.1.20:8000');
    });

    testWidgets('reports microphone permission denial without starting', (
      tester,
    ) async {
      final recorder = _FakeRecorderService(permissionGranted: false);
      final harness = await _pumpHome(
        tester,
        settings: _configuredSettings,
        recorder: recorder,
      );

      await tester.tap(find.byKey(const ValueKey<String>('record-button')));
      await tester.pumpAndSettle();

      expect(
        find.text('Microphone permission is required to record.'),
        findsOneWidget,
      );
      expect(recorder.permissionChecks, 1);
      expect(recorder.startedPaths, isEmpty);
      expect(await harness.store.loadHistory(), isEmpty);
    });

    testWidgets(
      'records, analyzes, displays, and persists a successful result',
      (tester) async {
        final backend = _FakeBackendClient(
          baseUrl: _configuredSettings.backendUrl,
          onAnalyze: (file, method) async => AnalyzeResponse(
            counts: <int>[2, 1, 3],
            formatted: '2 left, 1 space, 3 right',
          ),
        );
        final harness = await _pumpHome(
          tester,
          settings: const AppSettings(
            backendUrl: 'http://192.168.1.20:8000',
            method: 'yamnet',
          ),
          backendClientBuilder: (_) => backend,
          clock: () => _fixedTime,
        );

        await _recordAndStop(tester);

        expect(find.text('Analysis complete.'), findsOneWidget);
        expect(find.text('2 left, 1 space, 3 right'), findsNWidgets(2));
        expect(backend.analyzedMethods, <String>['default']);
        expect(backend.analyzedPaths, harness.recorder.startedPaths);
        expect(
          harness.recorder.startedPaths.single,
          contains('1720442096789123'),
        );
        expect(harness.recorder.startedPaths.single, endsWith('_001.wav'));

        final history = await harness.store.loadHistory();
        expect(history, hasLength(1));
        expect(history.single.status, AnalysisStatus.completed);
        expect(history.single.method, 'default');
        expect(history.single.counts, <int>[2, 1, 3]);
        expect(await harness.storage.exists(history.single.filePath), isTrue);
      },
    );

    testWidgets('keeps a failed upload and replaces it after Retry succeeds', (
      tester,
    ) async {
      var attempt = 0;
      final backend = _FakeBackendClient(
        baseUrl: _configuredSettings.backendUrl,
        onAnalyze: (file, method) async {
          attempt += 1;
          if (attempt == 1) {
            throw const BackendClientException(
              type: BackendFailureType.network,
              message: 'Backend unavailable.',
            );
          }
          return AnalyzeResponse(
            counts: <int>[4, 0, 1],
            formatted: 'Retry result',
          );
        },
      );
      final harness = await _pumpHome(
        tester,
        settings: _configuredSettings,
        backendClientBuilder: (_) => backend,
        clock: () => _fixedTime,
      );

      await _recordAndStop(tester);

      expect(find.text('Analysis failed'), findsOneWidget);
      expect(find.text('Backend unavailable.'), findsOneWidget);
      final failed = (await harness.store.loadHistory()).single;
      expect(failed.status, AnalysisStatus.failed);
      expect(await harness.storage.exists(failed.filePath), isTrue);

      await tester.tap(find.byKey(ValueKey<String>('retry-${failed.id}')));
      await tester.pumpAndSettle();

      expect(find.text('Retry result'), findsNWidgets(2));
      expect(find.text('Analysis failed'), findsNothing);
      expect(backend.analyzeCalls, 2);
      final retried = (await harness.store.loadHistory()).single;
      expect(retried.id, failed.id);
      expect(retried.status, AnalysisStatus.completed);
      expect(retried.counts, <int>[4, 0, 1]);
    });

    testWidgets('keeps a cancelled upload and allows it to be retried', (
      tester,
    ) async {
      final pending = Completer<AnalyzeResponse>();
      var attempt = 0;
      late _FakeBackendClient backend;
      backend = _FakeBackendClient(
        baseUrl: _configuredSettings.backendUrl,
        onAnalyze: (file, method) {
          attempt += 1;
          if (attempt == 1) return pending.future;
          return Future<AnalyzeResponse>.value(
            AnalyzeResponse(counts: <int>[1], formatted: 'After cancellation'),
          );
        },
        onCancel: () {
          if (!pending.isCompleted) {
            pending.completeError(
              const BackendClientException(
                type: BackendFailureType.cancelled,
                message: 'Audio analysis was cancelled.',
              ),
            );
          }
        },
      );
      final harness = await _pumpHome(
        tester,
        settings: _configuredSettings,
        backendClientBuilder: (_) => backend,
        clock: () => _fixedTime,
      );

      await tester.tap(find.byKey(const ValueKey<String>('record-button')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey<String>('record-button')));
      await tester.pump();
      await tester.pump();

      expect(
        find.byKey(const ValueKey<String>('cancel-analysis-button')),
        findsOneWidget,
      );
      await tester.tap(
        find.byKey(const ValueKey<String>('cancel-analysis-button')),
      );
      await tester.pumpAndSettle();

      expect(backend.cancelCalls, 1);
      expect(find.text('Analysis cancelled'), findsOneWidget);
      expect(
        find.text('Analysis cancelled. The recording was kept for Retry.'),
        findsOneWidget,
      );
      final cancelled = (await harness.store.loadHistory()).single;
      expect(cancelled.status, AnalysisStatus.cancelled);
      expect(await harness.storage.exists(cancelled.filePath), isTrue);

      await tester.tap(find.byKey(ValueKey<String>('retry-${cancelled.id}')));
      await tester.pumpAndSettle();

      expect(find.text('After cancellation'), findsNWidgets(2));
      expect(backend.analyzeCalls, 2);
      expect(
        (await harness.store.loadHistory()).single.status,
        AnalysisStatus.completed,
      );
    });

    testWidgets('sorts history newest-first and removes missing files', (
      tester,
    ) async {
      final harness = await _pumpHome(
        tester,
        settings: _configuredSettings,
        historyBuilder: (storage) async {
          final oldFile = await _writeAudio(storage, 'old.wav');
          final newFile = await _writeAudio(storage, 'new.wav');
          final missingPath = storage.pathFor('gone.wav');
          return <RecordingResult>[
            _result(
              id: 'old',
              timestamp: DateTime.utc(2026, 1, 1),
              formatted: 'Older result',
              filePath: oldFile.path,
            ),
            _result(
              id: 'missing',
              timestamp: DateTime.utc(2026, 3, 1),
              formatted: 'Missing result',
              filePath: missingPath,
            ),
            _result(
              id: 'new',
              timestamp: DateTime.utc(2026, 2, 1),
              formatted: 'Newer result',
              filePath: newFile.path,
            ),
          ];
        },
      );

      expect(find.text('Missing result'), findsNothing);
      expect(find.text('Newer result'), findsOneWidget);
      expect(find.text('Older result'), findsOneWidget);
      expect(
        tester.getTopLeft(find.text('Newer result')).dy,
        lessThan(tester.getTopLeft(find.text('Older result')).dy),
      );

      final persisted = await harness.store.loadHistory();
      expect(persisted.map((item) => item.id), <String>['new', 'old']);
    });

    testWidgets('dismisses history and deletes its audio file', (tester) async {
      late File audioFile;
      final harness = await _pumpHome(
        tester,
        settings: _configuredSettings,
        historyBuilder: (storage) async {
          audioFile = await _writeAudio(storage, 'delete-me.wav');
          return <RecordingResult>[
            _result(
              id: 'delete-me',
              timestamp: _fixedTime,
              formatted: 'Delete me',
              filePath: audioFile.path,
            ),
          ];
        },
      );

      await tester.fling(
        find.byKey(const ValueKey<String>('delete-me')),
        const Offset(-700, 0),
        1200,
      );
      await tester.pumpAndSettle();

      expect(find.text('Delete me'), findsNothing);
      expect(await harness.storage.exists(audioFile.path), isFalse);
      expect(await harness.store.loadHistory(), isEmpty);
    });

    testWidgets('tracks playback play, pause, completion, and replay', (
      tester,
    ) async {
      late File audioFile;
      final playback = _FakePlaybackService();
      await _pumpHome(
        tester,
        settings: _configuredSettings,
        playback: playback,
        historyBuilder: (storage) async {
          audioFile = await _writeAudio(storage, 'play-me.wav');
          return <RecordingResult>[
            _result(
              id: 'play-me',
              timestamp: _fixedTime,
              formatted: 'Playable result',
              filePath: audioFile.path,
            ),
          ];
        },
      );
      const playKey = ValueKey<String>('play-play-me');

      await tester.tap(find.byKey(playKey));
      await tester.pumpAndSettle();
      expect(playback.loadedPaths, <String>[audioFile.path]);
      expect(playback.playCalls, 1);
      expect(find.byTooltip('Pause recording'), findsOneWidget);

      await tester.tap(find.byKey(playKey));
      await tester.pumpAndSettle();
      expect(playback.pauseCalls, 1);
      expect(find.byTooltip('Play recording'), findsOneWidget);

      await tester.tap(find.byKey(playKey));
      await tester.pumpAndSettle();
      expect(playback.playCalls, 2);
      expect(playback.loadedPaths, hasLength(1));

      playback.complete();
      await tester.pump();
      expect(find.byTooltip('Play recording'), findsOneWidget);

      await tester.tap(find.byKey(playKey));
      await tester.pumpAndSettle();
      expect(playback.playCalls, 3);
      expect(playback.loadedPaths, <String>[audioFile.path, audioFile.path]);
    });

    testWidgets('keeps an interrupted recording for a later Retry', (
      tester,
    ) async {
      final backend = _FakeBackendClient(
        baseUrl: _configuredSettings.backendUrl,
      );
      final harness = await _pumpHome(
        tester,
        settings: _configuredSettings,
        backendClientBuilder: (_) => backend,
        clock: () => _fixedTime,
      );

      await tester.tap(find.byKey(const ValueKey<String>('record-button')));
      await tester.pumpAndSettle();
      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.paused);
      await tester.runAsync(() async {
        for (var index = 0; index < 50; index++) {
          if ((await harness.store.loadHistory()).isNotEmpty) return;
          await Future<void>.delayed(const Duration(milliseconds: 1));
        }
        fail('Interrupted recording was not persisted.');
      });
      expect(backend.analyzeCalls, 0);
      final interrupted = (await harness.store.loadHistory()).single;
      expect(interrupted.status, AnalysisStatus.failed);
      expect(await harness.storage.exists(interrupted.filePath), isTrue);

      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.resumed);
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.pump();
      await tester.pumpWidget(
        KeystrokeApp(
          home: KeystrokeHomePage(
            recorder: _FakeRecorderService(),
            playback: _FakePlaybackService(),
            localStore: harness.store,
            backendClientBuilder: (_) => backend,
            recordingStorage: harness.storage,
          ),
        ),
      );
      await tester.pumpAndSettle();
      expect(
        find.text('Recording was interrupted before analysis. Tap Retry.'),
        findsOneWidget,
      );
    });
  });
}

const AppSettings _configuredSettings = AppSettings(
  backendUrl: 'http://192.168.1.20:8000',
  method: 'rule',
);

final DateTime _fixedTime = DateTime.fromMicrosecondsSinceEpoch(
  1720442096789123,
  isUtc: true,
);

Future<_Harness> _pumpHome(
  WidgetTester tester, {
  AppSettings settings = const AppSettings(),
  _FakeRecorderService? recorder,
  _FakePlaybackService? playback,
  BackendClientBuilder? backendClientBuilder,
  Future<List<RecordingResult>> Function(_FakeRecordingStorage storage)?
  historyBuilder,
  Clock? clock,
}) async {
  SharedPreferences.setMockInitialValues(<String, Object>{});
  final preferences = await SharedPreferences.getInstance();
  final store = LocalStore(preferences);
  final storage = _FakeRecordingStorage();
  final testRecorder = recorder ?? _FakeRecorderService();
  testRecorder.onStart = storage.writeAudio;
  final testPlayback = playback ?? _FakePlaybackService();
  final fallbackBackend = _FakeBackendClient(
    baseUrl: settings.backendUrl.isEmpty
        ? 'http://127.0.0.1:8000'
        : settings.backendUrl,
  );

  await store.saveSettings(settings);
  await store.saveHistory(
    historyBuilder == null
        ? <RecordingResult>[]
        : await historyBuilder(storage),
  );

  await tester.pumpWidget(
    KeystrokeApp(
      home: KeystrokeHomePage(
        recorder: testRecorder,
        playback: testPlayback,
        localStore: store,
        backendClientBuilder: backendClientBuilder ?? (_) => fallbackBackend,
        recordingStorage: storage,
        clock: clock,
      ),
    ),
  );
  await tester.pumpAndSettle();

  return _Harness(
    recorder: testRecorder,
    playback: testPlayback,
    store: store,
    storage: storage,
  );
}

Future<void> _recordAndStop(WidgetTester tester) async {
  final recordButton = find.byKey(const ValueKey<String>('record-button'));
  await tester.tap(recordButton);
  await tester.pumpAndSettle();
  expect(find.text('Stop Recording'), findsOneWidget);

  await tester.tap(recordButton);
  await tester.pumpAndSettle();
}

Future<File> _writeAudio(_FakeRecordingStorage storage, String name) async =>
    storage.addAudio(name);

RecordingResult _result({
  required String id,
  required DateTime timestamp,
  required String formatted,
  required String filePath,
}) => RecordingResult(
  id: id,
  timestamp: timestamp,
  formatted: formatted,
  counts: <int>[1, 2],
  method: 'rule',
  filePath: filePath,
);

class _Harness {
  const _Harness({
    required this.recorder,
    required this.playback,
    required this.store,
    required this.storage,
  });

  final _FakeRecorderService recorder;
  final _FakePlaybackService playback;
  final LocalStore store;
  final _FakeRecordingStorage storage;
}

class _FakeRecordingStorage implements RecordingStorage {
  final Map<String, List<int>> _files = <String, List<int>>{};

  String pathFor(String name) => 'memory/recordings/$name';

  File addAudio(String name) {
    final file = File(pathFor(name));
    writeAudio(file.path);
    return file;
  }

  void writeAudio(String path) {
    _files[path] = <int>[82, 73, 70, 70, 87, 65, 86, 69];
  }

  @override
  Future<File> allocate(DateTime timestamp, int sequence) async {
    final filename =
        'recording_${timestamp.microsecondsSinceEpoch}_${sequence.toString().padLeft(3, '0')}.wav';
    final file = File(pathFor(filename));
    _files[file.path] = <int>[];
    return file;
  }

  @override
  Future<void> delete(String path) async {
    _files.remove(path);
  }

  @override
  Future<bool> exists(String path) async => _files.containsKey(path);

  @override
  Future<bool> hasAudio(File file) async =>
      _files[file.path]?.isNotEmpty ?? false;
}

class _FakeRecorderService implements RecorderService {
  _FakeRecorderService({this.permissionGranted = true});

  final bool permissionGranted;
  void Function(String path)? onStart;
  final List<String> startedPaths = <String>[];
  int permissionChecks = 0;
  int stopCalls = 0;
  int cancelCalls = 0;
  int disposeCalls = 0;
  String? _activePath;

  @override
  Future<bool> hasPermission() async {
    permissionChecks += 1;
    return permissionGranted;
  }

  @override
  Future<void> start(String path) async {
    _activePath = path;
    startedPaths.add(path);
    onStart?.call(path);
  }

  @override
  Future<String?> stop() async {
    stopCalls += 1;
    final path = _activePath;
    _activePath = null;
    return path;
  }

  @override
  Future<void> cancel() async {
    cancelCalls += 1;
    _activePath = null;
  }

  @override
  Future<void> dispose() async {
    disposeCalls += 1;
  }
}

class _FakePlaybackService implements PlaybackService {
  final StreamController<PlaybackStatus> _statuses =
      StreamController<PlaybackStatus>.broadcast(sync: true);
  final List<String> loadedPaths = <String>[];
  PlaybackStatus _status = PlaybackStatus.stopped;
  int playCalls = 0;
  int pauseCalls = 0;
  int stopCalls = 0;
  int disposeCalls = 0;

  @override
  PlaybackStatus get status => _status;

  @override
  Stream<PlaybackStatus> get statusStream => _statuses.stream;

  @override
  Future<void> load(String path) async {
    loadedPaths.add(path);
  }

  @override
  Future<void> play() async {
    playCalls += 1;
    _emit(PlaybackStatus.playing);
  }

  @override
  Future<void> pause() async {
    pauseCalls += 1;
    _emit(PlaybackStatus.paused);
  }

  @override
  Future<void> stop() async {
    stopCalls += 1;
    _emit(PlaybackStatus.stopped);
  }

  void complete() => _emit(PlaybackStatus.completed);

  void _emit(PlaybackStatus status) {
    _status = status;
    if (!_statuses.isClosed) _statuses.add(status);
  }

  @override
  Future<void> dispose() async {
    disposeCalls += 1;
    await _statuses.close();
  }
}

typedef _AnalyzeHandler =
    Future<AnalyzeResponse> Function(File file, String method);

class _FakeBackendClient implements BackendClient {
  _FakeBackendClient({required String baseUrl, this.onAnalyze, this.onCancel})
    : baseUri = Uri.parse(baseUrl);

  @override
  final Uri baseUri;
  final _AnalyzeHandler? onAnalyze;
  final void Function()? onCancel;
  final List<String> analyzedMethods = <String>[];
  final List<String> analyzedPaths = <String>[];
  bool _requestActive = false;
  int analyzeCalls = 0;
  int connectionTests = 0;
  int cancelCalls = 0;

  @override
  String get baseUrl => baseUri.toString();

  @override
  int get activeRequestCount => _requestActive ? 1 : 0;

  @override
  Future<void> testConnection() async {
    connectionTests += 1;
    _requestActive = true;
    try {
      await Future<void>.value();
    } finally {
      _requestActive = false;
    }
  }

  @override
  Future<ModelInfo> fetchModelInfo() async {
    return const ModelInfo(
      modelName: 'ExtraTrees',
      featureSet: 'D',
      normalization: 'P2',
      classesCount: 27,
      accuracy: 0.909,
      macroF1: 0.914,
      top3Accuracy: 0.971,
    );
  }

  @override
  Future<AnalyzeResponse> analyze({
    required File file,
    String method = 'default',
  }) async {
    analyzeCalls += 1;
    analyzedPaths.add(file.path);
    analyzedMethods.add(method);
    _requestActive = true;
    try {
      final handler = onAnalyze;
      if (handler != null) return await handler(file, method);
      return AnalyzeResponse(counts: <int>[1], formatted: 'Default result');
    } finally {
      _requestActive = false;
    }
  }

  @override
  void cancelActiveRequests() {
    cancelCalls += 1;
    onCancel?.call();
  }
}
