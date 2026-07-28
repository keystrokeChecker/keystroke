import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';

import 'audio_services.dart';
import 'backend_client.dart';
import 'local_store.dart';
import 'recording_result.dart';
import 'recording_storage.dart';

typedef BackendClientBuilder = BackendClient Function(String baseUrl);
typedef Clock = DateTime Function();

const Set<String> supportedPredictionMethods = <String>{'rule', 'ml', 'yamnet'};

class KeystrokeHomePage extends StatefulWidget {
  const KeystrokeHomePage({
    super.key,
    this.recorder,
    this.playback,
    this.localStore,
    this.backendClientBuilder,
    this.documentsDirectoryProvider,
    this.recordingStorage,
    this.clock,
  });

  final RecorderService? recorder;
  final PlaybackService? playback;
  final LocalStore? localStore;
  final BackendClientBuilder? backendClientBuilder;
  final DocumentsDirectoryProvider? documentsDirectoryProvider;
  final RecordingStorage? recordingStorage;
  final Clock? clock;

  @override
  State<KeystrokeHomePage> createState() => _KeystrokeHomePageState();
}

class _KeystrokeHomePageState extends State<KeystrokeHomePage>
    with WidgetsBindingObserver {
  late final RecorderService _recorder;
  late final PlaybackService _playback;
  late final BackendClientBuilder _backendClientBuilder;
  late final RecordingStorage _recordingStorage;
  late final Clock _clock;

  LocalStore? _localStore;
  StreamSubscription<PlaybackStatus>? _playbackSubscription;
  final List<RecordingResult> _history = <RecordingResult>[];

  bool _isInitializing = true;
  bool _isStarting = false;
  bool _isRecording = false;
  bool _isStopping = false;
  bool _isAnalyzing = false;
  bool _isCancelling = false;
  bool _isRetrying = false;
  bool _isPlaybackLoading = false;
  String _statusMessage = 'Loading settings and history...';
  String _resultText = '';
  String _backendUrl = '';
  String _selectedMethod = 'rule';

  String? _currentRecordingPath;
  String? _currentRecordingId;
  DateTime? _recordingStartedAt;
  int _recordingSequence = 0;
  bool _appIsActive = true;

  BackendClient? _activeBackendClient;
  String? _loadedRecordingId;
  PlaybackStatus _playbackStatus = PlaybackStatus.stopped;
  Future<void> _historySaveTail = Future<void>.value();
  Future<void>? _activeRecorderOperation;

  bool get _isBusy =>
      _isStarting ||
      _isStopping ||
      _isAnalyzing ||
      _isRetrying ||
      _isPlaybackLoading;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _recorder = widget.recorder ?? DeviceRecorderService();
    _playback = widget.playback ?? DevicePlaybackService();
    _backendClientBuilder =
        widget.backendClientBuilder ??
        (baseUrl) => BackendClient(baseUrl: baseUrl);
    _recordingStorage =
        widget.recordingStorage ??
        DeviceRecordingStorage(
          documentsDirectoryProvider: widget.documentsDirectoryProvider,
        );
    _clock = widget.clock ?? DateTime.now;
    final lifecycleState = WidgetsBinding.instance.lifecycleState;
    _appIsActive =
        lifecycleState == null || lifecycleState == AppLifecycleState.resumed;
    _playbackStatus = _playback.status;
    _playbackSubscription = _playback.statusStream.listen(
      _handlePlaybackStatus,
      onError: _handlePlaybackStreamError,
    );
    unawaited(_initialize());
  }

  Future<void> _initialize() async {
    late final LocalStore store;
    try {
      store = widget.localStore ?? await LocalStore.create();
      _localStore = store;
    } on Object {
      if (!mounted) return;
      setState(() {
        _isInitializing = false;
        _statusMessage =
            'Ready, but local settings and history are unavailable.';
      });
      return;
    }

    AppSettings settings;
    try {
      settings = await store.loadSettings();
    } on Object {
      settings = const AppSettings();
    }

    List<RecordingResult> loadedHistory;
    try {
      loadedHistory = await store.loadHistory();
    } on Object {
      loadedHistory = <RecordingResult>[];
    }

    final availableHistory = <RecordingResult>[];
    for (final recording in loadedHistory) {
      if (await _fileExists(recording.filePath)) {
        availableHistory.add(recording);
      }
    }
    availableHistory.sort(_newestFirst);

    final method = supportedPredictionMethods.contains(settings.method)
        ? settings.method
        : 'rule';
    var backendUrl = '';
    if (settings.backendUrl.trim().isNotEmpty) {
      try {
        backendUrl = BackendClient.normalizeBaseUrl(
          settings.backendUrl,
        ).toString();
      } on Object {
        backendUrl = '';
      }
    }
    _history
      ..clear()
      ..addAll(availableHistory);

    if (availableHistory.length != loadedHistory.length) {
      try {
        await store.saveHistory(availableHistory);
      } on Object {
        // The valid in-memory entries can still be shown this session.
      }
    }
    if (method != settings.method || backendUrl != settings.backendUrl) {
      try {
        await store.saveSettings(
          AppSettings(backendUrl: backendUrl, method: method),
        );
      } on Object {
        // Sanitized settings still remain safe for this session.
      }
    }

    if (!mounted) return;
    setState(() {
      _backendUrl = backendUrl;
      _selectedMethod = method;
      _isInitializing = false;
      _statusMessage = _backendUrl.isEmpty
          ? 'Configure the backend URL in Settings to begin.'
          : 'Ready to record.';
    });
  }

  Future<void> _runRecorderOperation(
    Future<void> Function() operation, {
    bool replaceActive = false,
  }) {
    final active = _activeRecorderOperation;
    if (active != null && !replaceActive) return active;

    late final Future<void> tracked;
    tracked = operation().whenComplete(() {
      if (identical(_activeRecorderOperation, tracked)) {
        _activeRecorderOperation = null;
      }
    });
    _activeRecorderOperation = tracked;
    return tracked;
  }

  void _finishStopping() {
    if (mounted) {
      setState(() {
        _isRecording = false;
        _isStopping = false;
      });
    } else {
      _isRecording = false;
      _isStopping = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    _appIsActive = state == AppLifecycleState.resumed;
    final interrupted =
        state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused ||
        state == AppLifecycleState.hidden ||
        state == AppLifecycleState.detached;
    if (!interrupted) return;

    if (_isRecording && !_isStopping) {
      unawaited(
        _runRecorderOperation(
          () => _stopRecording(analyze: false),
          replaceActive: true,
        ),
      );
    } else if (_isStarting) {
      unawaited(_stopForLifecycleInterruption());
    }
    if (_playbackStatus == PlaybackStatus.playing) {
      unawaited(_pausePlayback());
    }
  }

  Future<void> _stopForLifecycleInterruption() async {
    final active = _activeRecorderOperation;
    if (active != null) {
      try {
        await active;
      } on Object {
        // Start/stop failures are already converted to visible app status.
      }
    }
    if (_isRecording && !_isStopping) {
      await _runRecorderOperation(() => _stopRecording(analyze: false));
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _activeBackendClient?.cancelActiveRequests();
    unawaited(_playbackSubscription?.cancel());
    unawaited(_disposeServices());
    super.dispose();
  }

  Future<void> _disposeServices() async {
    final recorderOperation = _activeRecorderOperation;
    if (recorderOperation != null) {
      try {
        await recorderOperation;
      } on Object {
        // Recorder operations surface their own user-facing failures.
      }
    }
    final shouldCancelRecording = _isRecording || _isStarting;
    final unfinishedPath = shouldCancelRecording ? _currentRecordingPath : null;
    try {
      if (shouldCancelRecording) await _recorder.cancel();
    } on Object {
      // Native cleanup is best effort during widget teardown.
    }
    if (unfinishedPath != null) await _tryDeleteFile(unfinishedPath);
    try {
      await _recorder.dispose();
    } on Object {
      // Native cleanup is best effort during widget teardown.
    }
    try {
      await _playback.dispose();
    } on Object {
      // Native cleanup is best effort during widget teardown.
    }
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      await _runRecorderOperation(() => _stopRecording(analyze: true));
    } else {
      await _runRecorderOperation(_startRecording);
    }
  }

  Future<void> _startRecording() async {
    if (_isInitializing || _isBusy || _isRecording) return;
    if (_backendUrl.isEmpty) {
      _setStatus('Configure the backend URL in Settings before recording.');
      return;
    }

    setState(() {
      _isStarting = true;
      _statusMessage = 'Preparing the microphone...';
    });

    try {
      if (_playbackStatus == PlaybackStatus.playing) {
        await _playback.stop();
        _loadedRecordingId = null;
        _playbackStatus = PlaybackStatus.stopped;
      }
      final hasPermission = await _recorder.hasPermission();
      if (!mounted) return;
      if (!hasPermission) {
        setState(() {
          _isStarting = false;
          _statusMessage = 'Microphone permission is required to record.';
        });
        return;
      }
      if (!_appIsActive) {
        setState(() {
          _isStarting = false;
          _statusMessage =
              'Recording was not started while the app was inactive.';
        });
        return;
      }

      final startedAt = _clock();
      final file = await _buildRecordingFile(startedAt);
      if (!mounted) {
        await _deleteFile(file.path);
        return;
      }

      final id = _idFromFile(file);
      _currentRecordingPath = file.path;
      _currentRecordingId = id;
      _recordingStartedAt = startedAt;
      await _recorder.start(file.path);
      if (!mounted) return;
      setState(() {
        _isStarting = false;
        _isRecording = true;
        _resultText = '';
        _statusMessage = 'Recording... tap Stop when you are finished.';
      });
      if (!_appIsActive) {
        await _stopRecording(analyze: false);
      }
    } catch (error) {
      final path = _currentRecordingPath;
      _clearCurrentRecording();
      try {
        await _recorder.cancel();
      } on Object {
        // Starting failed, so native cancellation is best effort.
      }
      if (path != null) await _tryDeleteFile(path);
      if (mounted) {
        setState(() {
          _isStarting = false;
          _isRecording = false;
          _statusMessage =
              'Could not start recording: ${_friendlyError(error)}';
        });
      } else {
        _isStarting = false;
        _isRecording = false;
      }
    }
  }

  Future<void> _stopRecording({required bool analyze}) async {
    if (_isStopping || !_isRecording) return;

    final expectedPath = _currentRecordingPath;
    final id = _currentRecordingId;
    final startedAt = _recordingStartedAt;
    Object? stopError;
    String? stoppedPath;

    setState(() {
      _isStopping = true;
      _statusMessage = 'Stopping recording...';
    });

    try {
      stoppedPath = await _recorder.stop();
    } catch (error) {
      stopError = error;
    } finally {
      _clearCurrentRecording();
      if (mounted) {
        setState(() => _isRecording = false);
      } else {
        _isRecording = false;
      }
    }

    final path = stoppedPath ?? expectedPath;
    if (path == null || id == null || startedAt == null) {
      _finishStopping();
      _setStatus('Recording stopped, but no audio file was produced.');
      return;
    }

    if (expectedPath != null && expectedPath != path) {
      await _tryDeleteFile(expectedPath);
    }

    final file = File(path);
    if (!await _fileHasAudio(file)) {
      await _tryDeleteFile(path);
      _finishStopping();
      _setStatus('Recording stopped, but the audio file was empty or missing.');
      return;
    }

    if (stopError != null) {
      _finishStopping();
      await _storeFailedRecording(
        id: id,
        timestamp: startedAt,
        method: _selectedMethod,
        file: file,
        message:
            'Recording stopped with an error: ${_friendlyError(stopError)}',
      );
      return;
    }

    if (!analyze) {
      _finishStopping();
      await _storeFailedRecording(
        id: id,
        timestamp: startedAt,
        method: _selectedMethod,
        file: file,
        message: 'Recording was interrupted before analysis. Tap Retry.',
      );
      return;
    }

    _finishStopping();
    await _analyzeRecording(
      id: id,
      timestamp: startedAt,
      method: _selectedMethod,
      file: file,
    );
  }

  Future<void> _analyzeRecording({
    required String id,
    required DateTime timestamp,
    required String method,
    required File file,
  }) async {
    if (_isBusy || _isRecording) {
      await _storeFailedRecording(
        id: id,
        timestamp: timestamp,
        method: method,
        file: file,
        message:
            'Another operation prevented analysis from starting. Tap Retry.',
      );
      return;
    }
    if (_backendUrl.isEmpty) {
      await _storeFailedRecording(
        id: id,
        timestamp: timestamp,
        method: method,
        file: file,
        message: 'Backend URL is not configured. Open Settings, then Retry.',
      );
      return;
    }

    late final BackendClient client;
    try {
      client = _backendClientBuilder(_backendUrl);
    } on Object catch (error) {
      await _storeFailedRecording(
        id: id,
        timestamp: timestamp,
        method: method,
        file: file,
        message: 'Backend URL is invalid: ${_friendlyError(error)}',
      );
      return;
    }

    _activeBackendClient = client;
    if (mounted) {
      setState(() {
        _isAnalyzing = true;
        _isCancelling = false;
        _statusMessage = 'Uploading audio and waiting for analysis...';
      });
    }

    RecordingResult result;
    String finalStatus;
    try {
      final response = await client.analyze(file: file, method: method);
      result = RecordingResult(
        id: id,
        timestamp: timestamp,
        formatted: response.formatted,
        counts: response.counts,
        method: method,
        filePath: file.path,
      );
      finalStatus = 'Analysis complete.';
      _resultText = response.formatted;
    } on BackendClientException catch (error) {
      final cancelled = error.type == BackendFailureType.cancelled;
      result = RecordingResult(
        id: id,
        timestamp: timestamp,
        formatted: cancelled ? 'Analysis cancelled' : 'Analysis failed',
        counts: const <int>[],
        method: method,
        filePath: file.path,
        status: cancelled ? AnalysisStatus.cancelled : AnalysisStatus.failed,
        errorMessage: error.message,
      );
      finalStatus = cancelled
          ? 'Analysis cancelled. The recording was kept for Retry.'
          : '${error.message} The recording was kept for Retry.';
    } on Object catch (error) {
      result = RecordingResult(
        id: id,
        timestamp: timestamp,
        formatted: 'Analysis failed',
        counts: const <int>[],
        method: method,
        filePath: file.path,
        status: AnalysisStatus.failed,
        errorMessage: _friendlyError(error),
      );
      finalStatus =
          'Analysis failed: ${_friendlyError(error)}. The recording was kept for Retry.';
    } finally {
      if (identical(_activeBackendClient, client)) {
        _activeBackendClient = null;
      }
      if (mounted) {
        setState(() {
          _isAnalyzing = false;
          _isCancelling = false;
        });
      }
    }

    final saved = await _upsertHistory(result);
    _setStatus(
      saved ? finalStatus : '$finalStatus History could not be saved.',
    );
  }

  Future<void> _storeFailedRecording({
    required String id,
    required DateTime timestamp,
    required String method,
    required File file,
    required String message,
  }) async {
    final result = RecordingResult(
      id: id,
      timestamp: timestamp,
      formatted: 'Analysis failed',
      counts: const <int>[],
      method: method,
      filePath: file.path,
      status: AnalysisStatus.failed,
      errorMessage: message,
    );
    final saved = await _upsertHistory(result);
    _setStatus(saved ? message : '$message History could not be saved.');
  }

  void _cancelAnalysis() {
    final client = _activeBackendClient;
    if (client == null || _isCancelling) return;
    setState(() {
      _isCancelling = true;
      _statusMessage = 'Cancelling analysis...';
    });
    client.cancelActiveRequests();
  }

  Future<void> _retryRecording(RecordingResult item) async {
    if (_isBusy || _isRecording) return;
    setState(() => _isRetrying = true);
    try {
      final file = File(item.filePath);
      if (!await _fileHasAudio(file)) {
        final unavailable = item.copyWith(
          formatted: 'Analysis failed',
          counts: const <int>[],
          status: AnalysisStatus.failed,
          errorMessage: 'The saved audio file is missing or empty.',
        );
        final saved = await _upsertHistory(unavailable);
        _setStatus(
          saved
              ? 'The saved audio file is missing or empty.'
              : 'The saved audio file is missing or empty, and history could not be saved.',
        );
        return;
      }

      final method = supportedPredictionMethods.contains(item.method)
          ? item.method
          : _selectedMethod;
      if (mounted) {
        setState(() => _isRetrying = false);
      } else {
        _isRetrying = false;
      }
      await _analyzeRecording(
        id: item.id,
        timestamp: item.timestamp,
        method: method,
        file: file,
      );
    } finally {
      if (_isRetrying) {
        if (mounted) {
          setState(() => _isRetrying = false);
        } else {
          _isRetrying = false;
        }
      }
    }
  }

  Future<bool> _upsertHistory(RecordingResult result) async {
    void update() {
      final index = _history.indexWhere((item) => item.id == result.id);
      if (index == -1) {
        _history.add(result);
      } else {
        _history[index] = result;
      }
      _history.sort(_newestFirst);
    }

    if (mounted) {
      setState(update);
    } else {
      update();
    }
    return _persistHistory();
  }

  Future<bool> _persistHistory() async {
    final store = _localStore;
    if (store == null) return false;
    final snapshot = List<RecordingResult>.unmodifiable(_history);
    final save = _historySaveTail.then<void>(
      (_) => store.saveHistory(snapshot),
      onError: (Object _, StackTrace _) => store.saveHistory(snapshot),
    );
    _historySaveTail = save;
    try {
      await save;
      return true;
    } on Object {
      return false;
    }
  }

  Future<void> _openSettings() async {
    if (_isInitializing || _isRecording || _isBusy) return;
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => _SettingsSheet(
        initialSettings: AppSettings(
          backendUrl: _backendUrl,
          method: _selectedMethod,
        ),
        backendClientBuilder: _backendClientBuilder,
        onSave: _saveSettings,
      ),
    );
  }

  Future<String?> _saveSettings(AppSettings settings) async {
    final store = _localStore;
    if (store == null) {
      return 'Local settings storage is unavailable.';
    }
    try {
      await store.saveSettings(settings);
    } on Object catch (error) {
      return 'Could not save settings: ${_friendlyError(error)}';
    }

    if (!mounted) return null;
    setState(() {
      _backendUrl = settings.backendUrl;
      _selectedMethod = settings.method;
      _statusMessage = settings.backendUrl.isEmpty
          ? 'Backend configuration cleared.'
          : 'Settings saved. Ready to record.';
    });
    return null;
  }

  void _handlePlaybackStatus(PlaybackStatus status) {
    if (!mounted) return;
    setState(() {
      _playbackStatus = status;
      if (status == PlaybackStatus.completed) {
        _statusMessage = 'Playback complete.';
      }
    });
  }

  void _handlePlaybackStreamError(Object error, StackTrace stackTrace) {
    _setStatus('Playback failed: ${_friendlyError(error)}');
  }

  Future<void> _togglePlayback(RecordingResult item) async {
    if (_isRecording || _isBusy) return;
    setState(() => _isPlaybackLoading = true);
    try {
      if (_loadedRecordingId == item.id &&
          _playbackStatus == PlaybackStatus.playing) {
        await _playback.pause();
        if (mounted) {
          setState(() {
            _playbackStatus = PlaybackStatus.paused;
            _statusMessage = 'Playback paused.';
          });
        }
        return;
      }

      if (!await _fileHasAudio(File(item.filePath))) {
        _setStatus('The saved audio file is missing or empty.');
        return;
      }

      final mustReload =
          _loadedRecordingId != item.id ||
          _playbackStatus == PlaybackStatus.completed ||
          _playbackStatus == PlaybackStatus.stopped;
      if (mustReload) {
        await _playback.stop();
        await _playback.load(item.filePath);
        _loadedRecordingId = item.id;
      }
      if (mounted) {
        setState(() {
          _playbackStatus = PlaybackStatus.playing;
          _statusMessage = 'Playing saved recording.';
        });
      }
      final playFuture = _playback.play();
      unawaited(
        playFuture.onError((Object error, StackTrace stackTrace) {
          _setStatus('Playback failed: ${_friendlyError(error)}');
        }),
      );
    } on Object catch (error) {
      _setStatus('Playback failed: ${_friendlyError(error)}');
    } finally {
      if (mounted) {
        setState(() => _isPlaybackLoading = false);
      } else {
        _isPlaybackLoading = false;
      }
    }
  }

  Future<void> _pausePlayback() async {
    try {
      await _playback.pause();
      if (mounted) setState(() => _playbackStatus = PlaybackStatus.paused);
    } on Object catch (error) {
      _setStatus('Playback failed: ${_friendlyError(error)}');
    }
  }

  void _dismissRecording(RecordingResult item) {
    if (mounted) {
      setState(() => _history.removeWhere((entry) => entry.id == item.id));
    } else {
      _history.removeWhere((entry) => entry.id == item.id);
    }
    unawaited(_deleteDismissedRecording(item));
  }

  Future<void> _deleteDismissedRecording(RecordingResult item) async {
    Object? deletionError;
    if (_loadedRecordingId == item.id) {
      try {
        await _playback.stop();
      } on Object catch (error) {
        deletionError = error;
      }
      _loadedRecordingId = null;
      if (mounted) setState(() => _playbackStatus = PlaybackStatus.stopped);
    }
    try {
      await _deleteFile(item.filePath);
    } on Object catch (error) {
      deletionError ??= error;
    }
    final saved = await _persistHistory();
    if (deletionError != null) {
      _setStatus(
        'History entry removed, but its audio file could not be deleted.',
      );
    } else if (!saved) {
      _setStatus('History entry removed, but history could not be saved.');
    }
  }

  Future<File> _buildRecordingFile(DateTime timestamp) async {
    _recordingSequence += 1;
    return _recordingStorage.allocate(timestamp, _recordingSequence);
  }

  String _idFromFile(File file) {
    final segments = file.path.split(RegExp(r'[/\\]'));
    final filename = segments.isEmpty ? file.path : segments.last;
    return filename.endsWith('.wav')
        ? filename.substring(0, filename.length - 4)
        : filename;
  }

  void _clearCurrentRecording() {
    _currentRecordingPath = null;
    _currentRecordingId = null;
    _recordingStartedAt = null;
  }

  Future<bool> _fileExists(String path) async {
    try {
      return await _recordingStorage.exists(path);
    } on Object {
      return false;
    }
  }

  Future<bool> _fileHasAudio(File file) async {
    try {
      return await _recordingStorage.hasAudio(file);
    } on Object {
      return false;
    }
  }

  Future<void> _deleteFile(String path) => _recordingStorage.delete(path);

  Future<bool> _tryDeleteFile(String path) async {
    try {
      await _deleteFile(path);
      return true;
    } on Object {
      return false;
    }
  }

  int _newestFirst(RecordingResult left, RecordingResult right) =>
      right.timestamp.compareTo(left.timestamp);

  String _friendlyError(Object error) {
    if (error is BackendClientException) return error.message;
    if (error is ArgumentError) return error.message?.toString() ?? '$error';
    if (error is FileSystemException) return error.message;
    return error.toString();
  }

  void _setStatus(String message) {
    if (!mounted) return;
    setState(() => _statusMessage = message);
  }

  String _formatTimestamp(DateTime timestamp) {
    final local = timestamp.toLocal();
    String two(int value) => value.toString().padLeft(2, '0');
    return '${local.year}-${two(local.month)}-${two(local.day)} '
        '${two(local.hour)}:${two(local.minute)}';
  }

  @override
  Widget build(BuildContext context) {
    final controlsDisabled = _isInitializing || _isBusy;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Keystroke Sound Analyzer'),
        actions: <Widget>[
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: 'Settings',
            onPressed: controlsDisabled || _isRecording ? null : _openSettings,
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              if (_backendUrl.isEmpty && !_isInitializing)
                Card(
                  color: Theme.of(context).colorScheme.errorContainer,
                  child: ListTile(
                    leading: const Icon(Icons.settings_ethernet),
                    title: const Text('Backend setup required'),
                    subtitle: const Text(
                      'Open Settings and enter the computer address running the API.',
                    ),
                    trailing: TextButton(
                      onPressed: _openSettings,
                      child: const Text('SET UP'),
                    ),
                  ),
                ),
              const Text(
                'Keystrokes Detected',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey,
                  letterSpacing: 1.2,
                ),
              ),
              const SizedBox(height: 6),
              Card(
                color: Colors.deepPurple.shade100,
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    vertical: 20,
                    horizontal: 12,
                  ),
                  child: Center(
                    child: Text(
                      _resultText.isEmpty ? '—' : _resultText,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: Colors.deepPurple,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 14),
              ElevatedButton.icon(
                key: const ValueKey<String>('record-button'),
                icon: Icon(_isRecording ? Icons.stop : Icons.mic),
                label: Text(
                  _isRecording ? 'Stop Recording' : 'Start Recording',
                ),
                onPressed: controlsDisabled ? null : _toggleRecording,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isRecording ? Colors.redAccent : null,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),
              if (_isAnalyzing) ...<Widget>[
                const SizedBox(height: 8),
                OutlinedButton.icon(
                  key: const ValueKey<String>('cancel-analysis-button'),
                  onPressed: _isCancelling ? null : _cancelAnalysis,
                  icon: const Icon(Icons.cancel_outlined),
                  label: Text(
                    _isCancelling ? 'Cancelling...' : 'Cancel Analysis',
                  ),
                ),
              ],
              if (_isInitializing || _isStopping || _isAnalyzing) ...<Widget>[
                const SizedBox(height: 8),
                const LinearProgressIndicator(),
              ],
              const SizedBox(height: 10),
              Semantics(
                liveRegion: true,
                child: Text(
                  _statusMessage,
                  key: const ValueKey<String>('status-message'),
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.deepPurple.shade700,
                  ),
                ),
              ),
              const SizedBox(height: 14),
              const Text(
                'History',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Colors.deepPurple,
                ),
              ),
              const SizedBox(height: 6),
              Expanded(
                child: _history.isEmpty
                    ? const Center(child: Text('No saved recordings yet.'))
                    : ListView.builder(
                        itemCount: _history.length,
                        itemBuilder: (context, index) {
                          final item = _history[index];
                          final completed =
                              item.status == AnalysisStatus.completed;
                          final isPlaying =
                              _loadedRecordingId == item.id &&
                              _playbackStatus == PlaybackStatus.playing;
                          return Dismissible(
                            key: ValueKey<String>(item.id),
                            direction: controlsDisabled || _isRecording
                                ? DismissDirection.none
                                : DismissDirection.endToStart,
                            background: Container(
                              color: Colors.redAccent,
                              alignment: Alignment.centerRight,
                              padding: const EdgeInsets.symmetric(
                                horizontal: 20,
                              ),
                              child: const Icon(
                                Icons.delete,
                                color: Colors.white,
                              ),
                            ),
                            onDismissed: (_) => _dismissRecording(item),
                            child: Card(
                              child: Padding(
                                padding: const EdgeInsets.all(12),
                                child: Row(
                                  children: <Widget>[
                                    Icon(
                                      completed
                                          ? Icons.check_circle_outline
                                          : item.status ==
                                                AnalysisStatus.cancelled
                                          ? Icons.cancel_outlined
                                          : Icons.error_outline,
                                      color: completed
                                          ? Colors.green
                                          : Theme.of(context).colorScheme.error,
                                    ),
                                    const SizedBox(width: 10),
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: <Widget>[
                                          Text(
                                            item.formatted,
                                            style: const TextStyle(
                                              fontSize: 17,
                                              fontWeight: FontWeight.w600,
                                            ),
                                          ),
                                          const SizedBox(height: 3),
                                          Text(
                                            completed
                                                ? '${item.method.toUpperCase()} • ${item.counts.join('|')}'
                                                : item.method.toUpperCase(),
                                            style: TextStyle(
                                              fontSize: 12,
                                              color: Colors.deepPurple.shade600,
                                            ),
                                          ),
                                          if (item.errorMessage != null) ...[
                                            const SizedBox(height: 3),
                                            Text(
                                              item.errorMessage!,
                                              maxLines: 2,
                                              overflow: TextOverflow.ellipsis,
                                              style: TextStyle(
                                                fontSize: 12,
                                                color: Theme.of(
                                                  context,
                                                ).colorScheme.error,
                                              ),
                                            ),
                                          ],
                                          Text(
                                            _formatTimestamp(item.timestamp),
                                            style: TextStyle(
                                              fontSize: 11,
                                              color: Colors.grey.shade600,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    if (!completed)
                                      IconButton(
                                        key: ValueKey<String>(
                                          'retry-${item.id}',
                                        ),
                                        icon: const Icon(Icons.refresh),
                                        tooltip: 'Retry analysis',
                                        onPressed:
                                            controlsDisabled || _isRecording
                                            ? null
                                            : () => _retryRecording(item),
                                      ),
                                    IconButton(
                                      key: ValueKey<String>('play-${item.id}'),
                                      icon: Icon(
                                        isPlaying
                                            ? Icons.pause
                                            : Icons.play_arrow,
                                      ),
                                      tooltip: isPlaying
                                          ? 'Pause recording'
                                          : 'Play recording',
                                      onPressed:
                                          controlsDisabled || _isRecording
                                          ? null
                                          : () => _togglePlayback(item),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SettingsSheet extends StatefulWidget {
  const _SettingsSheet({
    required this.initialSettings,
    required this.backendClientBuilder,
    required this.onSave,
  });

  final AppSettings initialSettings;
  final BackendClientBuilder backendClientBuilder;
  final Future<String?> Function(AppSettings settings) onSave;

  @override
  State<_SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<_SettingsSheet> {
  late final TextEditingController _backendController;
  late String _method;
  bool _isTesting = false;
  bool _isSaving = false;
  String? _feedback;
  bool _feedbackIsError = false;
  BackendClient? _testingClient;

  @override
  void initState() {
    super.initState();
    _backendController = TextEditingController(
      text: widget.initialSettings.backendUrl,
    );
    _method = widget.initialSettings.method;
  }

  @override
  void dispose() {
    _testingClient?.cancelActiveRequests();
    _backendController.dispose();
    super.dispose();
  }

  String? _normalizeUrl({required bool allowEmpty}) {
    final value = _backendController.text.trim();
    if (value.isEmpty && allowEmpty) return '';
    if (value.isEmpty) {
      _showFeedback('Enter a backend URL first.', isError: true);
      return null;
    }
    try {
      return BackendClient.normalizeBaseUrl(value).toString();
    } on Object {
      _showFeedback(
        'Enter a valid address, for example http://192.168.1.20:8000.',
        isError: true,
      );
      return null;
    }
  }

  Future<void> _testConnection() async {
    final normalized = _normalizeUrl(allowEmpty: false);
    if (normalized == null) return;
    setState(() {
      _isTesting = true;
      _feedback = 'Testing connection...';
      _feedbackIsError = false;
    });
    try {
      final client = widget.backendClientBuilder(normalized);
      _testingClient = client;
      await client.testConnection();
      if (!mounted) return;
      _backendController.text = normalized;
      _showFeedback('Connection successful.', isError: false);
    } on BackendClientException catch (error) {
      _showFeedback(error.message, isError: true);
    } on Object catch (error) {
      _showFeedback('Connection failed: $error', isError: true);
    } finally {
      _testingClient = null;
      if (mounted) setState(() => _isTesting = false);
    }
  }

  Future<void> _save() async {
    final normalized = _normalizeUrl(allowEmpty: true);
    if (normalized == null) return;
    setState(() => _isSaving = true);
    final error = await widget.onSave(
      AppSettings(backendUrl: normalized, method: _method),
    );
    if (!mounted) return;
    if (error != null) {
      setState(() => _isSaving = false);
      _showFeedback(error, isError: true);
      return;
    }
    Navigator.of(context).pop();
  }

  void _showFeedback(String message, {required bool isError}) {
    if (!mounted) return;
    setState(() {
      _feedback = message;
      _feedbackIsError = isError;
    });
  }

  @override
  Widget build(BuildContext context) {
    final busy = _isTesting || _isSaving;
    return Padding(
      padding: EdgeInsets.fromLTRB(
        20,
        20,
        20,
        20 + MediaQuery.viewInsetsOf(context).bottom,
      ),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Text('Settings', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 16),
            TextField(
              key: const ValueKey<String>('backend-url-field'),
              controller: _backendController,
              enabled: !busy,
              decoration: const InputDecoration(
                labelText: 'Backend server URL',
                hintText: 'http://192.168.1.20:8000',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.url,
              textInputAction: TextInputAction.done,
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              key: const ValueKey<String>('method-dropdown'),
              initialValue: _method,
              decoration: const InputDecoration(
                labelText: 'Prediction method',
                border: OutlineInputBorder(),
              ),
              items: const <DropdownMenuItem<String>>[
                DropdownMenuItem(value: 'rule', child: Text('Rule-based')),
                DropdownMenuItem(value: 'ml', child: Text('ML model')),
                DropdownMenuItem(value: 'yamnet', child: Text('YAMNet')),
              ],
              onChanged: busy
                  ? null
                  : (value) {
                      if (value != null) setState(() => _method = value);
                    },
            ),
            if (_feedback != null) ...<Widget>[
              const SizedBox(height: 12),
              Text(
                _feedback!,
                key: const ValueKey<String>('settings-feedback'),
                style: TextStyle(
                  color: _feedbackIsError
                      ? Theme.of(context).colorScheme.error
                      : Colors.green.shade700,
                ),
              ),
            ],
            const SizedBox(height: 16),
            Row(
              children: <Widget>[
                Expanded(
                  child: OutlinedButton.icon(
                    key: const ValueKey<String>('test-connection-button'),
                    onPressed: busy ? null : _testConnection,
                    icon: const Icon(Icons.wifi_find),
                    label: Text(_isTesting ? 'Testing...' : 'Test Connection'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    key: const ValueKey<String>('save-settings-button'),
                    onPressed: busy ? null : _save,
                    icon: const Icon(Icons.save),
                    label: Text(_isSaving ? 'Saving...' : 'Save'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
