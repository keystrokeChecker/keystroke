import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'audio_services.dart';
import 'backend_client.dart';
import 'local_store.dart';
import 'recording_result.dart';
import 'recording_storage.dart';

typedef BackendClientBuilder = BackendClient Function(String baseUrl);
typedef Clock = DateTime Function();

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
  String _reconstructedText = '';
  List<KeystrokeEvent> _lastEvents = const <KeystrokeEvent>[];
  ModelInfo? _modelInfo;
  String _backendUrl = '';

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

  Timer? _recordingTimer;
  Duration _elapsedRecording = Duration.zero;

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

    _playbackSubscription = _playback.statusStream.listen((status) {
      if (!mounted) return;
      setState(() {
        _playbackStatus = status;
        if (status == PlaybackStatus.stopped) {
          _isPlaybackLoading = false;
        }
      });
    });

    _initialize();
  }

  Future<void> _initialize() async {
    final store = widget.localStore ?? await LocalStore.create();
    _localStore = store;

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
      if (await _recordingStorage.exists(recording.filePath)) {
        availableHistory.add(recording);
      }
    }
    availableHistory.sort(_newestFirst);

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
    if (backendUrl != settings.backendUrl) {
      try {
        await store.saveSettings(
          AppSettings(backendUrl: backendUrl),
        );
      } on Object {
        // Sanitized settings still remain safe for this session.
      }
    }

    final hasInterrupted = availableHistory.isNotEmpty &&
        availableHistory.first.status == AnalysisStatus.failed &&
        availableHistory.first.errorMessage == 'Recording was interrupted before analysis.';

    if (!mounted) return;
    setState(() {
      _backendUrl = backendUrl;
      _isInitializing = false;
      _statusMessage = hasInterrupted
          ? 'Recording was interrupted before analysis. Tap Retry.'
          : _backendUrl.isEmpty
              ? 'Configure the backend URL in Settings to begin.'
              : 'Ready to record.';
    });

    if (_backendUrl.isNotEmpty) {
      _autoFetchModelInfo();
    }
  }

  Future<void> _autoFetchModelInfo() async {
    try {
      final client = _backendClientBuilder(_backendUrl);
      final info = await client.fetchModelInfo();
      if (mounted) {
        setState(() => _modelInfo = info);
      }
    } catch (_) {
      // Best-effort metadata query
    }
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
    _recordingTimer?.cancel();
    _recordingTimer = null;
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
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive ||
        state == AppLifecycleState.detached) {
      _pauseOrStopPlayback();
      if (_isRecording) {
        unawaited(_handleAppInterruption());
      }
    }
  }

  Future<void> _handleAppInterruption() async {
    final id = _currentRecordingId;
    final timestamp = _recordingStartedAt;
    final path = _currentRecordingPath;

    _finishStopping();
    try {
      await _recorder.stop();
    } on Object {
      // Best-effort cleanup
    }

    if (id != null && timestamp != null && path != null) {
      final file = File(path);
      if (await _recordingStorage.hasAudio(file)) {
        await _storeInterruptedRecording(
          id: id,
          timestamp: timestamp,
          file: file,
        );
      } else {
        await _recordingStorage.delete(path);
      }
    }
    _currentRecordingId = null;
    _recordingStartedAt = null;
    _currentRecordingPath = null;
    _setStatus(
      'Recording was interrupted before analysis. Tap Retry.',
    );
  }

  Future<void> _pauseOrStopPlayback() async {
    if (_playbackStatus == PlaybackStatus.playing) {
      try {
        await _playback.pause();
      } on Object {
        // Best-effort playback control
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _recordingTimer?.cancel();
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
    if (unfinishedPath != null) await _recordingStorage.delete(unfinishedPath);
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
    if (_backendUrl.isEmpty) {
      _setStatus('Configure the backend URL in Settings before recording.');
      return;
    }
    if (_isBusy || _isRecording) return;
    setState(() => _isStarting = true);

    bool hasPermission = false;
    try {
      hasPermission = await _recorder.hasPermission();
    } on Object catch (error) {
      _finishStopping();
      _setStatus(
        'Could not check microphone permission: ${_friendlyError(error)}',
      );
      return;
    }

    if (!hasPermission) {
      _finishStopping();
      _setStatus('Microphone permission is required to record.');
      return;
    }

    final timestamp = _clock();
    _recordingSequence += 1;
    File targetFile;
    try {
      targetFile = await _recordingStorage.allocate(timestamp, _recordingSequence);
    } on Object catch (error) {
      _finishStopping();
      _setStatus('Could not allocate storage: ${_friendlyError(error)}');
      return;
    }
    final targetPath = targetFile.path;
    final recordingId =
        '${timestamp.microsecondsSinceEpoch}_${_recordingSequence.toString().padLeft(3, '0')}';

    try {
      await _recorder.start(targetPath);
      _currentRecordingPath = targetPath;
      _currentRecordingId = recordingId;
      _recordingStartedAt = timestamp;
      _elapsedRecording = Duration.zero;
      _recordingTimer?.cancel();
      _recordingTimer = Timer.periodic(const Duration(milliseconds: 100), (_) {
        if (mounted && _isRecording) {
          setState(() {
            _elapsedRecording += const Duration(milliseconds: 100);
          });
        }
      });

      if (!mounted) return;
      setState(() {
        _isStarting = false;
        _isRecording = true;
        _statusMessage = 'Recording keystrokes... Tap Stop when done.';
      });
    } on Object catch (error) {
      _finishStopping();
      await _recordingStorage.delete(targetPath);
      _currentRecordingPath = null;
      _currentRecordingId = null;
      _recordingStartedAt = null;
      _setStatus('Could not start recording: ${_friendlyError(error)}');
    }
  }

  Future<void> _stopRecording({required bool analyze}) async {
    if (!_isRecording || _isStopping) return;
    setState(() => _isStopping = true);
    _recordingTimer?.cancel();
    _recordingTimer = null;

    final id = _currentRecordingId;
    final timestamp = _recordingStartedAt;
    final path = _currentRecordingPath;

    _currentRecordingId = null;
    _recordingStartedAt = null;
    _currentRecordingPath = null;

    try {
      await _recorder.stop();
    } on Object catch (error) {
      _finishStopping();
      if (path != null) await _recordingStorage.delete(path);
      _setStatus('Could not stop recording cleanly: ${_friendlyError(error)}');
      return;
    }

    _finishStopping();
    if (id == null || timestamp == null || path == null) return;

    final file = File(path);
    if (!await _recordingStorage.hasAudio(file)) {
      await _recordingStorage.delete(path);
      _setStatus(
        'Recording was empty or could not be saved. Nothing to analyze.',
      );
      return;
    }

    if (analyze && _appIsActive) {
      await _analyzeRecording(
        id: id,
        timestamp: timestamp,
        file: file,
      );
    } else {
      await _storeInterruptedRecording(
        id: id,
        timestamp: timestamp,
        file: file,
      );
    }
  }

  Future<void> _analyzeRecording({
    required String id,
    required DateTime timestamp,
    String method = 'default',
    required File file,
  }) async {
    if (_backendUrl.isEmpty) {
      await _storeFailedRecording(
        id: id,
        timestamp: timestamp,
        method: method,
        file: file,
        message: 'Backend server URL is not configured.',
      );
      _setStatus(
        'Backend URL is missing. The recording was saved and can be retried.',
      );
      return;
    }

    final client = _backendClientBuilder(_backendUrl);
    _activeBackendClient = client;
    if (mounted) {
      setState(() {
        _isAnalyzing = true;
        _isCancelling = false;
        _statusMessage = 'Analyzing keystroke acoustics...';
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
        reconstructedSequence: response.reconstructedSequence,
        detectedEventsCount: response.detectedEventsCount,
        events: response.events,
      );
      finalStatus = 'Analysis complete.';
      _resultText = response.formatted;
      _reconstructedText = response.reconstructedSequence;
      _lastEvents = response.events;
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
    if (!saved) {
      finalStatus = '$finalStatus (History could not be saved to disk).';
    }
    _setStatus(finalStatus);
  }

  Future<void> _storeInterruptedRecording({
    required String id,
    required DateTime timestamp,
    String method = 'ml',
    required File file,
  }) async {
    final result = RecordingResult(
      id: id,
      timestamp: timestamp,
      formatted: 'Analysis pending',
      counts: const <int>[],
      method: method,
      filePath: file.path,
      status: AnalysisStatus.failed,
      errorMessage: 'Recording was interrupted before analysis.',
    );
    await _upsertHistory(result);
  }

  Future<void> _storeFailedRecording({
    required String id,
    required DateTime timestamp,
    String method = 'ml',
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
    await _upsertHistory(result);
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
      if (!await _recordingStorage.hasAudio(file)) {
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

      if (mounted) {
        setState(() => _isRetrying = false);
      } else {
        _isRetrying = false;
      }
      await _analyzeRecording(
        id: item.id,
        timestamp: item.timestamp,
        method: item.method,
        file: file,
      );
    } finally {
      if (mounted && _isRetrying) {
        setState(() => _isRetrying = false);
      }
    }
  }

  Future<bool> _upsertHistory(RecordingResult entry) async {
    final index = _history.indexWhere((element) => element.id == entry.id);
    if (index >= 0) {
      _history[index] = entry;
    } else {
      _history.insert(0, entry);
    }
    _history.sort(_newestFirst);
    if (mounted) setState(() {});
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
      backgroundColor: const Color(0xFF151829),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) => _SettingsSheet(
        initialSettings: AppSettings(
          backendUrl: _backendUrl,
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
      _statusMessage = settings.backendUrl.isEmpty
          ? 'Backend configuration cleared.'
          : 'Settings saved. Ready to record.';
    });
    if (_backendUrl.isNotEmpty) {
      _autoFetchModelInfo();
    }
    return null;
  }

  Future<void> _togglePlay(RecordingResult item) async {
    if (_loadedRecordingId == item.id &&
        _playbackStatus == PlaybackStatus.playing) {
      try {
        await _playback.pause();
      } on Object catch (error) {
        _setStatus('Playback error: ${_friendlyError(error)}');
      }
      return;
    }

    if (_loadedRecordingId == item.id &&
        _playbackStatus == PlaybackStatus.paused) {
      try {
        await _playback.play();
      } on Object catch (error) {
        _setStatus('Playback error: ${_friendlyError(error)}');
      }
      return;
    }

    setState(() {
      _isPlaybackLoading = true;
      _loadedRecordingId = item.id;
    });

    final file = File(item.filePath);
    if (!await _recordingStorage.hasAudio(file)) {
      setState(() => _isPlaybackLoading = false);
      _setStatus('Audio file is missing or unreadable.');
      return;
    }

    try {
      await _playback.load(item.filePath);
      await _playback.play();
      if (mounted) setState(() => _isPlaybackLoading = false);
    } on Object catch (error) {
      if (mounted) {
        setState(() {
          _isPlaybackLoading = false;
          _loadedRecordingId = null;
        });
      }
      _setStatus('Could not start playback: ${_friendlyError(error)}');
    }
  }

  Future<void> _dismissRecording(RecordingResult item) async {
    final index = _history.indexWhere((element) => element.id == item.id);
    if (index < 0) return;
    final removed = _history.removeAt(index);
    if (_loadedRecordingId == removed.id) {
      unawaited(_playback.stop());
      _loadedRecordingId = null;
    }
    setState(() {});
    await _recordingStorage.delete(removed.filePath);
    final saved = await _persistHistory();
    if (!saved) {
      _setStatus('Recording deleted, but history could not be saved to disk.');
    }
  }

  void _setStatus(String message) {
    if (!mounted) return;
    setState(() => _statusMessage = message);
  }

  int _newestFirst(RecordingResult a, RecordingResult b) =>
      b.timestamp.compareTo(a.timestamp);

  String _friendlyError(Object error) {
    final raw = error.toString().trim();
    if (raw.isEmpty) return 'An unknown error occurred.';
    return raw.replaceFirst(RegExp(r'^[A-Za-z0-9_]+Exception:\s*'), '');
  }

  String _formatTimestamp(DateTime timestamp) {
    final local = timestamp.toLocal();
    final year = local.year.toString().padLeft(4, '0');
    final month = local.month.toString().padLeft(2, '0');
    final day = local.day.toString().padLeft(2, '0');
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    final second = local.second.toString().padLeft(2, '0');
    return '$year-$month-$day $hour:$minute:$second';
  }

  String _formatDuration(Duration d) {
    final minutes = d.inMinutes.toString().padLeft(2, '0');
    final seconds = (d.inSeconds % 60).toString().padLeft(2, '0');
    final tenths = ((d.inMilliseconds % 1000) ~/ 100).toString();
    return '$minutes:$seconds.$tenths';
  }

  void _showEventDetailsModal(KeystrokeEvent ev) {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: const Color(0xFF191D32),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: <Widget>[
                  Text(
                    'Key Event #${ev.eventIndex}: "${ev.prediction == 'SPACE' ? '␣ SPACE' : ev.prediction}"',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: ev.status == 'CONFIDENT'
                          ? Colors.green.withOpacity(0.2)
                          : Colors.amber.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(
                        color: ev.status == 'CONFIDENT'
                            ? Colors.greenAccent
                            : Colors.amberAccent,
                      ),
                    ),
                    child: Text(
                      '${(ev.confidence * 100).toStringAsFixed(1)}% ${ev.status}',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: ev.status == 'CONFIDENT'
                            ? Colors.greenAccent
                            : Colors.amberAccent,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                'Timestamp: ${ev.timestampS.toStringAsFixed(3)}s in audio stream',
                style: const TextStyle(color: Colors.grey, fontSize: 13),
              ),
              const SizedBox(height: 16),
              const Text(
                'Top Prediction Probabilities:',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                  color: Color(0xFFA78BFA),
                ),
              ),
              const SizedBox(height: 8),
              ...ev.topK.map((item) {
                final pct = (item.probability * 100).clamp(0.0, 100.0);
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: <Widget>[
                          Text(
                            item.key == 'SPACE' ? 'SPACE' : item.key,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                            ),
                          ),
                          Text('${pct.toStringAsFixed(1)}%'),
                        ],
                      ),
                      const SizedBox(height: 4),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: pct / 100.0,
                          minHeight: 6,
                          backgroundColor: const Color(0xFF262B46),
                          valueColor: AlwaysStoppedAnimation<Color>(
                            item.key == ev.prediction
                                ? const Color(0xFF06B6D4)
                                : const Color(0xFF8B5CF6),
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              }),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final controlsDisabled = _isBusy || _isInitializing;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF0C0E17),
        elevation: 0,
        title: Row(
          children: <Widget>[
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF8B5CF6), Color(0xFF06B6D4)],
                ),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.graphic_eq, color: Colors.white, size: 20),
            ),
            const SizedBox(width: 10),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                const Text(
                  'KEYSTROKE AI',
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 16,
                    letterSpacing: 1.2,
                    color: Colors.white,
                  ),
                ),
                Text(
                  _modelInfo != null
                      ? '${_modelInfo!.modelName} • ${(_modelInfo!.accuracy * 100).toStringAsFixed(1)}% Acc'
                      : 'Acoustic Forensics Engine v2',
                  style: const TextStyle(
                    fontSize: 11,
                    color: Color(0xFF06B6D4),
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ],
        ),
        actions: <Widget>[
          IconButton(
            icon: const Icon(Icons.settings_outlined, color: Color(0xFFA78BFA)),
            tooltip: 'Settings',
            onPressed: controlsDisabled || _isRecording ? null : _openSettings,
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              // Top server connection status bar
              if (_backendUrl.isEmpty && !_isInitializing)
                Card(
                  color: const Color(0xFF2D151F),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                    side: const BorderSide(color: Color(0xFFE11D48)),
                  ),
                  child: ListTile(
                    leading: const Icon(Icons.warning_amber_rounded, color: Color(0xFFF43F5E)),
                    title: const Text(
                      'Backend setup required',
                      style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
                    ),
                    subtitle: const Text(
                      'Open Settings and enter the computer address running the API.',
                      style: TextStyle(color: Color(0xFFFDA4AF), fontSize: 12),
                    ),
                    trailing: TextButton.icon(
                      onPressed: _openSettings,
                      icon: const Icon(Icons.settings, size: 16, color: Colors.white),
                      label: const Text(
                        'SET UP',
                        style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
                      ),
                      style: TextButton.styleFrom(
                        backgroundColor: const Color(0xFFE11D48),
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      ),
                    ),
                  ),
                )
              else if (_backendUrl.isNotEmpty) ...<Widget>[
                InkWell(
                  onTap: controlsDisabled || _isRecording ? null : _openSettings,
                  borderRadius: BorderRadius.circular(10),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: const Color(0xFF16192B),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: const Color(0xFF262B46)),
                    ),
                    child: Row(
                      children: <Widget>[
                        Container(
                          width: 8,
                          height: 8,
                          decoration: const BoxDecoration(
                            shape: BoxShape.circle,
                            color: Color(0xFF10B981),
                            boxShadow: [
                              BoxShadow(color: Color(0xFF10B981), blurRadius: 4),
                            ],
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Connected: $_backendUrl',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: Color(0xFFE2E8F0),
                            ),
                          ),
                        ),
                        if (_modelInfo != null)
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                            decoration: BoxDecoration(
                              color: const Color(0xFF8B5CF6).withOpacity(0.2),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              '${_modelInfo!.modelName} (${(_modelInfo!.accuracy * 100).toStringAsFixed(1)}%)',
                              style: const TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFFA78BFA),
                              ),
                            ),
                          ),
                        const SizedBox(width: 6),
                        const Icon(Icons.tune, size: 16, color: Color(0xFF64748B)),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 8),
              ],

              // Reconstructed text & detected keystrokes card
              const Text(
                'Keystrokes Detected',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFF94A3B8),
                  letterSpacing: 1.1,
                ),
              ),
              const SizedBox(height: 4),

              Card(
                color: const Color(0xFF131627),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                  side: const BorderSide(color: Color(0xFF262B46)),
                ),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 14),
                  child: Column(
                    children: <Widget>[
                      if (_reconstructedText.isNotEmpty) ...[
                        Row(
                          children: <Widget>[
                            const Icon(Icons.terminal, size: 16, color: Color(0xFF06B6D4)),
                            const SizedBox(width: 6),
                            const Text(
                              'DECODED SEQUENCE',
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.bold,
                                letterSpacing: 1.2,
                                color: Color(0xFF06B6D4),
                              ),
                            ),
                            const Spacer(),
                            IconButton(
                              icon: const Icon(Icons.copy, size: 16, color: Color(0xFFA78BFA)),
                              tooltip: 'Copy to clipboard',
                              onPressed: () {
                                Clipboard.setData(ClipboardData(text: _reconstructedText));
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                    content: Text('Decoded text copied to clipboard!'),
                                    duration: Duration(seconds: 2),
                                  ),
                                );
                              },
                            ),
                          ],
                        ),
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
                          decoration: BoxDecoration(
                            color: const Color(0xFF0A0C16),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: const Color(0xFF8B5CF6).withOpacity(0.4)),
                          ),
                          child: SelectableText(
                            _reconstructedText,
                            style: const TextStyle(
                              fontFamily: 'monospace',
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              letterSpacing: 2.0,
                              color: Color(0xFF38BDF8),
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                      ],
                      Text(
                        _resultText.isEmpty ? '—' : _resultText,
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: _reconstructedText.isNotEmpty ? 14 : 22,
                          fontWeight: FontWeight.bold,
                          color: _resultText.isEmpty
                              ? const Color(0xFF64748B)
                              : const Color(0xFFA78BFA),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              // Interactive Keystroke Timeline
              if (_lastEvents.isNotEmpty) ...<Widget>[
                const SizedBox(height: 8),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: <Widget>[
                    const Row(
                      children: <Widget>[
                        Icon(Icons.timeline, size: 14, color: Color(0xFF06B6D4)),
                        SizedBox(width: 6),
                        Text(
                          'TIMELINE STREAM',
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.1,
                            color: Color(0xFF06B6D4),
                          ),
                        ),
                      ],
                    ),
                    Text(
                      '${_lastEvents.length} detected keystrokes',
                      style: const TextStyle(fontSize: 11, color: Color(0xFF94A3B8)),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                SizedBox(
                  height: 48,
                  child: ListView.separated(
                    scrollDirection: Axis.horizontal,
                    itemCount: _lastEvents.length,
                    separatorBuilder: (_, __) => const SizedBox(width: 6),
                    itemBuilder: (context, idx) {
                      final ev = _lastEvents[idx];
                      final isConfident = ev.status == 'CONFIDENT';
                      final isLow = ev.status == 'LOW_CONFIDENCE';
                      final borderColor = isConfident
                          ? const Color(0xFF10B981)
                          : isLow
                              ? const Color(0xFFF59E0B)
                              : const Color(0xFF64748B);
                      final bgColor = isConfident
                          ? const Color(0xFF064E3B).withOpacity(0.4)
                          : isLow
                              ? const Color(0xFF78350F).withOpacity(0.4)
                              : const Color(0xFF1E293B).withOpacity(0.4);

                      final keyLabel = ev.prediction == 'SPACE' ? '␣' : ev.prediction;
                      final timeLabel = '${ev.timestampS.toStringAsFixed(2)}s';
                      final pctLabel = '${(ev.confidence * 100).toInt()}%';

                      return InkWell(
                        onTap: () => _showEventDetailsModal(ev),
                        borderRadius: BorderRadius.circular(8),
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: bgColor,
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: borderColor, width: 1.2),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: <Widget>[
                              Text(
                                keyLabel,
                                style: const TextStyle(
                                  fontWeight: FontWeight.w900,
                                  fontSize: 16,
                                  color: Colors.white,
                                ),
                              ),
                              const SizedBox(width: 6),
                              Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: <Widget>[
                                  Text(
                                    timeLabel,
                                    style: const TextStyle(fontSize: 9, color: Color(0xFF94A3B8)),
                                  ),
                                  Text(
                                    pctLabel,
                                    style: TextStyle(
                                      fontSize: 9,
                                      fontWeight: FontWeight.bold,
                                      color: borderColor,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ],

              const SizedBox(height: 8),

              // Recording Hub / Action Button
              Container(
                padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
                decoration: BoxDecoration(
                  color: const Color(0xFF141726),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: _isRecording
                        ? const Color(0xFFF43F5E)
                        : const Color(0xFF262B46),
                    width: _isRecording ? 1.5 : 1.0,
                  ),
                  boxShadow: _isRecording
                      ? [
                          BoxShadow(
                            color: const Color(0xFFF43F5E).withOpacity(0.2),
                            blurRadius: 16,
                            spreadRadius: 2,
                          ),
                        ]
                      : null,
                ),
                child: Column(
                  children: <Widget>[
                    if (_isRecording) ...[
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: <Widget>[
                          Container(
                            width: 10,
                            height: 10,
                            decoration: const BoxDecoration(
                              shape: BoxShape.circle,
                              color: Color(0xFFF43F5E),
                              boxShadow: [
                                BoxShadow(color: Color(0xFFF43F5E), blurRadius: 6),
                              ],
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            'LIVE CAPTURE: ${_formatDuration(_elapsedRecording)}',
                            style: const TextStyle(
                              fontFamily: 'monospace',
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                              color: Color(0xFFF43F5E),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                    ],
                    SizedBox(
                      width: double.infinity,
                      height: 52,
                      child: ElevatedButton.icon(
                        key: const ValueKey<String>('record-button'),
                        icon: Icon(
                          _isRecording ? Icons.stop_circle : Icons.mic_none_outlined,
                          size: 24,
                        ),
                        label: Text(
                          _isRecording ? 'Stop Recording' : 'Start Recording',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.5,
                          ),
                        ),
                        onPressed: controlsDisabled ? null : _toggleRecording,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: _isRecording
                              ? const Color(0xFFE11D48)
                              : const Color(0xFF8B5CF6),
                          foregroundColor: Colors.white,
                          elevation: _isRecording ? 6 : 4,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ),
                    if (_isAnalyzing) ...<Widget>[
                      const SizedBox(height: 8),
                      OutlinedButton.icon(
                        key: const ValueKey<String>('cancel-analysis-button'),
                        onPressed: _isCancelling ? null : _cancelAnalysis,
                        icon: const Icon(Icons.cancel_outlined, color: Color(0xFFF43F5E)),
                        label: Text(
                          _isCancelling ? 'Cancelling...' : 'Cancel Analysis',
                          style: const TextStyle(color: Color(0xFFF43F5E)),
                        ),
                        style: OutlinedButton.styleFrom(
                          side: const BorderSide(color: Color(0xFFF43F5E)),
                        ),
                      ),
                    ],
                    if (_isInitializing || _isStopping || _isAnalyzing) ...<Widget>[
                      const SizedBox(height: 8),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: const LinearProgressIndicator(
                          minHeight: 4,
                          backgroundColor: Color(0xFF262B46),
                          valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF06B6D4)),
                        ),
                      ),
                    ],
                  ],
                ),
              ),

              const SizedBox(height: 6),
              Semantics(
                liveRegion: true,
                child: Text(
                  _statusMessage,
                  key: const ValueKey<String>('status-message'),
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: _statusMessage.contains('failed') ||
                            _statusMessage.contains('error') ||
                            _statusMessage.contains('denied')
                        ? const Color(0xFFF43F5E)
                        : const Color(0xFF94A3B8),
                  ),
                ),
              ),

              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: <Widget>[
                  const Text(
                    'History',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 13,
                      color: Color(0xFFA78BFA),
                      letterSpacing: 1.0,
                    ),
                  ),
                  Text(
                    '${_history.length} records',
                    style: const TextStyle(fontSize: 11, color: Color(0xFF64748B)),
                  ),
                ],
              ),
              const SizedBox(height: 4),

              // History list view
              Expanded(
                child: _history.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: const <Widget>[
                            Icon(Icons.history, size: 36, color: Color(0xFF262B46)),
                            SizedBox(height: 8),
                            Text(
                              'No saved recordings yet.',
                              style: TextStyle(color: Color(0xFF64748B), fontSize: 13),
                            ),
                          ],
                        ),
                      )
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
                              margin: const EdgeInsets.symmetric(vertical: 4),
                              decoration: BoxDecoration(
                                color: const Color(0xFFE11D48),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              alignment: Alignment.centerRight,
                              padding: const EdgeInsets.symmetric(horizontal: 20),
                              child: const Icon(
                                Icons.delete_outline,
                                color: Colors.white,
                              ),
                            ),
                            onDismissed: (_) => _dismissRecording(item),
                            child: Card(
                              margin: const EdgeInsets.symmetric(vertical: 4),
                              color: const Color(0xFF16192B),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                                side: const BorderSide(color: Color(0xFF262B46)),
                              ),
                              child: Padding(
                                padding: const EdgeInsets.all(12),
                                child: Row(
                                  children: <Widget>[
                                    Icon(
                                      completed
                                          ? Icons.check_circle
                                          : item.status ==
                                                  AnalysisStatus.cancelled
                                              ? Icons.cancel_outlined
                                              : Icons.error_outline,
                                      color: completed
                                          ? const Color(0xFF10B981)
                                          : const Color(0xFFF43F5E),
                                      size: 20,
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
                                              fontSize: 15,
                                              fontWeight: FontWeight.w700,
                                              color: Colors.white,
                                            ),
                                          ),
                                          if (item.reconstructedSequence.isNotEmpty) ...[
                                            const SizedBox(height: 3),
                                            Container(
                                              padding: const EdgeInsets.symmetric(
                                                horizontal: 6,
                                                vertical: 2,
                                              ),
                                              decoration: BoxDecoration(
                                                color: const Color(0xFF8B5CF6).withOpacity(0.15),
                                                borderRadius: BorderRadius.circular(4),
                                                border: Border.all(
                                                  color: const Color(0xFF8B5CF6).withOpacity(0.3),
                                                  width: 0.8,
                                                ),
                                              ),
                                              child: Text(
                                                'Typed: "${item.reconstructedSequence}" (${item.detectedEventsCount} keys)',
                                                style: const TextStyle(
                                                  fontSize: 11,
                                                  fontWeight: FontWeight.w600,
                                                  color: Color(0xFF38BDF8),
                                                  fontFamily: 'monospace',
                                                ),
                                              ),
                                            ),
                                          ],
                                          const SizedBox(height: 2),
                                          Text(
                                            completed
                                                ? (_modelInfo != null
                                                    ? '${_modelInfo!.modelName} Model • ${item.detectedEventsCount > 0 ? '${item.detectedEventsCount} keys' : 'Analyzed'}'
                                                    : 'AI Keystroke Model • ${item.detectedEventsCount > 0 ? '${item.detectedEventsCount} keys' : 'Analyzed'}')
                                                : 'Analysis Failed',
                                            style: const TextStyle(
                                              fontSize: 11,
                                              color: Color(0xFF06B6D4),
                                            ),
                                          ),
                                          if (item.errorMessage != null) ...[
                                            const SizedBox(height: 2),
                                            Text(
                                              item.errorMessage!,
                                              maxLines: 2,
                                              overflow: TextOverflow.ellipsis,
                                              style: const TextStyle(
                                                fontSize: 11,
                                                color: Color(0xFFF43F5E),
                                              ),
                                            ),
                                          ],
                                          Text(
                                            _formatTimestamp(item.timestamp),
                                            style: const TextStyle(
                                              fontSize: 10,
                                              color: Color(0xFF64748B),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    if (!completed)
                                      IconButton(
                                        key: ValueKey<String>('retry-${item.id}'),
                                        tooltip: 'Retry analysis',
                                        icon: const Icon(Icons.refresh, color: Color(0xFFF59E0B)),
                                        onPressed: controlsDisabled || _isRecording
                                            ? null
                                            : () => _retryRecording(item),
                                      )
                                    else
                                      IconButton(
                                        key: ValueKey<String>('play-${item.id}'),
                                        tooltip: isPlaying ? 'Pause recording' : 'Play recording',
                                        icon: Icon(
                                          isPlaying
                                              ? Icons.pause_circle_filled
                                              : Icons.play_circle_filled,
                                          color: const Color(0xFF8B5CF6),
                                          size: 28,
                                        ),
                                        onPressed: controlsDisabled || _isRecording
                                            ? null
                                            : () => _togglePlay(item),
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
  bool _isTesting = false;
  bool _isSaving = false;
  String? _feedback;
  bool _feedbackIsError = false;
  BackendClient? _testingClient;
  ModelInfo? _modelInfo;

  @override
  void initState() {
    super.initState();
    _backendController = TextEditingController(
      text: widget.initialSettings.backendUrl,
    );
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
      _modelInfo = null;
    });
    try {
      final client = widget.backendClientBuilder(normalized);
      _testingClient = client;
      await client.testConnection();
      try {
        final info = await client.fetchModelInfo();
        if (mounted) setState(() => _modelInfo = info);
      } catch (_) {
        // Model info endpoint is optional
      }
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
      AppSettings(backendUrl: normalized),
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
            Row(
              children: <Widget>[
                const Icon(Icons.settings, color: Color(0xFF8B5CF6)),
                const SizedBox(width: 10),
                Text(
                  'Settings',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            TextField(
              key: const ValueKey<String>('backend-url-field'),
              controller: _backendController,
              enabled: !busy,
              style: const TextStyle(color: Colors.white),
              decoration: const InputDecoration(
                labelText: 'Backend server URL',
                hintText: 'http://192.168.1.20:8000',
                border: OutlineInputBorder(),
                filled: true,
                fillColor: Color(0xFF1E2238),
              ),
              keyboardType: TextInputType.url,
              textInputAction: TextInputAction.done,
            ),
            if (_feedback != null) ...<Widget>[
              const SizedBox(height: 12),
              Text(
                _feedback!,
                key: const ValueKey<String>('settings-feedback'),
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: _feedbackIsError
                      ? const Color(0xFFF43F5E)
                      : const Color(0xFF10B981),
                ),
              ),
            ],
            if (_modelInfo != null) ...<Widget>[
              const SizedBox(height: 10),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFF064E3B).withOpacity(0.3),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: const Color(0xFF10B981)),
                ),
                child: Row(
                  children: <Widget>[
                    const Icon(Icons.verified, size: 20, color: Color(0xFF10B981)),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: <Widget>[
                          Text(
                            'Active Model: ${_modelInfo!.modelName}',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                              fontSize: 13,
                            ),
                          ),
                          Text(
                            'Accuracy: ${(_modelInfo!.accuracy * 100).toStringAsFixed(1)}% • Features: ${_modelInfo!.featureSet} • Classes: ${_modelInfo!.classesCount}',
                            style: const TextStyle(
                              color: Color(0xFF34D399),
                              fontSize: 11,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
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
                    icon: const Icon(Icons.wifi_find, color: Color(0xFF06B6D4)),
                    label: Text(
                      _isTesting ? 'Testing...' : 'Test Connection',
                      style: const TextStyle(color: Color(0xFF06B6D4)),
                    ),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: Color(0xFF06B6D4)),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    key: const ValueKey<String>('save-settings-button'),
                    onPressed: busy ? null : _save,
                    icon: const Icon(Icons.save),
                    label: Text(_isSaving ? 'Saving...' : 'Save'),
                    style: FilledButton.styleFrom(
                      backgroundColor: const Color(0xFF8B5CF6),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
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
