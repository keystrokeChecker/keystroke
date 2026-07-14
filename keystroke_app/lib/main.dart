import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:just_audio/just_audio.dart';

import 'recording_result.dart';
import 'recordings_page.dart';

void main() {
  runApp(const KeystrokeApp());
}

// Samsung Voice Recorder palette
const _samsungBlue = Color(0xFF0188E2);
const _samsungRed = Color(0xFFFA4549);
const _samsungTeal = Color(0xFF00C1D4);

class KeystrokeApp extends StatefulWidget {
  const KeystrokeApp({super.key});

  @override
  State<KeystrokeApp> createState() => _KeystrokeAppState();
}

class _KeystrokeAppState extends State<KeystrokeApp> {
  bool _isDark = false;

  final ThemeData _lightTheme = ThemeData(
    colorScheme: const ColorScheme.light(
      primary: _samsungBlue,
      secondary: _samsungTeal,
      surface: Colors.white,
    ),
    useMaterial3: true,
    scaffoldBackgroundColor: const Color(0xFFF2F2F2),
    appBarTheme: const AppBarTheme(
      backgroundColor: Color(0xFFF2F2F2),
      foregroundColor: Colors.black87,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: Colors.white,
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
  );

  final ThemeData _darkTheme = ThemeData(
    colorScheme: const ColorScheme.dark(
      primary: _samsungBlue,
      secondary: _samsungTeal,
      surface: Color(0xFF1E1E1E),
    ),
    useMaterial3: true,
    scaffoldBackgroundColor: const Color(0xFF000000),
    appBarTheme: const AppBarTheme(
      backgroundColor: Color(0xFF000000),
      foregroundColor: Colors.white,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: const Color(0xFF1E1E1E),
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
  );

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Keystroke Analyzer',
      theme: _lightTheme,
      darkTheme: _darkTheme,
      themeMode: _isDark ? ThemeMode.dark : ThemeMode.light,
      home: KeystrokeHomePage(
        onThemeChanged: (isDark) => setState(() => _isDark = isDark),
        isDark: _isDark,
      ),
    );
  }
}

class KeystrokeHomePage extends StatefulWidget {
  const KeystrokeHomePage({
    super.key,
    required this.onThemeChanged,
    required this.isDark,
  });

  final ValueChanged<bool> onThemeChanged;
  final bool isDark;

  @override
  State<KeystrokeHomePage> createState() => _KeystrokeHomePageState();
}

class _KeystrokeHomePageState extends State<KeystrokeHomePage>
    with TickerProviderStateMixin {
  final AudioRecorder _recorder = AudioRecorder();
  final TextEditingController _backendController = TextEditingController();
  final List<RecordingResult> _history = [];
  final AudioPlayer _player = AudioPlayer();

  bool _isRecording = false;
  bool _isStopping = false;
  bool _isUploading = false;
  String _statusMessage = 'Tap the red button to record';
  String _resultText = '';
  String _selectedMethod = 'rule';
  int _selectedTabIndex = 0;

  Duration _recordDuration = Duration.zero;
  Timer? _recordTimer;
  late AnimationController _waveController;

  @override
  void initState() {
    super.initState();
    _backendController.text = 'http://<YOUR_MACHINE_IP>:8000';
    _waveController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
    _loadHistory();
  }

  @override
  void dispose() {
    _recordTimer?.cancel();
    _waveController.dispose();
    _backendController.dispose();
    _recorder.dispose();
    _player.dispose();
    super.dispose();
  }

  String _formatDuration(Duration d) {
    final h = d.inHours;
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    if (h > 0) return '$h:$m:$s';
    return '$m:$s';
  }

  void _startRecordTimer() {
    _recordDuration = Duration.zero;
    _recordTimer?.cancel();
    _recordTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() => _recordDuration += const Duration(seconds: 1));
    });
  }

  void _stopRecordTimer() {
    _recordTimer?.cancel();
    _recordTimer = null;
  }

  Future<File> _buildTempFile() async {
    final docDir = await getApplicationDocumentsDirectory();
    final recDir = Directory('${docDir.path}/recordings');
    if (!await recDir.exists()) await recDir.create(recursive: true);
    final timestamp = DateTime.now();
    final filename =
        'recording_${timestamp.year}${timestamp.month.toString().padLeft(2, '0')}${timestamp.day.toString().padLeft(2, '0')}_'
        '${timestamp.hour.toString().padLeft(2, '0')}${timestamp.minute.toString().padLeft(2, '0')}${timestamp.second.toString().padLeft(2, '0')}.wav';
    final file = File('${recDir.path}/$filename');
    if (await file.exists()) await file.delete();
    return file;
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      await _stopRecording();
      return;
    }
    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      setState(() => _statusMessage = 'Microphone permission is required.');
      return;
    }
    final file = await _buildTempFile();
    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.wav),
      path: file.path,
    );
    _startRecordTimer();
    setState(() {
      _isRecording = true;
      _statusMessage = 'Recording…';
      _resultText = '';
    });
  }

  Future<void> _stopRecording() async {
    if (_isStopping) return;
    _isStopping = true;
    if (!_isRecording) {
      _isStopping = false;
      return;
    }
    try {
      final path = await _recorder.stop();
      _stopRecordTimer();
      setState(() => _isRecording = false);
      if (path == null) {
        setState(() => _statusMessage = 'Recording failed or no audio saved.');
        return;
      }
      final file = File(path);
      if (!await file.exists()) {
        setState(() => _statusMessage = 'Recorded audio file not found.');
        return;
      }
      setState(() {
        _isUploading = true;
        _statusMessage = 'Analyzing keystrokes…';
      });
      await _uploadRecording(file);
    } on Exception catch (e) {
      setState(() => _statusMessage = 'Stop failed: $e');
    } finally {
      _isStopping = false;
      _isUploading = false;
      if (mounted) setState(() => _isRecording = false);
    }
  }

  void _selectTab(int index) {
    if (_selectedTabIndex == index) return;
    setState(() => _selectedTabIndex = index);
  }

  Future<void> _uploadRecording(File file) async {
    final backendUrl = _backendController.text.trim();
    if (backendUrl.isEmpty) {
      setState(() => _statusMessage = 'Set backend URL in settings first.');
      return;
    }
    final uri = Uri.tryParse(backendUrl);
    if (uri == null || (uri.scheme != 'http' && uri.scheme != 'https')) {
      setState(() => _statusMessage = 'Enter a valid backend URL in settings.');
      return;
    }
    final analyzeUri = uri.replace(
      path: uri.path.endsWith('/') ? '${uri.path}analyze' : '${uri.path}/analyze',
    );
    try {
      final request = http.MultipartRequest('POST', analyzeUri);
      request.fields['method'] = _selectedMethod;
      request.files.add(
        await http.MultipartFile.fromPath('file', file.path, filename: 'recording.wav'),
      );
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);
      if (response.statusCode != 200) {
        setState(() => _statusMessage = 'Backend error ${response.statusCode}');
        return;
      }
      final Map<String, dynamic> body = jsonDecode(response.body);
      final List<int> counts = (body['counts'] as List).map((e) => e as int).toList();
      final String formatted = body['formatted'] as String;
      final result = RecordingResult(
        timestamp: DateTime.now(),
        formatted: formatted,
        counts: counts,
        method: _selectedMethod,
        filePath: file.path,
      );
      setState(() {
        _resultText = formatted;
        _statusMessage = 'Analysis complete';
        _history.insert(0, result);
      });
      await _saveHistory();
    } catch (error) {
      setState(() => _statusMessage = 'Upload failed: $error');
    }
  }

  void _openSettingsPanel(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        return Padding(
          padding: EdgeInsets.only(
            left: 20,
            right: 20,
            top: 20,
            bottom: MediaQuery.of(ctx).viewInsets.bottom + 24,
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey.shade400,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                'Settings',
                style: Theme.of(ctx).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
              ),
              const SizedBox(height: 16),
              TextField(
                controller: _backendController,
                decoration: const InputDecoration(
                  labelText: 'Backend server URL',
                  hintText: 'http://192.168.x.y:8000',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.url,
              ),
              const SizedBox(height: 12),
              DropdownButtonFormField<String>(
                initialValue: _selectedMethod,
                decoration: const InputDecoration(
                  labelText: 'Prediction method',
                  border: OutlineInputBorder(),
                ),
                items: const [
                  DropdownMenuItem(value: 'rule', child: Text('Rule-based')),
                  DropdownMenuItem(value: 'ml', child: Text('ML-based')),
                ],
                onChanged: (value) {
                  if (value != null) setState(() => _selectedMethod = value);
                },
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Dark mode'),
                  Switch(
                    value: widget.isDark,
                    activeTrackColor: _samsungBlue.withValues(alpha: 0.45),
                    activeThumbColor: _samsungBlue,
                    onChanged: widget.onThemeChanged,
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _saveHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = _history.map((e) => e.toJson()).toList();
    await prefs.setString('keystroke_history', jsonEncode(jsonList));
  }

  Future<void> _loadHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonString = prefs.getString('keystroke_history');
    if (jsonString == null) return;
    try {
      final List<dynamic> decoded = jsonDecode(jsonString);
      final loaded = decoded
          .map((e) => RecordingResult.fromJson(e as Map<String, dynamic>))
          .toList();
      setState(() => _history.addAll(loaded.reversed));
    } catch (_) {}
  }

  Future<void> _playAudio(String path) async {
    try {
      await _player.setFilePath(path);
      await _player.play();
    } catch (e) {
      setState(() => _statusMessage = 'Playback error: $e');
    }
  }

  Future<void> _deleteEntry(int index) async {
    final item = _history[index];
    final file = File(item.filePath);
    if (await file.exists()) await file.delete();
    setState(() => _history.removeAt(index));
    await _saveHistory();
  }

  String _formatTimestamp(DateTime dt) {
    final local = dt.toLocal();
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final itemDay = DateTime(local.year, local.month, local.day);
    final time =
        '${local.hour.toString().padLeft(2, '0')}:${local.minute.toString().padLeft(2, '0')}';
    if (itemDay == today) return 'Today  $time';
    if (itemDay == today.subtract(const Duration(days: 1))) return 'Yesterday  $time';
    return '${local.month}/${local.day}/${local.year}  $time';
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? const Color(0xFF000000) : const Color(0xFFF2F2F2);

    return Scaffold(
      backgroundColor: bgColor,
      appBar: AppBar(
        backgroundColor: bgColor,
        foregroundColor: isDark ? Colors.white : Colors.black87,
        elevation: 0,
        centerTitle: false,
        title: Text(
          _selectedTabIndex == 0 ? 'Voice Recorder' : 'Saved Recordings',
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined, size: 26),
            tooltip: 'Settings',
            onPressed: () => _openSettingsPanel(context),
          ),
        ],
      ),
      body: Stack(
        children: [
          if (_isRecording)
            Positioned.fill(
              child: DecoratedBox(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: isDark
                        ? [
                            _samsungRed.withValues(alpha: 0.35),
                            const Color(0xFF000000),
                            const Color(0xFF000000),
                          ]
                        : [
                            _samsungRed.withValues(alpha: 0.18),
                            const Color(0xFFF2F2F2),
                            const Color(0xFFF2F2F2),
                          ],
                    stops: const [0.0, 0.45, 1.0],
                  ),
                ),
              ),
            ),
          SafeArea(
            child: IndexedStack(
              index: _selectedTabIndex,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(
                      child: _isRecording ? _buildRecordingView() : _buildRecordTab(context),
                    ),
                    _buildBottomControls(),
                  ],
                ),
                RecordingsPage(
                  history: _history,
                  player: _player,
                  onDelete: _deleteEntry,
                ),
              ],
            ),
          ),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedTabIndex,
        onTap: _selectTab,
        type: BottomNavigationBarType.fixed,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.mic_none_rounded),
            label: 'Record',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.list_alt_rounded),
            label: 'Recordings',
          ),
        ],
      ),
    );
  }

  Widget _buildRecordingView() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          _formatDuration(_recordDuration),
          style: TextStyle(
            fontSize: 56,
            fontWeight: FontWeight.w200,
            letterSpacing: 2,
            color: _isRecording ? _samsungRed : null,
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
        ),
        const SizedBox(height: 32),
        _SamsungWaveform(
          animation: _waveController,
          isActive: _isRecording,
          color: _samsungRed,
        ),
        const SizedBox(height: 24),
        Text(
          _statusMessage,
          style: TextStyle(
            fontSize: 14,
            color: Theme.of(context).textTheme.bodyMedium?.color?.withValues(alpha: 0.6),
          ),
        ),
      ],
    );
  }

  Widget _buildRecordTab(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final recentHistory = _history.take(2).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (_resultText.isNotEmpty || _isUploading) ...[
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
            child: _ResultCard(
              result: _resultText,
              isLoading: _isUploading,
            ),
          ),
        ],
        if (_resultText.isEmpty && !_isUploading)
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 24, 20, 0),
            child: Text(
              _statusMessage,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 15,
                color: Theme.of(context).textTheme.bodyMedium?.color?.withValues(alpha: 0.5),
              ),
            ),
          ),
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  'Recent analyses',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: isDark ? Colors.white54 : Colors.black54,
                    letterSpacing: 0.5,
                  ),
                ),
              ),
              TextButton(
                onPressed: () => _selectTab(1),
                child: const Text('View all'),
              ),
            ],
          ),
        ),
        Expanded(
          child: recentHistory.isEmpty
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.graphic_eq_rounded,
                        size: 48,
                        color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.3),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'No recordings yet',
                        style: TextStyle(
                          fontSize: 16,
                          color: Theme.of(context).textTheme.bodyMedium?.color?.withValues(alpha: 0.4),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Tap the red button below to start',
                        style: TextStyle(
                          fontSize: 13,
                          color: Theme.of(context).textTheme.bodyMedium?.color?.withValues(alpha: 0.3),
                        ),
                      ),
                    ],
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  itemCount: recentHistory.length,
                  itemBuilder: (context, index) {
                    final item = recentHistory[index];
                    return _RecordingListTile(
                      title: item.formatted,
                      subtitle: item.method.toUpperCase(),
                      timestamp: _formatTimestamp(item.timestamp),
                      onPlay: () => _playAudio(item.filePath),
                    );
                  },
                ),
        ),
      ],
    );
  }

  Widget _buildBottomControls() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(0, 8, 0, 28),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          _SamsungRecordButton(
            isRecording: _isRecording,
            isDisabled: _isStopping || _isUploading,
            onPressed: _toggleRecording,
          ),
          if (!_isRecording) ...[
            const SizedBox(height: 12),
            Text(
              'Tap to record',
              style: TextStyle(
                fontSize: 13,
                color: Theme.of(context).textTheme.bodyMedium?.color?.withValues(alpha: 0.4),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// Samsung-style large red circle record button.
/// Idle: solid red circle. Recording: white ring with red stop square inside.
class _SamsungRecordButton extends StatelessWidget {
  const _SamsungRecordButton({
    required this.isRecording,
    required this.isDisabled,
    required this.onPressed,
  });

  final bool isRecording;
  final bool isDisabled;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: isDisabled ? null : onPressed,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOutCubic,
        width: isRecording ? 72 : 80,
        height: isRecording ? 72 : 80,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: isRecording ? Colors.white : _samsungRed,
          boxShadow: [
            BoxShadow(
              color: _samsungRed.withValues(alpha: isRecording ? 0.25 : 0.45),
              blurRadius: isRecording ? 16 : 24,
              spreadRadius: isRecording ? 2 : 4,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Center(
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 250),
            curve: Curves.easeOutCubic,
            width: isRecording ? 28 : 80,
            height: isRecording ? 28 : 80,
            decoration: BoxDecoration(
              color: isRecording ? _samsungRed : _samsungRed,
              borderRadius: BorderRadius.circular(isRecording ? 6 : 40),
            ),
          ),
        ),
      ),
    );
  }
}

/// Animated waveform bars like Samsung Voice Recorder.
class _SamsungWaveform extends StatelessWidget {
  const _SamsungWaveform({
    required this.animation,
    required this.isActive,
    required this.color,
  });

  final Animation<double> animation;
  final bool isActive;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: animation,
      builder: (context, _) {
        return SizedBox(
          height: 100,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: List.generate(40, (i) {
              final phase = (animation.value * 2 * math.pi) + (i * 0.45);
              final wave = isActive
                  ? (math.sin(phase) * 0.5 + 0.5) * 0.7 + 0.15
                  : 0.12;
              final h = 8.0 + wave * 84;
              return Container(
                width: 3,
                height: h,
                margin: const EdgeInsets.symmetric(horizontal: 1.5),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: isActive ? 0.85 : 0.25),
                  borderRadius: BorderRadius.circular(2),
                ),
              );
            }),
          ),
        );
      },
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result, required this.isLoading});

  final String result;
  final bool isLoading;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 20),
      decoration: BoxDecoration(
        color: Theme.of(context).cardTheme.color,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: _samsungBlue.withValues(alpha: 0.15),
        ),
      ),
      child: Column(
        children: [
          Text(
            'KEYSTROKES DETECTED',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              letterSpacing: 1.5,
              color: Theme.of(context).colorScheme.secondary,
            ),
          ),
          const SizedBox(height: 12),
          if (isLoading)
            const SizedBox(
              height: 36,
              width: 36,
              child: CircularProgressIndicator(strokeWidth: 2.5, color: _samsungBlue),
            )
          else
            Text(
              result,
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 36,
                fontWeight: FontWeight.w700,
                color: _samsungBlue,
                letterSpacing: -0.5,
              ),
            ),
        ],
      ),
    );
  }
}

class _RecordingListTile extends StatelessWidget {
  const _RecordingListTile({
    required this.title,
    required this.subtitle,
    required this.timestamp,
    required this.onPlay,
  });

  final String title;
  final String subtitle;
  final String timestamp;
  final VoidCallback onPlay;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.fromLTRB(16, 14, 8, 14),
      decoration: BoxDecoration(
        color: Theme.of(context).cardTheme.color,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isDark ? Colors.white10 : Colors.black.withValues(alpha: 0.06),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: _samsungBlue.withValues(alpha: 0.12),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.graphic_eq, color: _samsungBlue, size: 22),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w600,
                    letterSpacing: -0.2,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  subtitle,
                  style: TextStyle(
                    fontSize: 12,
                    color: Theme.of(context).colorScheme.secondary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  timestamp,
                  style: TextStyle(
                    fontSize: 11,
                    color: isDark ? Colors.white38 : Colors.black38,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            icon: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: _samsungBlue.withValues(alpha: 0.12),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.play_arrow_rounded, color: _samsungBlue, size: 24),
            ),
            onPressed: onPlay,
          ),
        ],
      ),
    );
  }
}
