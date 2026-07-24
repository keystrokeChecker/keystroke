import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';
import 'dart:ui';
import 'package:path_provider/path_provider.dart'; // permanent storage
import 'package:shared_preferences/shared_preferences.dart'; // history persistence
import 'package:just_audio/just_audio.dart'; // playback

void main() {
  runApp(const KeystrokeApp());
}

class KeystrokeApp extends StatelessWidget {
  const KeystrokeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Keystroke Analyzer',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const KeystrokeHomePage(),
    );
  }
}

class RecordingResult {
  final DateTime timestamp;
  final String formatted;
  final List<int> counts;
  final String method;
  final String filePath; // persisted audio file location

  RecordingResult({
    required this.timestamp,
    required this.formatted,
    required this.counts,
    required this.method,
    required this.filePath,
  });

  // JSON serialization for SharedPreferences
  factory RecordingResult.fromJson(Map<String, dynamic> json) => RecordingResult(
        timestamp: DateTime.parse(json['timestamp'] as String),
        formatted: json['formatted'] as String,
        counts: List<int>.from(json['counts'] as List),
        method: json['method'] as String,
        filePath: json['filePath'] as String,
      );

  Map<String, dynamic> toJson() => {
        'timestamp': timestamp.toIso8601String(),
        'formatted': formatted,
        'counts': counts,
        'method': method,
        'filePath': filePath,
      };
}

class KeystrokeHomePage extends StatefulWidget {
  const KeystrokeHomePage({super.key});

  @override
  State<KeystrokeHomePage> createState() => _KeystrokeHomePageState();
}

class _KeystrokeHomePageState extends State<KeystrokeHomePage> {
  final AudioRecorder _recorder = AudioRecorder();
  final TextEditingController _backendController = TextEditingController();
  final List<RecordingResult> _history = [];
  final AudioPlayer _player = AudioPlayer(); // playback engine

  bool _isRecording = false;
  bool _isStopping = false; // guard double‑stop
  String _statusMessage = 'Ready to record.';
  String _resultText = '';
  String _selectedMethod = 'rule';

  @override
  void initState() {
    super.initState();
    _backendController.text = 'http://<YOUR_MACHINE_IP>:8000';
    _loadHistory(); // restore saved history
  }

  @override
  void dispose() {
    _backendController.dispose();
    _recorder.dispose(); // clean native resources
    _player.dispose(); // release playback resources
    super.dispose();
  }

  Future<File> _buildTempFile() async {
    // Permanent recordings folder inside app's documents directory
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

    setState(() {
      _isRecording = true;
      _statusMessage = 'Recording... tap Stop to finish.';
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

      setState(() => _statusMessage = 'Uploading audio and waiting for backend response...');
      await _uploadRecording(file);
    } on Exception catch (e) {
      setState(() => _statusMessage = 'Stop failed: $e');
    } finally {
      _isStopping = false;
      setState(() => _isRecording = false);
    }
  }

  Future<void> _uploadRecording(File file) async {
    final backendUrl = _backendController.text.trim();
    if (backendUrl.isEmpty) {
      setState(() => _statusMessage = 'Please enter the backend URL first.');
      return;
    }
    final uri = Uri.tryParse(backendUrl);
    if (uri == null || (uri.scheme != 'http' && uri.scheme != 'https')) {
      setState(() => _statusMessage = 'Enter a valid backend URL, e.g. http://192.168.1.100:8000');
      return;
    }
    final analyzeUri = uri.replace(
      path: uri.path.endsWith('/') ? '${uri.path}analyze' : '${uri.path}/analyze',
    );
    try {
      final request = http.MultipartRequest('POST', analyzeUri);
      request.fields['method'] = _selectedMethod;
      request.files.add(await http.MultipartFile.fromPath('file', file.path, filename: 'recording.wav'));

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode != 200) {
        setState(() => _statusMessage = 'Backend returned ${response.statusCode}: ${response.reasonPhrase}');
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
        _statusMessage = 'Received result from backend.';
        _history.insert(0, result);
      });
      await _saveHistory(); // persist updated list
    } catch (error) {
      setState(() => _statusMessage = 'Upload failed: $error');
    }
  }

  void _openSettingsPanel(BuildContext context) {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
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
                value: _selectedMethod,
                decoration: const InputDecoration(
                  labelText: 'Prediction method',
                  border: OutlineInputBorder(),
                ),
                items: const [
                  DropdownMenuItem(value: 'rule', child: Text('Rule‑based')),
                  DropdownMenuItem(value: 'ml', child: Text('ML‑based')),
                ],
                onChanged: (value) {
                  if (value != null) {
                    setState(() => _selectedMethod = value);
                  }
                },
              ),
            ],
          ),
        );
      },
    );
  }

  // --------------------------------------------------------------
  // Persistence helpers
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
      setState(() => _history.addAll(loaded.reversed)); // newest first
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Keystroke Sound Analyzer'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: 'Settings',
            onPressed: () => _openSettingsPanel(context),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Result Card
            const Text(
              'Keystrokes Detected',
              style: TextStyle(fontSize: 14, color: Colors.grey, letterSpacing: 1.2),
            ),
            const SizedBox(height: 6),
            Card(
              color: Colors.deepPurple.shade100,
              elevation: 6,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              margin: const EdgeInsets.symmetric(horizontal: 24),
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 12),
                child: Center(
                  child: Text(
                    _resultText.isEmpty ? '—' : _resultText,
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.deepPurple),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
            // Record button with animation
            AnimatedScale(
              scale: _isRecording ? 1.1 : 1.0,
              duration: const Duration(milliseconds: 300),
              child: ElevatedButton.icon(
                icon: Icon(_isRecording ? Icons.stop : Icons.mic),
                label: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
                onPressed: _isStopping ? null : _toggleRecording,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isRecording ? Colors.redAccent : null,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  elevation: _isRecording ? 12 : 4,
                  shadowColor: _isRecording ? Colors.redAccent.withOpacity(0.6) : null,
                ),
              ),
            ),
            const SizedBox(height: 12),
            // Status message
            Text(
              _statusMessage,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.deepPurple.shade700),
            ),
            const SizedBox(height: 16),
            const Text(
              'History',
              style: TextStyle(fontWeight: FontWeight.bold, color: Colors.deepPurple),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: _history.isEmpty
                  ? const Center(child: Text('No completed recordings yet.'))
                  : ListView.builder(
                      itemCount: _history.length,
                      itemBuilder: (context, index) {
                        final item = _history[index];
                        return Dismissible(
                          key: ValueKey(item.filePath),
                          direction: DismissDirection.endToStart,
                          background: Container(
                            color: Colors.redAccent,
                            alignment: Alignment.centerRight,
                            padding: const EdgeInsets.symmetric(horizontal: 20),
                            child: const Icon(Icons.delete, color: Colors.white),
                          ),
                          onDismissed: (_) async {
                            await _deleteEntry(index);
                          },
                          child: Card(
                            elevation: 4,
                            margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                            child: Padding(
                              padding: const EdgeInsets.all(12),
                              child: Row(
                                children: [
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(item.formatted, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                                        const SizedBox(height: 4),
                                        Text('${item.method.toUpperCase()} • ${item.counts.join('|')}', style: TextStyle(fontSize: 12, color: Colors.deepPurple.shade600)),
                                        Text('${item.timestamp.toLocal()}', style: TextStyle(fontSize: 11, color: Colors.grey.shade600)),
                                      ],
                                    ),
                                  ),
                                  IconButton(
                                    icon: const Icon(Icons.play_arrow),
                                    tooltip: 'Play recording',
                                    onPressed: () => _playAudio(item.filePath),
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
    );
  }
}
