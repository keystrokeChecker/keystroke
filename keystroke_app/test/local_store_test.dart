import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:keystroke_app/local_store.dart';
import 'package:keystroke_app/recording_result.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('LocalStore', () {
    test('returns default settings when none have been saved', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{});
      final store = LocalStore(await SharedPreferences.getInstance());

      final settings = await store.loadSettings();

      expect(settings.backendUrl, isEmpty);
      expect(settings.method, 'ml');
    });

    test('saves and loads settings', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{});
      final store = LocalStore(await SharedPreferences.getInstance());

      await store.saveSettings(
        const AppSettings(backendUrl: 'http://192.168.1.10:8000', method: 'ml'),
      );
      final settings = await store.loadSettings();

      expect(settings.backendUrl, 'http://192.168.1.10:8000');
      expect(settings.method, 'ml');
    });

    test('loads legacy per-field settings', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{
        LocalStore.backendUrlKey: 'http://192.168.1.11:8000',
        LocalStore.methodKey: 'yamnet',
      });
      final store = LocalStore(await SharedPreferences.getInstance());

      final settings = await store.loadSettings();

      expect(settings.backendUrl, 'http://192.168.1.11:8000');
      expect(settings.method, 'yamnet');
    });

    test('preserves newest-first history order', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{});
      final store = LocalStore(await SharedPreferences.getInstance());
      final newest = _recording('newest', DateTime.utc(2026, 7, 28, 11));
      final oldest = _recording('oldest', DateTime.utc(2026, 7, 28, 10));

      await store.saveHistory(<RecordingResult>[newest, oldest]);
      final restored = await store.loadHistory();

      expect(restored.map((recording) => recording.id), <String>[
        'newest',
        'oldest',
      ]);
    });

    test('skips malformed history entries and keeps valid neighbors', () async {
      final newest = _recording('newest', DateTime.utc(2026, 7, 28, 11));
      final oldest = _recording('oldest', DateTime.utc(2026, 7, 28, 10));
      SharedPreferences.setMockInitialValues(<String, Object>{
        LocalStore.historyKey: jsonEncode(<Object?>[
          newest.toJson(),
          'not an object',
          <String, Object?>{'id': 'broken', 'timestamp': 'invalid'},
          oldest.toJson(),
        ]),
      });
      final store = LocalStore(await SharedPreferences.getInstance());

      final restored = await store.loadHistory();

      expect(restored.map((recording) => recording.id), <String>[
        'newest',
        'oldest',
      ]);
    });

    test('returns empty history for malformed top-level JSON', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{
        LocalStore.historyKey: '{invalid json',
      });
      final store = LocalStore(await SharedPreferences.getInstance());

      expect(await store.loadHistory(), isEmpty);
    });
  });
}

RecordingResult _recording(String id, DateTime timestamp) => RecordingResult(
  id: id,
  timestamp: timestamp,
  formatted: '3|5',
  counts: <int>[3, 5],
  method: 'rule',
  filePath: '$id.wav',
);
