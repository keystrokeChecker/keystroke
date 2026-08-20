import 'package:flutter_test/flutter_test.dart';
import 'package:keystroke_app/backend_client.dart';
import 'package:keystroke_app/recording_result.dart';

void main() {
  group('RecordingResult', () {
    test('round-trips all current fields through JSON', () {
      final timestamp = DateTime.utc(2026, 7, 28, 10, 30);
      final result = RecordingResult(
        id: 'recording-1',
        timestamp: timestamp,
        formatted: '4|7',
        counts: <int>[4, 7],
        method: 'ml',
        filePath: r'C:\recordings\one.wav',
        status: AnalysisStatus.failed,
        errorMessage: 'Backend unavailable',
        reconstructedSequence: 'HELLO',
        detectedEventsCount: 2,
        events: const <KeystrokeEvent>[
          KeystrokeEvent(
            eventIndex: 0,
            timestampS: 0.15,
            prediction: 'H',
            confidence: 0.95,
            status: 'CONFIDENT',
            topK: [TopKPredictionItem(key: 'H', probability: 0.95)],
          ),
        ],
      );

      final restored = RecordingResult.fromJson(result.toJson());

      expect(restored.id, 'recording-1');
      expect(restored.timestamp, timestamp);
      expect(restored.formatted, '4|7');
      expect(restored.counts, <int>[4, 7]);
      expect(restored.method, 'ml');
      expect(restored.filePath, r'C:\recordings\one.wav');
      expect(restored.status, AnalysisStatus.failed);
      expect(restored.errorMessage, 'Backend unavailable');
      expect(restored.reconstructedSequence, 'HELLO');
      expect(restored.detectedEventsCount, 2);
      expect(restored.events.length, 1);
      expect(restored.events.first.prediction, 'H');
    });

    test('loads legacy records with stable compatibility defaults', () {
      final legacyJson = <String, dynamic>{
        'timestamp': '2026-07-28T10:30:00.000Z',
        'formatted': '3|5',
        'counts': <int>[3, 5],
        'method': 'rule',
        'filePath': r'C:\recordings\legacy.wav',
      };

      final first = RecordingResult.fromJson(legacyJson);
      final second = RecordingResult.fromJson(legacyJson);

      expect(first.id, startsWith('legacy-'));
      expect(second.id, first.id);
      expect(first.status, AnalysisStatus.completed);
      expect(first.errorMessage, isNull);
    });

    test('round-trips an explicitly cancelled analysis', () {
      final cancelled = RecordingResult(
        id: 'cancelled-1',
        timestamp: DateTime.utc(2026, 7, 28, 12),
        formatted: 'Analysis cancelled',
        counts: const <int>[],
        method: 'yamnet',
        filePath: 'cancelled.wav',
        status: AnalysisStatus.cancelled,
        errorMessage: 'Audio analysis was cancelled.',
      );

      final restored = RecordingResult.fromJson(cancelled.toJson());

      expect(restored.status, AnalysisStatus.cancelled);
      expect(restored.errorMessage, 'Audio analysis was cancelled.');
    });

    test('copyWith is immutable and can clear an error message', () {
      final sourceCounts = <int>[1, 2];
      final original = RecordingResult(
        id: 'recording-1',
        timestamp: DateTime.utc(2026),
        formatted: '1|2',
        counts: sourceCounts,
        method: 'rule',
        filePath: 'one.wav',
        status: AnalysisStatus.failed,
        errorMessage: 'Failed',
      );
      sourceCounts[0] = 99;

      final updated = original.copyWith(
        formatted: '2|2',
        counts: <int>[2, 2],
        status: AnalysisStatus.completed,
        errorMessage: null,
      );

      expect(original.formatted, '1|2');
      expect(original.counts, <int>[1, 2]);
      expect(original.status, AnalysisStatus.failed);
      expect(original.errorMessage, 'Failed');
      expect(updated.formatted, '2|2');
      expect(updated.counts, <int>[2, 2]);
      expect(updated.status, AnalysisStatus.completed);
      expect(updated.errorMessage, isNull);
      expect(() => updated.counts.add(3), throwsUnsupportedError);
    });

    test('rejects invalid field types and values', () {
      final valid = <String, dynamic>{
        'id': 'recording-1',
        'timestamp': '2026-07-28T10:30:00.000Z',
        'formatted': '3',
        'counts': <int>[3],
        'method': 'rule',
        'filePath': 'one.wav',
        'status': 'completed',
        'errorMessage': null,
      };

      expect(
        () => RecordingResult.fromJson(<String, dynamic>{
          ...valid,
          'timestamp': 'not-a-date',
        }),
        throwsFormatException,
      );
      expect(
        () => RecordingResult.fromJson(<String, dynamic>{
          ...valid,
          'counts': <Object>[3, '4'],
        }),
        throwsFormatException,
      );
      expect(
        () => RecordingResult.fromJson(<String, dynamic>{
          ...valid,
          'counts': <int>[-1],
        }),
        throwsFormatException,
      );
      expect(
        () => RecordingResult.fromJson(<String, dynamic>{
          ...valid,
          'status': 'pending',
        }),
        throwsFormatException,
      );
      expect(
        () => RecordingResult.fromJson(<String, dynamic>{
          ...valid,
          'errorMessage': 500,
        }),
        throwsFormatException,
      );
      expect(
        () => RecordingResult.fromJson(<String, dynamic>{...valid, 'id': ''}),
        throwsFormatException,
      );
      expect(
        () => RecordingResult.fromJson(<String, dynamic>{
          ...valid,
          'status': null,
        }),
        throwsFormatException,
      );
    });
  });
}
