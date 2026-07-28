import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:keystroke_app/recording_storage.dart';

void main() {
  test(
    'allocates unique microsecond filenames and manages audio files',
    () async {
      final documents = await Directory.systemTemp.createTemp(
        'recording_storage_test_',
      );
      addTearDown(() async {
        if (await documents.exists()) {
          await documents.delete(recursive: true);
        }
      });
      final storage = DeviceRecordingStorage(
        documentsDirectoryProvider: () async => documents,
      );
      final timestamp = DateTime.fromMicrosecondsSinceEpoch(
        1720442096789123,
        isUtc: true,
      );

      final first = await storage.allocate(timestamp, 1);
      final second = await storage.allocate(timestamp, 2);

      expect(first.path, contains('1720442096789123_001.wav'));
      expect(second.path, contains('1720442096789123_002.wav'));
      expect(first.path, isNot(second.path));
      expect(await storage.hasAudio(first), isFalse);

      await first.writeAsBytes(<int>[82, 73, 70, 70]);
      expect(await storage.exists(first.path), isTrue);
      expect(await storage.hasAudio(first), isTrue);

      await storage.delete(first.path);
      expect(await storage.exists(first.path), isFalse);
    },
  );
}
