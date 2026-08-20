import 'dart:io';

import 'package:path_provider/path_provider.dart';

typedef DocumentsDirectoryProvider = Future<Directory> Function();

abstract interface class RecordingStorage {
  Future<File> allocate(DateTime timestamp, int sequence);

  Future<bool> exists(String path);

  Future<bool> hasAudio(File file);

  Future<void> delete(String path);
}

class DeviceRecordingStorage implements RecordingStorage {
  DeviceRecordingStorage({
    DocumentsDirectoryProvider? documentsDirectoryProvider,
  }) : _documentsDirectoryProvider =
           documentsDirectoryProvider ?? getApplicationDocumentsDirectory;

  final DocumentsDirectoryProvider _documentsDirectoryProvider;

  @override
  Future<File> allocate(DateTime timestamp, int sequence) async {
    final documents = await _documentsDirectoryProvider();
    final recordings = Directory(
      '${documents.path}${Platform.pathSeparator}recordings',
    );
    await recordings.create(recursive: true);
    final filename =
        'recording_${timestamp.microsecondsSinceEpoch}_${sequence.toString().padLeft(3, '0')}.wav';
    return File('${recordings.path}${Platform.pathSeparator}$filename');
  }

  @override
  Future<bool> exists(String path) => File(path).exists();

  @override
  Future<bool> hasAudio(File file) async =>
      await file.exists() && await file.length() > 0;

  @override
  Future<void> delete(String path) async {
    final file = File(path);
    if (await file.exists()) await file.delete();
  }
}
