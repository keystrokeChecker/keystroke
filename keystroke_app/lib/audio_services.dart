import 'package:just_audio/just_audio.dart';
import 'package:record/record.dart';

abstract interface class RecorderService {
  Future<bool> hasPermission();

  Future<void> start(String path);

  Future<String?> stop();

  Future<void> cancel();

  Future<void> dispose();
}

class DeviceRecorderService implements RecorderService {
  DeviceRecorderService({AudioRecorder? recorder})
    : _recorder = recorder ?? AudioRecorder();

  final AudioRecorder _recorder;

  @override
  Future<bool> hasPermission() => _recorder.hasPermission();

  @override
  Future<void> start(String path) => _recorder.start(
    const RecordConfig(encoder: AudioEncoder.wav),
    path: path,
  );

  @override
  Future<String?> stop() => _recorder.stop();

  @override
  Future<void> cancel() => _recorder.cancel();

  @override
  Future<void> dispose() => _recorder.dispose();
}

abstract interface class PlaybackService {
  PlaybackStatus get status;

  Stream<PlaybackStatus> get statusStream;

  Future<void> load(String path);

  Future<void> play();

  Future<void> pause();

  Future<void> stop();

  Future<void> dispose();
}

class DevicePlaybackService implements PlaybackService {
  DevicePlaybackService({AudioPlayer? player})
    : _player = player ?? AudioPlayer();

  final AudioPlayer _player;

  @override
  PlaybackStatus get status => _mapPlayerState(_player.playerState);

  @override
  Stream<PlaybackStatus> get statusStream =>
      _player.playerStateStream.map(_mapPlayerState).distinct();

  @override
  Future<void> load(String path) async {
    await _player.setFilePath(path);
  }

  @override
  Future<void> play() => _player.play();

  @override
  Future<void> pause() => _player.pause();

  @override
  Future<void> stop() => _player.stop();

  @override
  Future<void> dispose() => _player.dispose();

  PlaybackStatus _mapPlayerState(PlayerState state) {
    if (state.processingState == ProcessingState.completed) {
      return PlaybackStatus.completed;
    }
    if (state.playing) return PlaybackStatus.playing;
    if (state.processingState == ProcessingState.idle) {
      return PlaybackStatus.stopped;
    }
    return PlaybackStatus.paused;
  }
}

enum PlaybackStatus { stopped, paused, playing, completed }
