import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import 'recording_result.dart';

class AppSettings {
  final String backendUrl;
  final String method;

  const AppSettings({this.backendUrl = '', this.method = 'ml'});

  AppSettings copyWith({String? backendUrl, String? method}) => AppSettings(
    backendUrl: backendUrl ?? this.backendUrl,
    method: method ?? this.method,
  );
}

class LocalStore {
  static const String settingsKey = 'keystroke_settings';
  static const String backendUrlKey = 'keystroke_backend_url';
  static const String methodKey = 'keystroke_prediction_method';
  static const String historyKey = 'keystroke_history';

  final SharedPreferences _preferences;

  LocalStore(this._preferences);

  static Future<LocalStore> create() async {
    final preferences = await SharedPreferences.getInstance();
    return LocalStore(preferences);
  }

  Future<AppSettings> loadSettings() async {
    final encoded = _readString(settingsKey);
    if (encoded != null) {
      try {
        final decoded = jsonDecode(encoded);
        if (decoded is Map) {
          final settings = Map<String, dynamic>.from(decoded);
          final backendUrl = settings['backendUrl'];
          final method = settings['method'];
          if (backendUrl is String) {
            return AppSettings(
              backendUrl: backendUrl,
              method: method is String ? method : 'ml',
            );
          }
        }
      } on Object {
        // Fall through to legacy per-field settings.
      }
    }
    return AppSettings(
      backendUrl: _readString(backendUrlKey) ?? '',
      method: _readString(methodKey) ?? 'ml',
    );
  }

  Future<void> saveSettings(AppSettings settings) async {
    final saved = await _preferences.setString(
      settingsKey,
      jsonEncode(<String, String>{
        'backendUrl': settings.backendUrl,
        'method': settings.method,
      }),
    );
    if (!saved) throw StateError('Settings could not be persisted.');
  }

  Future<List<RecordingResult>> loadHistory() async {
    final encoded = _readString(historyKey);
    if (encoded == null) return <RecordingResult>[];

    final Object? decoded;
    try {
      decoded = jsonDecode(encoded);
    } on FormatException {
      return <RecordingResult>[];
    }

    if (decoded is! List) return <RecordingResult>[];

    final history = <RecordingResult>[];
    for (final entry in decoded) {
      if (entry is! Map) continue;

      try {
        final json = Map<String, dynamic>.from(entry);
        history.add(RecordingResult.fromJson(json));
      } on Object {
        // A damaged record should not prevent the remaining history from loading.
      }
    }
    return history;
  }

  Future<void> saveHistory(List<RecordingResult> history) async {
    final encoded = jsonEncode(
      history.map((recording) => recording.toJson()).toList(),
    );
    final saved = await _preferences.setString(historyKey, encoded);
    if (!saved) throw StateError('History could not be persisted.');
  }

  String? _readString(String key) {
    try {
      return _preferences.getString(key);
    } on Object {
      return null;
    }
  }
}
