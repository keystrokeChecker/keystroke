enum AnalysisStatus { completed, failed, cancelled }

class RecordingResult {
  static const Object _notProvided = Object();

  final String id;
  final DateTime timestamp;
  final String formatted;
  final List<int> counts;
  final String method;
  final String filePath;
  final AnalysisStatus status;
  final String? errorMessage;

  RecordingResult({
    required this.id,
    required this.timestamp,
    required this.formatted,
    required List<int> counts,
    required this.method,
    required this.filePath,
    this.status = AnalysisStatus.completed,
    this.errorMessage,
  }) : counts = List<int>.unmodifiable(counts);

  factory RecordingResult.fromJson(Map<String, dynamic> json) {
    final timestampValue = _requiredString(json, 'timestamp');
    final timestamp = DateTime.tryParse(timestampValue);
    if (timestamp == null) {
      throw const FormatException('Recording timestamp is invalid.');
    }

    final filePath = _requiredString(json, 'filePath');
    final id = json.containsKey('id')
        ? _validatedId(json['id'])
        : _legacyId(timestamp, filePath);

    final rawCounts = json['counts'];
    if (rawCounts is! List) {
      throw const FormatException('Recording counts must be a list.');
    }

    final counts = <int>[];
    for (final value in rawCounts) {
      if (value is int && value >= 0) {
        counts.add(value);
      } else if (value is num &&
          value.isFinite &&
          value >= 0 &&
          value == value.round()) {
        counts.add(value.toInt());
      } else {
        throw const FormatException(
          'Every recording count must be an integer.',
        );
      }
    }

    final status = json.containsKey('status')
        ? _parseStatus(json['status'])
        : AnalysisStatus.completed;

    final rawErrorMessage = json['errorMessage'];
    if (rawErrorMessage != null && rawErrorMessage is! String) {
      throw const FormatException(
        'Recording errorMessage must be a string or null.',
      );
    }

    return RecordingResult(
      id: id,
      timestamp: timestamp,
      formatted: _requiredString(json, 'formatted'),
      counts: counts,
      method: _requiredString(json, 'method'),
      filePath: filePath,
      status: status,
      errorMessage: rawErrorMessage as String?,
    );
  }

  RecordingResult copyWith({
    String? id,
    DateTime? timestamp,
    String? formatted,
    List<int>? counts,
    String? method,
    String? filePath,
    AnalysisStatus? status,
    Object? errorMessage = _notProvided,
  }) {
    if (!identical(errorMessage, _notProvided) &&
        errorMessage != null &&
        errorMessage is! String) {
      throw ArgumentError.value(
        errorMessage,
        'errorMessage',
        'must be a string or null',
      );
    }

    return RecordingResult(
      id: id ?? this.id,
      timestamp: timestamp ?? this.timestamp,
      formatted: formatted ?? this.formatted,
      counts: counts ?? this.counts,
      method: method ?? this.method,
      filePath: filePath ?? this.filePath,
      status: status ?? this.status,
      errorMessage: identical(errorMessage, _notProvided)
          ? this.errorMessage
          : errorMessage as String?,
    );
  }

  Map<String, dynamic> toJson() => <String, dynamic>{
    'id': id,
    'timestamp': timestamp.toIso8601String(),
    'formatted': formatted,
    'counts': counts,
    'method': method,
    'filePath': filePath,
    'status': status.name,
    'errorMessage': errorMessage,
  };

  static String _requiredString(Map<String, dynamic> json, String key) {
    final value = json[key];
    if (value is! String) {
      throw FormatException('Recording $key must be a string.');
    }
    return value;
  }

  static String _validatedId(Object? rawId) {
    if (rawId is! String || rawId.trim().isEmpty) {
      throw const FormatException('Recording id must be a non-empty string.');
    }
    return rawId;
  }

  static AnalysisStatus _parseStatus(Object? rawStatus) {
    if (rawStatus is! String) {
      throw const FormatException('Recording status must be a string.');
    }

    for (final status in AnalysisStatus.values) {
      if (status.name == rawStatus) return status;
    }
    throw FormatException('Unknown recording status: $rawStatus.');
  }

  static String _legacyId(DateTime timestamp, String filePath) {
    var pathHash = 0x811c9dc5;
    for (final codeUnit in filePath.codeUnits) {
      pathHash ^= codeUnit;
      pathHash = (pathHash * 0x01000193) & 0xffffffff;
    }
    final hash = pathHash.toRadixString(16).padLeft(8, '0');
    return 'legacy-${timestamp.microsecondsSinceEpoch}-$hash';
  }
}
