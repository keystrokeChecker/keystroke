class RecordingResult {
  final DateTime timestamp;
  final String formatted;
  final List<int> counts;
  final String filePath;

  RecordingResult({
    required this.timestamp,
    required this.formatted,
    required this.counts,
    required this.filePath,
  });

  factory RecordingResult.fromJson(Map<String, dynamic> json) =>
      RecordingResult(
        timestamp: DateTime.parse(json['timestamp'] as String),
        formatted: json['formatted'] as String,
        counts: List<int>.from(json['counts'] as List),
        filePath: json['filePath'] as String,
      );

  Map<String, dynamic> toJson() => {
    'timestamp': timestamp.toIso8601String(),
    'formatted': formatted,
    'counts': counts,
    'filePath': filePath,
  };
}
