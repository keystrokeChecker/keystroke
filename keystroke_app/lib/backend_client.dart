import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

typedef HttpClientFactory = http.Client Function();

enum BackendFailureType {
  invalidRequest,
  timeout,
  cancelled,
  network,
  http,
  invalidResponse,
}

class BackendClientException implements Exception {
  const BackendClientException({
    required this.type,
    required this.message,
    this.statusCode,
    this.detail,
  });

  final BackendFailureType type;
  final String message;
  final int? statusCode;
  final String? detail;

  @override
  String toString() => message;
}

class TopKPredictionItem {
  const TopKPredictionItem({
    required this.key,
    required this.probability,
  });

  final String key;
  final double probability;

  factory TopKPredictionItem.fromJson(Map<String, dynamic> json) {
    return TopKPredictionItem(
      key: json['key'] as String? ?? '',
      probability: (json['probability'] as num?)?.toDouble() ?? 0.0,
    );
  }

  Map<String, dynamic> toJson() => <String, dynamic>{
    'key': key,
    'probability': probability,
  };
}

class KeystrokeEvent {
  const KeystrokeEvent({
    required this.eventIndex,
    required this.timestampS,
    required this.prediction,
    required this.confidence,
    required this.status,
    this.topK = const <TopKPredictionItem>[],
  });

  final int eventIndex;
  final double timestampS;
  final String prediction;
  final double confidence;
  final String status;
  final List<TopKPredictionItem> topK;

  factory KeystrokeEvent.fromJson(Map<String, dynamic> json) {
    final rawTopK = json['top_k'] as List? ?? json['topK'] as List? ?? const [];
    final topKList = <TopKPredictionItem>[];
    for (final item in rawTopK) {
      if (item is Map) {
        topKList.add(TopKPredictionItem.fromJson(Map<String, dynamic>.from(item)));
      }
    }

    return KeystrokeEvent(
      eventIndex: (json['event_index'] as num?)?.toInt() ?? (json['eventIndex'] as num?)?.toInt() ?? 0,
      timestampS: (json['timestamp_s'] as num?)?.toDouble() ?? (json['timestampS'] as num?)?.toDouble() ?? 0.0,
      prediction: json['prediction'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      status: json['status'] as String? ?? 'CONFIDENT',
      topK: List<TopKPredictionItem>.unmodifiable(topKList),
    );
  }

  Map<String, dynamic> toJson() => <String, dynamic>{
    'event_index': eventIndex,
    'timestamp_s': timestampS,
    'prediction': prediction,
    'confidence': confidence,
    'status': status,
    'top_k': topK.map((item) => item.toJson()).toList(),
  };
}

class ModelInfo {
  const ModelInfo({
    required this.modelName,
    required this.featureSet,
    required this.normalization,
    required this.classesCount,
    required this.accuracy,
    required this.macroF1,
    required this.top3Accuracy,
  });

  final String modelName;
  final String featureSet;
  final String normalization;
  final int classesCount;
  final double accuracy;
  final double macroF1;
  final double top3Accuracy;

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    final metrics = (json['validation_metrics'] as Map?) ?? const {};
    return ModelInfo(
      modelName: json['model_name'] as String? ?? 'Unknown',
      featureSet: json['feature_set'] as String? ?? 'D',
      normalization: json['normalization'] as String? ?? 'P2',
      classesCount: (json['classes_count'] as num?)?.toInt() ?? 27,
      accuracy: (metrics['accuracy'] as num?)?.toDouble() ?? 0.0,
      macroF1: (metrics['macro_f1'] as num?)?.toDouble() ?? 0.0,
      top3Accuracy: (metrics['top3_accuracy'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class AnalyzeResponse {
  AnalyzeResponse({
    required List<int> counts,
    required this.formatted,
    this.reconstructedSequence = '',
    this.detectedEventsCount = 0,
    List<KeystrokeEvent>? events,
  })  : counts = List<int>.unmodifiable(counts),
        events = List<KeystrokeEvent>.unmodifiable(events ?? const <KeystrokeEvent>[]);

  final List<int> counts;
  final String formatted;
  final String reconstructedSequence;
  final int detectedEventsCount;
  final List<KeystrokeEvent> events;
}

class BackendClient {
  factory BackendClient({
    required String baseUrl,
    Duration timeout = const Duration(seconds: 20),
    HttpClientFactory? clientFactory,
  }) {
    if (timeout <= Duration.zero) {
      throw ArgumentError.value(
        timeout,
        'timeout',
        'Must be greater than zero',
      );
    }

    return BackendClient._(
      normalizeBaseUrl(baseUrl),
      timeout,
      clientFactory ?? http.Client.new,
    );
  }

  BackendClient._(this.baseUri, this._timeout, this._clientFactory);

  final Uri baseUri;
  final Duration _timeout;
  final HttpClientFactory _clientFactory;
  final Set<http.Client> _activeClients = <http.Client>{};
  final Set<http.Client> _cancelledClients = <http.Client>{};

  String get baseUrl => baseUri.toString();

  int get activeRequestCount => _activeClients.length;

  static Uri normalizeBaseUrl(String value) {
    final candidate = value.trim();
    if (candidate.isEmpty) {
      throw ArgumentError.value(value, 'baseUrl', 'Must not be empty');
    }

    final Uri uri;
    try {
      uri = Uri.parse(candidate);
      // Accessing the port also validates malformed explicit port values.
      uri.port;
    } on FormatException {
      throw ArgumentError.value(value, 'baseUrl', 'Must be a valid URL');
    }

    final scheme = uri.scheme.toLowerCase();
    if ((scheme != 'http' && scheme != 'https') || uri.host.isEmpty) {
      throw ArgumentError.value(
        value,
        'baseUrl',
        'Must be an absolute http(s) URL with a host',
      );
    }
    if (uri.hasQuery || uri.hasFragment) {
      throw ArgumentError.value(
        value,
        'baseUrl',
        'Must not contain a query string or fragment',
      );
    }

    var normalizedPath = uri.path;
    while (normalizedPath.endsWith('/')) {
      normalizedPath = normalizedPath.substring(0, normalizedPath.length - 1);
    }

    return uri.replace(
      scheme: scheme,
      host: uri.host.toLowerCase(),
      path: normalizedPath,
    );
  }

  Future<void> testConnection() async {
    final response = await _withClient(
      action: 'Connection test',
      operation: (client) => client.get(_endpoint('health')),
    );
    _requireSuccess(response);

    final body = _decodeObject(response.body);
    if (body['status'] != 'ok') {
      throw const BackendClientException(
        type: BackendFailureType.invalidResponse,
        message: 'Backend health response must contain {"status":"ok"}.',
      );
    }
  }

  Future<ModelInfo> fetchModelInfo() async {
    final response = await _withClient(
      action: 'Model info query',
      operation: (client) => client.get(_endpoint('model-info')),
    );
    _requireSuccess(response);
    final body = _decodeObject(response.body);
    return ModelInfo.fromJson(body);
  }

  Future<AnalyzeResponse> analyze({
    required File file,
    String method = 'default',
  }) async {
    final normalizedMethod = method.trim().toLowerCase();
    final effectiveMethod =
        normalizedMethod.isEmpty ? 'default' : normalizedMethod;

    await _validateAudioFile(file);

    final response = await _withClient(
      action: 'Audio analysis',
      operation: (client) async {
        final request = http.MultipartRequest('POST', _endpoint('analyze'));
        request.fields['method'] = effectiveMethod;
        request.files.add(
          await http.MultipartFile.fromPath(
            'file',
            file.path,
            filename: _fileName(file.path),
          ),
        );
        final streamedResponse = await client.send(request);
        return http.Response.fromStream(streamedResponse);
      },
    );
    _requireSuccess(response);

    final body = _decodeObject(response.body);
    final rawCounts = body['counts'];
    final formatted = body['formatted'];
    if (rawCounts is! List) {
      throw const BackendClientException(
        type: BackendFailureType.invalidResponse,
        message: 'Backend response field "counts" must be a list.',
      );
    }
    if (formatted is! String) {
      throw const BackendClientException(
        type: BackendFailureType.invalidResponse,
        message: 'Backend response field "formatted" must be a string.',
      );
    }

    final counts = <int>[];
    for (final count in rawCounts) {
      if (count is! int || count < 0) {
        throw const BackendClientException(
          type: BackendFailureType.invalidResponse,
          message:
              'Backend response field "counts" must contain non-negative integers.',
        );
      }
      counts.add(count);
    }

    final reconstructed = body['reconstructed_sequence'] as String? ??
        body['reconstructedSequence'] as String? ??
        '';
    final detectedCount = (body['detected_events_count'] as num?)?.toInt() ??
        (body['detectedEventsCount'] as num?)?.toInt() ??
        0;

    final eventsList = <KeystrokeEvent>[];
    final rawEvents = body['events'];
    if (rawEvents is List) {
      for (final rawEv in rawEvents) {
        if (rawEv is Map) {
          eventsList.add(KeystrokeEvent.fromJson(Map<String, dynamic>.from(rawEv)));
        }
      }
    }

    return AnalyzeResponse(
      counts: counts,
      formatted: formatted,
      reconstructedSequence: reconstructed,
      detectedEventsCount: detectedCount,
      events: eventsList,
    );
  }

  void cancelActiveRequests() {
    for (final client in _activeClients.toList(growable: false)) {
      _cancelledClients.add(client);
      try {
        client.close();
      } on Object {
        // Closing one faulty injected client must not prevent cancelling the
        // remaining requests.
      }
    }
  }

  Uri _endpoint(String name) =>
      baseUri.replace(pathSegments: <String>[...baseUri.pathSegments, name]);

  Future<T> _withClient<T>({
    required String action,
    required Future<T> Function(http.Client client) operation,
  }) async {
    final client = _clientFactory();
    _activeClients.add(client);
    try {
      final result = await operation(client).timeout(_timeout);
      if (_cancelledClients.contains(client)) {
        throw BackendClientException(
          type: BackendFailureType.cancelled,
          message: '$action was cancelled.',
        );
      }
      return result;
    } catch (error) {
      if (_cancelledClients.contains(client)) {
        throw BackendClientException(
          type: BackendFailureType.cancelled,
          message: '$action was cancelled.',
        );
      }
      if (error is BackendClientException) {
        rethrow;
      }
      if (error is TimeoutException) {
        throw BackendClientException(
          type: BackendFailureType.timeout,
          message: '$action timed out after ${_describeDuration(_timeout)}.',
        );
      }
      if (error is FileSystemException) {
        throw BackendClientException(
          type: BackendFailureType.invalidRequest,
          message: 'Could not read the audio file: ${error.message}',
          detail: error.message,
        );
      }
      if (error is http.ClientException || error is IOException) {
        throw BackendClientException(
          type: BackendFailureType.network,
          message: '$action failed because the backend could not be reached.',
          detail: error.toString(),
        );
      }
      rethrow;
    } finally {
      _activeClients.remove(client);
      final wasCancelled = _cancelledClients.remove(client);
      if (!wasCancelled) {
        try {
          client.close();
        } on Object {
          // Request cleanup must not replace the operation's useful result or
          // error with a client-specific close failure.
        }
      }
    }
  }

  Future<void> _validateAudioFile(File file) async {
    try {
      if (!await file.exists()) {
        throw const BackendClientException(
          type: BackendFailureType.invalidRequest,
          message: 'Audio file does not exist.',
        );
      }
      if (await file.length() == 0) {
        throw const BackendClientException(
          type: BackendFailureType.invalidRequest,
          message: 'Audio file is empty.',
        );
      }
    } on FileSystemException catch (error) {
      throw BackendClientException(
        type: BackendFailureType.invalidRequest,
        message: 'Could not read the audio file: ${error.message}',
        detail: error.message,
      );
    }
  }

  void _requireSuccess(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return;
    }

    final detail = _responseDetail(response.body);
    final reason = detail ?? response.reasonPhrase;
    final suffix = reason == null || reason.trim().isEmpty ? '' : ': $reason';
    throw BackendClientException(
      type: BackendFailureType.http,
      message: 'Backend returned HTTP ${response.statusCode}$suffix',
      statusCode: response.statusCode,
      detail: detail,
    );
  }

  Map<String, dynamic> _decodeObject(String source) {
    final dynamic decoded;
    try {
      decoded = jsonDecode(source);
    } on FormatException {
      throw const BackendClientException(
        type: BackendFailureType.invalidResponse,
        message: 'Backend returned invalid JSON.',
      );
    }
    if (decoded is! Map<String, dynamic>) {
      throw const BackendClientException(
        type: BackendFailureType.invalidResponse,
        message: 'Backend JSON response must be an object.',
      );
    }
    return decoded;
  }

  String? _responseDetail(String source) {
    final trimmed = source.trim();
    if (trimmed.isEmpty) {
      return null;
    }

    try {
      final decoded = jsonDecode(trimmed);
      if (decoded is Map<String, dynamic>) {
        final detail = decoded['detail'] ?? decoded['message'];
        if (detail is String && detail.trim().isNotEmpty) {
          return _truncate(detail.trim());
        }
        if (detail != null) {
          return _truncate(jsonEncode(detail));
        }
      } else if (decoded is String && decoded.trim().isNotEmpty) {
        return _truncate(decoded.trim());
      }
    } on FormatException {
      // Plain-text and HTML errors are still useful to the caller.
    }
    return _truncate(trimmed);
  }

  String _truncate(String value) {
    const maximumLength = 500;
    if (value.length <= maximumLength) {
      return value;
    }
    return '${value.substring(0, maximumLength)}...';
  }

  String _fileName(String path) {
    final parts = path.split(RegExp(r'[/\\]'));
    final name = parts.isEmpty ? '' : parts.last;
    return name.isEmpty ? 'recording.wav' : name;
  }

  String _describeDuration(Duration duration) {
    if (duration.inMilliseconds < 1000) {
      return '${duration.inMilliseconds} ms';
    }
    if (duration.inMilliseconds % 1000 == 0) {
      return '${duration.inSeconds} s';
    }
    return '${duration.inMilliseconds / 1000} s';
  }
}
