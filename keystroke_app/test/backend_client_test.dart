import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:keystroke_app/backend_client.dart';

void main() {
  group('BackendClient URL handling', () {
    test('normalizes whitespace, case, and trailing slashes', () {
      final client = BackendClient(
        baseUrl: '  HTTP://Example.COM:8000/api///  ',
      );

      expect(client.baseUrl, 'http://example.com:8000/api');
    });

    test('rejects non-http and ambiguous base URLs', () {
      for (final value in <String>[
        '',
        'localhost:8000',
        'ftp://example.com',
        'http:///missing-host',
        'https://example.com?token=secret',
        'https://example.com/#fragment',
      ]) {
        expect(
          () => BackendClient(baseUrl: value),
          throwsArgumentError,
          reason: value,
        );
      }
    });
  });

  group('testConnection', () {
    test('GETs health below a base path and requires status ok', () async {
      late _RecordingClient recordingClient;
      final client = BackendClient(
        baseUrl: 'http://localhost:8000/v1/',
        clientFactory: () {
          recordingClient = _RecordingClient((request) async {
            expect(request.method, 'GET');
            expect(request.url, Uri.parse('http://localhost:8000/v1/health'));
            return _jsonResponse(<String, dynamic>{'status': 'ok'});
          });
          return recordingClient;
        },
      );

      await client.testConnection();

      expect(client.activeRequestCount, 0);
      expect(recordingClient.closeCount, 1);
    });

    test('rejects a successful response with the wrong schema', () async {
      final client = BackendClient(
        baseUrl: 'http://localhost:8000',
        clientFactory: () => _RecordingClient(
          (_) async => _jsonResponse(<String, dynamic>{'status': 'degraded'}),
        ),
      );

      await expectLater(
        client.testConnection(),
        throwsA(
          isA<BackendClientException>().having(
            (error) => error.type,
            'type',
            BackendFailureType.invalidResponse,
          ),
        ),
      );
    });

    test('surfaces FastAPI detail and HTTP status', () async {
      final client = BackendClient(
        baseUrl: 'http://localhost:8000',
        clientFactory: () => _RecordingClient(
          (_) async => _jsonResponse(<String, dynamic>{
            'detail': 'service unavailable',
          }, statusCode: 503),
        ),
      );

      await expectLater(
        client.testConnection(),
        throwsA(
          isA<BackendClientException>()
              .having((error) => error.type, 'type', BackendFailureType.http)
              .having((error) => error.statusCode, 'statusCode', 503)
              .having((error) => error.detail, 'detail', 'service unavailable'),
        ),
      );
    });
  });

  group('analyze', () {
    late Directory temporaryDirectory;
    late File audioFile;

    setUp(() async {
      temporaryDirectory = await Directory.systemTemp.createTemp(
        'backend_client_test_',
      );
      audioFile = File(
        '${temporaryDirectory.path}${Platform.pathSeparator}sample.wav',
      );
      await audioFile.writeAsBytes(<int>[1, 2, 3, 4]);
    });

    tearDown(() async {
      await temporaryDirectory.delete(recursive: true);
    });

    test('uploads file and method and returns a typed response', () async {
      final client = BackendClient(
        baseUrl: 'http://localhost:8000',
        clientFactory: () => _RecordingClient((request) async {
          expect(request.method, 'POST');
          expect(request.url, Uri.parse('http://localhost:8000/analyze'));
          expect(request, isA<http.MultipartRequest>());
          final multipart = request as http.MultipartRequest;
          expect(multipart.fields, <String, String>{'method': 'yamnet'});
          expect(multipart.files, hasLength(1));
          expect(multipart.files.single.field, 'file');
          expect(multipart.files.single.filename, 'sample.wav');
          await request.finalize().drain<void>();
          return _jsonResponse(<String, dynamic>{
            'counts': <int>[5, 4, 6],
            'formatted': '5, 4, 6',
          });
        }),
      );

      final result = await client.analyze(file: audioFile, method: ' YAMNET ');

      expect(result.counts, <int>[5, 4, 6]);
      expect(result.formatted, '5, 4, 6');
      expect(() => result.counts.add(7), throwsA(isA<UnsupportedError>()));
    });

    test('rejects malformed JSON and invalid result fields', () async {
      final responses = <http.StreamedResponse>[
        _textResponse('not-json'),
        _jsonResponse(<String, dynamic>{
          'counts': <dynamic>[1, 2.5],
          'formatted': '1, 2.5',
        }),
        _jsonResponse(<String, dynamic>{
          'counts': <int>[1, 2],
          'formatted': 12,
        }),
      ];
      var responseIndex = 0;
      final client = BackendClient(
        baseUrl: 'http://localhost:8000',
        clientFactory: () => _RecordingClient((request) async {
          await request.finalize().drain<void>();
          return responses[responseIndex++];
        }),
      );

      for (var index = 0; index < responses.length; index++) {
        await expectLater(
          client.analyze(file: audioFile, method: 'rule'),
          throwsA(
            isA<BackendClientException>().having(
              (error) => error.type,
              'type',
              BackendFailureType.invalidResponse,
            ),
          ),
        );
      }
    });

    test('does not create a request for a missing or empty file', () async {
      var factoryCalls = 0;
      final client = BackendClient(
        baseUrl: 'http://localhost:8000',
        clientFactory: () {
          factoryCalls++;
          return _RecordingClient(
            (_) async => _jsonResponse(<String, dynamic>{}),
          );
        },
      );

      await expectLater(
        client.analyze(
          file: File('${temporaryDirectory.path}/missing.wav'),
          method: 'rule',
        ),
        throwsA(isA<BackendClientException>()),
      );
      await audioFile.writeAsBytes(<int>[]);
      await expectLater(
        client.analyze(file: audioFile, method: 'rule'),
        throwsA(isA<BackendClientException>()),
      );

      expect(factoryCalls, 0);
    });
  });

  test('times out, closes the request client, and never retries', () async {
    var factoryCalls = 0;
    late _BlockingClient requestClient;
    final client = BackendClient(
      baseUrl: 'http://localhost:8000',
      timeout: const Duration(milliseconds: 20),
      clientFactory: () {
        factoryCalls++;
        requestClient = _BlockingClient();
        return requestClient;
      },
    );

    await expectLater(
      client.testConnection(),
      throwsA(
        isA<BackendClientException>().having(
          (error) => error.type,
          'type',
          BackendFailureType.timeout,
        ),
      ),
    );

    expect(factoryCalls, 1);
    expect(requestClient.closeCount, 1);
    expect(client.activeRequestCount, 0);
  });

  test('cancelActiveRequests cancels and closes every active client', () async {
    final createdClients = <_BlockingClient>[];
    final client = BackendClient(
      baseUrl: 'http://localhost:8000',
      clientFactory: () {
        final requestClient = _BlockingClient();
        createdClients.add(requestClient);
        return requestClient;
      },
    );

    final first = client.testConnection();
    final second = client.testConnection();
    final firstExpectation = expectLater(
      first,
      throwsA(
        isA<BackendClientException>().having(
          (error) => error.type,
          'type',
          BackendFailureType.cancelled,
        ),
      ),
    );
    final secondExpectation = expectLater(
      second,
      throwsA(
        isA<BackendClientException>().having(
          (error) => error.type,
          'type',
          BackendFailureType.cancelled,
        ),
      ),
    );
    await Future<void>.delayed(Duration.zero);
    expect(client.activeRequestCount, 2);

    client.cancelActiveRequests();

    await firstExpectation;
    await secondExpectation;
    expect(createdClients.map((item) => item.closeCount), everyElement(1));
    expect(client.activeRequestCount, 0);
  });
}

class _RecordingClient extends http.BaseClient {
  _RecordingClient(this._handler);

  final Future<http.StreamedResponse> Function(http.BaseRequest request)
  _handler;
  int closeCount = 0;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) =>
      _handler(request);

  @override
  void close() {
    closeCount++;
  }
}

class _BlockingClient extends http.BaseClient {
  final Completer<http.StreamedResponse> _response =
      Completer<http.StreamedResponse>();
  int closeCount = 0;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) =>
      _response.future;

  @override
  void close() {
    closeCount++;
    if (!_response.isCompleted) {
      _response.completeError(http.ClientException('closed'));
    }
  }
}

http.StreamedResponse _jsonResponse(Object body, {int statusCode = 200}) =>
    _textResponse(jsonEncode(body), statusCode: statusCode);

http.StreamedResponse _textResponse(String body, {int statusCode = 200}) {
  return http.StreamedResponse(
    Stream<List<int>>.value(utf8.encode(body)),
    statusCode,
    headers: <String, String>{'content-type': 'application/json'},
  );
}
