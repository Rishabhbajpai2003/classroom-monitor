import 'dart:html' as html;

import 'package:http/http.dart' as http;

import 'api_service.dart';

/// Stores the runtime backend URL for the Flutter web app.
class UrlConfigService {
  static const _storageKey = 'classroom_monitor_api_url';
  static const _defaultUrl = 'http://localhost:8000';

  static final UrlConfigService _instance = UrlConfigService._internal();

  factory UrlConfigService() => _instance;

  UrlConfigService._internal();

  String get currentUrl => ApiService.baseUrl;

  bool get isConfigured {
    final saved = html.window.localStorage[_storageKey];
    return saved != null && saved.trim().isNotEmpty;
  }

  void loadSavedUrl() {
    final saved = html.window.localStorage[_storageKey];
    if (saved != null && saved.trim().isNotEmpty) {
      ApiService.baseUrl = _normalize(saved);
    } else {
      ApiService.baseUrl = _defaultUrl;
    }
  }

  void saveUrl(String url) {
    final normalized = _normalize(url);
    html.window.localStorage[_storageKey] = normalized;
    ApiService.baseUrl = normalized;
  }

  void clearUrl() {
    html.window.localStorage.remove(_storageKey);
    ApiService.baseUrl = _defaultUrl;
  }

  Future<UrlValidationResult> validate(String url) async {
    final normalized = _normalize(url);
    final uri = Uri.tryParse(normalized);

    if (uri == null || (!uri.isScheme('http') && !uri.isScheme('https'))) {
      return const UrlValidationResult(
        isValid: false,
        isOnline: false,
        message: 'URL must start with http:// or https://',
      );
    }

    try {
      final response = await http
          .get(Uri.parse('$normalized/health'))
          .timeout(const Duration(seconds: 8));

      if (response.statusCode == 200) {
        return const UrlValidationResult(
          isValid: true,
          isOnline: true,
          message: 'Connected! Backend health check passed.',
        );
      }

      return UrlValidationResult(
        isValid: true,
        isOnline: false,
        message:
            'Server responded with status ${response.statusCode}. Check that this is the classroom-monitor backend.',
      );
    } on Exception catch (error) {
      return UrlValidationResult(
        isValid: true,
        isOnline: false,
        message: 'Could not reach backend: ${_friendly(error)}',
      );
    }
  }

  String _normalize(String url) {
    var value = url.trim();
    if (value.endsWith('/')) {
      value = value.substring(0, value.length - 1);
    }
    return value;
  }

  String _friendly(Exception error) {
    final text = error.toString();
    if (text.contains('timeout')) {
      return 'connection timed out after 8 seconds';
    }
    if (text.contains('refused')) {
      return 'connection refused';
    }
    if (text.contains('SocketException')) {
      return 'server unreachable';
    }
    return text.replaceFirst('Exception: ', '');
  }
}

class UrlValidationResult {
  final bool isValid;
  final bool isOnline;
  final String message;

  const UrlValidationResult({
    required this.isValid,
    required this.isOnline,
    required this.message,
  });
}
