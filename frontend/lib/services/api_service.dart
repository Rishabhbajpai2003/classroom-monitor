import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

/// API service for communicating with the classroom-monitor FastAPI backend.
class ApiService {
  static String _baseUrl = 'http://localhost:8000';

  static String get baseUrl => _baseUrl;

  static set baseUrl(String value) {
    if (value.endsWith('/')) {
      _baseUrl = value.substring(0, value.length - 1);
    } else {
      _baseUrl = value;
    }
  }

  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  Future<bool> healthCheck() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 6));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<List<String>> getAvailableDates({String? subject}) async {
    try {
      final query = subject != null ? '?subject=${Uri.encodeQueryComponent(subject)}' : '';
      final uri = Uri.parse('$baseUrl/api/dates$query');
      final response = await http.get(uri).timeout(const Duration(seconds: 10));
      if (response.statusCode != 200) {
        return [];
      }
      final data = json.decode(response.body) as List<dynamic>;
      return data.map((item) => item.toString()).toList();
    } catch (e) {
      print('Error fetching available dates: $e');
      return [];
    }
  }

  Future<AnalysisJobStatus> getAnalysisJobStatus(String jobId) async {
    final uri = Uri.parse('$baseUrl/api/analysis/jobs/$jobId');
    final response = await http.get(uri).timeout(const Duration(seconds: 15));
    if (response.statusCode != 200) {
      throw Exception('Failed to fetch analysis job: ${response.statusCode}');
    }
    final jsonMap = json.decode(response.body) as Map<String, dynamic>;
    return AnalysisJobStatus.fromJson(jsonMap);
  }

  Future<Uint8List> downloadFile(String serverPath) async {
    final encodedPath = Uri.encodeQueryComponent(serverPath);
    final uri = Uri.parse('$baseUrl/api/files?path=$encodedPath');
    final response = await http.get(uri).timeout(const Duration(minutes: 5));
    if (response.statusCode != 200) {
      throw Exception('Download failed: ${response.statusCode}');
    }
    return response.bodyBytes;
  }

  Future<String> fetchTextFile(String serverPath) async {
    final bytes = await downloadFile(serverPath);
    return utf8.decode(bytes);
  }

  Future<String> askRagQuestion(String query, {String? courseId}) async {
    try {
      final uri = Uri.parse('$baseUrl/api/query-rag');
      final response = await http
          .post(uri, body: {
            'query': query,
            'course_id': courseId ?? 'global',
          })
          .timeout(const Duration(seconds: 30));

      if (response.statusCode != 200) {
        throw Exception('RAG query failed (${response.statusCode})');
      }

      final jsonMap = json.decode(response.body) as Map<String, dynamic>;
      return jsonMap['answer']?.toString() ?? 'No response received.';
    } catch (e) {
      print('RAG Error: $e');
      return _getMockRagResponse(query);
    }
  }

  String _getMockRagResponse(String query) {
    final lowerQuery = query.toLowerCase();
    if (lowerQuery.contains('cnn') || lowerQuery.contains('convolutional')) {
      return 'A Convolutional Neural Network (CNN) learns spatial features using shared filters and is widely used for images.';
    } else if (lowerQuery.contains('backpropagation')) {
      return 'Backpropagation computes gradients of the loss with respect to model weights using the chain rule.';
    } else if (lowerQuery.contains('overfitting')) {
      return 'Overfitting happens when a model memorizes training noise and performs worse on unseen data.';
    }
    return 'The study assistant endpoint is available, and this is a fallback response while the full RAG stack is being wired up.';
  }

  Future<List<GradioAnalysisResult>> getAttendanceRuns(String date, {String? subjectName}) async {
    try {
      final query = subjectName != null ? '?subject=${Uri.encodeQueryComponent(subjectName)}' : '';
      final uri = Uri.parse('$baseUrl/api/attendance/$date$query');
      final response = await http.get(uri).timeout(const Duration(seconds: 10));

      if (response.statusCode == 404) {
        return [];
      }
      if (response.statusCode != 200) {
        throw Exception('Server error: ${response.statusCode}');
      }

      final data = json.decode(response.body) as List<dynamic>;
      return data
          .map((run) => GradioAnalysisResult.fromApiJson(run as Map<String, dynamic>, baseUrl))
          .toList();
    } catch (e) {
      print('getAttendanceRuns error: $e');
      return [];
    }
  }

  Future<AttendanceSummaryResponse> getAttendance(String date, {String subjectId = 'default', String? subjectName}) async {
    final runs = await getAttendanceRuns(date, subjectName: subjectName);
    if (runs.isEmpty) {
      throw Exception('No analysis data found for $date');
    }

    final firstRun = runs.first;
    final csvText = await fetchTextFile(firstRun.attendanceCsvPath!);
    return _parseAttendanceCsvText(csvText, date, subjectId);
  }

  AttendanceSummaryResponse _parseAttendanceCsvText(String csv, String date, String subjectId) {
    final lines = csv.trim().split('\n');
    if (lines.length < 2) {
      return AttendanceSummaryResponse(
        date: date,
        subjectId: subjectId,
        totalStudents: 0,
        presentCount: 0,
        absentCount: 0,
        averageAttention: 0,
        phoneUsageCount: 0,
        records: [],
      );
    }

    final header = lines[0].toLowerCase().split(',');
    int idxPresent = header.indexOf('present');
    int idxAttention = header.indexOf('attention_percentage');
    if (idxAttention < 0) idxAttention = header.indexOf('attention_score');
    final idxPhone = header.indexOf('usingphone_frames');

    int present = 0;
    double sumAttention = 0;
    int phoneUsers = 0;
    final total = lines.length - 1;

    for (int i = 1; i < lines.length; i++) {
      final cols = lines[i].split(',');
      if (idxPresent >= 0 && idxPresent < cols.length) {
        final presentValue = cols[idxPresent].toLowerCase().trim();
        if (presentValue == 'yes' || presentValue == 'present' || presentValue == 'true' || presentValue == '1') {
          present++;
        }
      }
      if (idxAttention >= 0 && idxAttention < cols.length) {
        sumAttention += double.tryParse(cols[idxAttention]) ?? 0;
      }
      if (idxPhone >= 0 && idxPhone < cols.length) {
        final phoneFrames = int.tryParse(cols[idxPhone]) ?? 0;
        if (phoneFrames > 0) phoneUsers++;
      }
    }

    return AttendanceSummaryResponse(
      date: date,
      subjectId: subjectId,
      totalStudents: total,
      presentCount: present,
      absentCount: total - present,
      averageAttention: total > 0 ? sumAttention / total : 0,
      phoneUsageCount: phoneUsers,
      records: [],
    );
  }
}

class AnalysisJobStatus {
  final String jobId;
  final String status;
  final double progress;
  final String currentStep;
  final String? errorMessage;
  final GradioAnalysisResult? result;

  const AnalysisJobStatus({
    required this.jobId,
    required this.status,
    required this.progress,
    required this.currentStep,
    this.errorMessage,
    this.result,
  });

  bool get isCompleted => status == 'completed';
  bool get isFailed => status == 'failed';

  factory AnalysisJobStatus.fromJson(Map<String, dynamic> json) {
    final resultJson = json['result'];
    return AnalysisJobStatus(
      jobId: json['job_id']?.toString() ?? '',
      status: json['status']?.toString() ?? 'unknown',
      progress: (json['progress'] as num?)?.toDouble() ?? 0.0,
      currentStep: json['current_step']?.toString() ?? '',
      errorMessage: json['error_message']?.toString(),
      result: resultJson is Map<String, dynamic>
          ? GradioAnalysisResult.fromApiJson(resultJson, ApiService.baseUrl)
          : null,
    );
  }
}

class GradioAnalysisResult {
  final String? attentionVideoPath;
  final String? attendanceCsvPath;
  final String? activityVideoPath;
  final String? activityCsvPath;
  final String? speechVideoPath;
  final String? speechCsvPath;
  final String? seatMapPngPath;
  final String? seatMapJsonPath;
  final String? seatingTimelinePath;
  final String? attendanceEventsPath;
  final String logText;
  final String baseUrl;
  final String? logTextPath;
  final String? runId;
  final String? topic;
  final String? timestamp;

  GradioAnalysisResult({
    this.attentionVideoPath,
    this.attendanceCsvPath,
    this.activityVideoPath,
    this.activityCsvPath,
    this.speechVideoPath,
    this.speechCsvPath,
    this.seatMapPngPath,
    this.seatMapJsonPath,
    this.seatingTimelinePath,
    this.attendanceEventsPath,
    required this.logText,
    required this.baseUrl,
    this.logTextPath,
    this.runId,
    this.topic,
    this.timestamp,
  });

  factory GradioAnalysisResult.fromApiJson(Map<String, dynamic> json, String baseUrl) {
    return GradioAnalysisResult(
      attentionVideoPath: json['attentionVideoPath']?.toString(),
      attendanceCsvPath: json['attendanceCsvPath']?.toString(),
      activityVideoPath: json['activityVideoPath']?.toString(),
      activityCsvPath: json['activityCsvPath']?.toString(),
      speechVideoPath: json['speechVideoPath']?.toString(),
      speechCsvPath: json['speechCsvPath']?.toString(),
      seatMapPngPath: json['seatMapPngPath']?.toString(),
      seatMapJsonPath: json['seatMapJsonPath']?.toString(),
      seatingTimelinePath: json['seatingTimelinePath']?.toString(),
      attendanceEventsPath: json['attendanceEventsPath']?.toString(),
      logText: json['logText']?.toString() ?? '',
      baseUrl: baseUrl,
      logTextPath: json['logTextPath']?.toString(),
      runId: json['run_id']?.toString() ?? json['runId']?.toString(),
      topic: json['topic']?.toString(),
      timestamp: json['timestamp']?.toString(),
    );
  }

  factory GradioAnalysisResult.fromCacheJson(Map<String, dynamic> json) {
    return GradioAnalysisResult.fromApiJson(json, json['baseUrl'] as String? ?? ApiService.baseUrl);
  }

  Map<String, dynamic> toJson() => {
        'attentionVideoPath': attentionVideoPath,
        'attendanceCsvPath': attendanceCsvPath,
        'activityVideoPath': activityVideoPath,
        'activityCsvPath': activityCsvPath,
        'speechVideoPath': speechVideoPath,
        'speechCsvPath': speechCsvPath,
        'seatMapPngPath': seatMapPngPath,
        'seatMapJsonPath': seatMapJsonPath,
        'seatingTimelinePath': seatingTimelinePath,
        'attendanceEventsPath': attendanceEventsPath,
        'logText': logText,
        'logTextPath': logTextPath,
        'baseUrl': baseUrl,
        'run_id': runId,
        'topic': topic,
        'timestamp': timestamp,
      };

  bool get hasAttentionVideo => attentionVideoPath != null;
  bool get hasAttendanceCsv => attendanceCsvPath != null;
  bool get hasActivityVideo => activityVideoPath != null;
  bool get hasActivityCsv => activityCsvPath != null;
  bool get hasSpeechVideo => speechVideoPath != null;
  bool get hasSpeechCsv => speechCsvPath != null;
  bool get hasSeatMapPng => seatMapPngPath != null;
  bool get hasSeatMapJson => seatMapJsonPath != null;
  bool get hasSeatingTimeline => seatingTimelinePath != null;
  bool get hasAttendanceEvents => attendanceEventsPath != null;

  String? get attentionVideoUrl => _resolveFileUrl(attentionVideoPath);
  String? get attendanceCsvUrl => _resolveFileUrl(attendanceCsvPath);
  String? get activityVideoUrl => _resolveFileUrl(activityVideoPath);
  String? get activityCsvUrl => _resolveFileUrl(activityCsvPath);
  String? get speechVideoUrl => _resolveFileUrl(speechVideoPath);
  String? get speechCsvUrl => _resolveFileUrl(speechCsvPath);
  String? get seatMapPngUrl => _resolveFileUrl(seatMapPngPath);
  String? get seatMapJsonUrl => _resolveFileUrl(seatMapJsonPath);
  String? get seatingTimelineUrl => _resolveFileUrl(seatingTimelinePath);
  String? get attendanceEventsUrl => _resolveFileUrl(attendanceEventsPath);

  String? _resolveFileUrl(String? path) {
    if (path == null || path.isEmpty) return null;
    return '$baseUrl/api/files?path=${Uri.encodeQueryComponent(path)}';
  }

  String? get pipelineLogUrl {
    if (logTextPath != null) {
      return _resolveFileUrl(logTextPath);
    }
    return null;
  }

  bool get isSuccess => logText.toLowerCase().contains('pipeline complete');
}

class AttendanceSummaryResponse {
  final String date;
  final String subjectId;
  final int totalStudents;
  final int presentCount;
  final int absentCount;
  final double averageAttention;
  final int phoneUsageCount;
  final List<dynamic> records;

  AttendanceSummaryResponse({
    required this.date,
    required this.subjectId,
    required this.totalStudents,
    required this.presentCount,
    required this.absentCount,
    required this.averageAttention,
    required this.phoneUsageCount,
    required this.records,
  });

  factory AttendanceSummaryResponse.fromJson(Map<String, dynamic> json) {
    return AttendanceSummaryResponse(
      date: json['date']?.toString() ?? '',
      subjectId: json['subject_id']?.toString() ?? '',
      totalStudents: (json['total_students'] as num?)?.toInt() ?? 0,
      presentCount: (json['present_count'] as num?)?.toInt() ?? 0,
      absentCount: (json['absent_count'] as num?)?.toInt() ?? 0,
      averageAttention: (json['average_attention'] as num?)?.toDouble() ?? 0.0,
      phoneUsageCount: (json['phone_usage_count'] as num?)?.toInt() ?? 0,
      records: (json['records'] as List?) ?? [],
    );
  }
}
