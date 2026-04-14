import 'dart:convert';
import 'dart:html' as html;
import 'package:http/http.dart' as http;
import 'api_service.dart';
import 'auth_service.dart';

/// Holds live attendance + activity stats parsed from the model's CSV outputs.
class LiveAttendanceData {
  final String date;
  final String cameraId;
  final int totalStudents;
  final int presentCount;
  final int absentCount;
  final double averageAttention; // 0-100
  final List<StudentRecord> records;

  // Activity summary
  final int sittingCount;
  final int standingCount;
  final int walkingCount;
  final int phoneUsageCount;

  LiveAttendanceData({
    required this.date,
    required this.cameraId,
    required this.totalStudents,
    required this.presentCount,
    required this.absentCount,
    required this.averageAttention,
    required this.records,
    this.sittingCount = 0,
    this.standingCount = 0,
    this.walkingCount = 0,
    this.phoneUsageCount = 0,
  });

  int get lateCount => 0;
  double get attendancePercentage =>
      totalStudents > 0 ? (presentCount / totalStudents) * 100 : 0;

  Map<String, dynamic> toJson() => {
        'date': date,
        'cameraId': cameraId,
        'totalStudents': totalStudents,
        'presentCount': presentCount,
        'absentCount': absentCount,
        'averageAttention': averageAttention,
        'records': records.map((r) => r.toJson()).toList(),
        'sittingCount': sittingCount,
        'standingCount': standingCount,
        'walkingCount': walkingCount,
        'phoneUsageCount': phoneUsageCount,
      };

  factory LiveAttendanceData.fromJson(Map<String, dynamic> json) {
    return LiveAttendanceData(
      date: json['date']?.toString() ?? '',
      cameraId: json['cameraId']?.toString() ?? '',
      totalStudents: (json['totalStudents'] as num?)?.toInt() ?? 0,
      presentCount: (json['presentCount'] as num?)?.toInt() ?? 0,
      absentCount: (json['absentCount'] as num?)?.toInt() ?? 0,
      averageAttention: (json['averageAttention'] as num?)?.toDouble() ?? 0.0,
      records: (json['records'] as List?)
              ?.map((r) => StudentRecord.fromJson(r as Map<String, dynamic>))
              .toList() ??
          [],
      sittingCount: (json['sittingCount'] as num?)?.toInt() ?? 0,
      standingCount: (json['standingCount'] as num?)?.toInt() ?? 0,
      walkingCount: (json['walkingCount'] as num?)?.toInt() ?? 0,
      phoneUsageCount: (json['phoneUsageCount'] as num?)?.toInt() ?? 0,
    );
  }
}

class StudentRecord {
  final String studentId;
  final String name;
  final bool isPresent;
  final double attentionScore; // 0-100
  final String cameraId;

  const StudentRecord({
    required this.studentId,
    required this.name,
    required this.isPresent,
    required this.attentionScore,
    required this.cameraId,
  });

  Map<String, dynamic> toJson() => {
        'studentId': studentId,
        'name': name,
        'isPresent': isPresent,
        'attentionScore': attentionScore,
        'cameraId': cameraId,
      };

  factory StudentRecord.fromJson(Map<String, dynamic> json) {
    return StudentRecord(
      studentId: json['studentId'] as String,
      name: json['name'] as String,
      isPresent: json['isPresent'] as bool,
      attentionScore: (json['attentionScore'] as num).toDouble(),
      cameraId: json['cameraId'] as String,
    );
  }
}

/// Singleton service that stores the most-recent analysis results
/// and makes them available to the dashboard / attendance screens.
class AnalysisDataService {
  static final AnalysisDataService _instance = AnalysisDataService._internal();
  factory AnalysisDataService() => _instance;
  AnalysisDataService._internal() {
    _loadFromStorage();
  }

  // Most recent live data — null until the first successful analysis
  LiveAttendanceData? latestData;
  GradioAnalysisResult? latestResult;

  static const _storageKeyBase = 'classroom_monitor_live_data';
  static const _resultStorageKeyBase = 'classroom_monitor_latest_result';

  String get _storageKey => AuthService().scopedKey(_storageKeyBase);
  String get _resultStorageKey => AuthService().scopedKey(_resultStorageKeyBase);

  void _loadFromStorage() {
    try {
      final jsonStr = html.window.localStorage[_storageKey];
      if (jsonStr != null && jsonStr.isNotEmpty) {
        final map = json.decode(jsonStr) as Map<String, dynamic>;
        latestData = LiveAttendanceData.fromJson(map);
      }
    } catch (e) {
      print('Failed to load live data from storage: $e');
    }
    
    try {
      final resultStr = html.window.localStorage[_resultStorageKey];
      if (resultStr != null && resultStr.isNotEmpty) {
        final map = json.decode(resultStr) as Map<String, dynamic>;
        latestResult = GradioAnalysisResult.fromCacheJson(map);
      }
    } catch (e) {
      print('Failed to load latest result from storage: $e');
    }
  }

  void _saveToStorage(LiveAttendanceData data, GradioAnalysisResult? result) {
    try {
      final jsonStr = json.encode(data.toJson());
      html.window.localStorage[_storageKey] = jsonStr;
      
      if (result != null) {
        final resultStr = json.encode(result.toJson());
        html.window.localStorage[_resultStorageKey] = resultStr;
      }
    } catch (e) {
      print('Failed to save data to storage: $e');
    }
  }

  // Listeners to notify when data updates
  final List<void Function(LiveAttendanceData)> _listeners = [];

  void addListener(void Function(LiveAttendanceData) listener) {
    _listeners.add(listener);
  }

  void removeListener(void Function(LiveAttendanceData) listener) {
    _listeners.remove(listener);
  }

  void _notify(LiveAttendanceData data) {
    for (final l in _listeners) {
      l(data);
    }
  }

  /// Download and parse both CSVs from a completed [GradioAnalysisResult].
  /// Updates [latestData] and notifies all listeners.
  Future<void> parseFromResult(GradioAnalysisResult result) async {
    final List<StudentRecord> records = [];
    int presentCount = 0;
    double totalAttention = 0;
    int attentionRows = 0;
    int sittingCount = 0;
    int standingCount = 0;
    int walkingCount = 0;
    int phoneUsageCount = 0;
    String cameraId = 'cam_01';

    // ── 1. Parse attendance_report.csv ───────────────────────────────────
    if (result.attendanceCsvUrl != null) {
      try {
        final csvText = await _fetchText(result.attendanceCsvUrl!);
        final parsed = _parseAttendanceCsv(csvText);
        records.addAll(parsed.records);
        presentCount = parsed.presentCount;
        totalAttention = parsed.totalAttention;
        attentionRows = parsed.attentionRows;
        phoneUsageCount = parsed.phoneUsageCount;
        cameraId = parsed.cameraId;
      } catch (e) {
        print('Error parsing attendance CSV: $e');
        // If download fails, continue with what we have
      }
    }

    // ── 2. Parse person_activity_summary.csv ─────────────────────────────
    if (result.activityCsvUrl != null) {
      try {
        final csvText = await _fetchText(result.activityCsvUrl!);
        final actSummary = _parseActivityCsv(csvText);
        sittingCount = actSummary['sitting'] ?? 0;
        standingCount = actSummary['standing'] ?? 0;
        walkingCount = actSummary['walking'] ?? 0;
      } catch (e) {
        print('Error parsing activity CSV: $e');
      }
    }

    final total = records.isNotEmpty ? records.length : presentCount;
    final absentCount = total - presentCount;
    final avgAttention =
        attentionRows > 0 ? totalAttention / attentionRows : 0.0;

    // Extract date from timestamp (YYYY-MM-DD HH:MM:SS)
    String runDate = DateTime.now().toIso8601String().split('T')[0];
    if (result.timestamp != null && result.timestamp!.length >= 10) {
      runDate = result.timestamp!.substring(0, 10);
    }

    final data = LiveAttendanceData(
      date: runDate,
      cameraId: cameraId,
      totalStudents: total,
      presentCount: presentCount,
      absentCount: absentCount < 0 ? 0 : absentCount,
      averageAttention: avgAttention,
      records: records,
      sittingCount: sittingCount,
      standingCount: standingCount,
      walkingCount: walkingCount,
      phoneUsageCount: phoneUsageCount,
    );

    latestData = data;
    latestResult = result;
    _saveToStorage(data, result);
    _notify(data);
  }

  // ── CSV parsers ───────────────────────────────────────────────────────────

  _AttendanceParse _parseAttendanceCsv(String csv) {
    final lines = csv.trim().split('\n');
    if (lines.isEmpty) return _AttendanceParse.empty();

    // Find column indices from header
    // Real CSV columns (from classroom-monitor model):
    // Session_ID, Camera_ID, Global_Student_ID, Local_Track_ID, Total_Frames,
    // Presence_Time_Seconds, Present, Attentive_Frames, Distracted_Frames,
    // HandRaise_Frames, UsingPhone_Frames, Confidence_Weighted_Attention_Score, Attention_Percentage
    final header = _splitCsv(lines[0]);
    int idxId = _findCol(header, ['global_student_id', 'student_id', 'id', 'face_id']);
    int idxName = _findCol(header, ['name', 'student_name', 'label']);
    int idxStatus = _findCol(header, ['present', 'status', 'attendance', 'state']);
    // Prioritize Attention_Percentage (0-100) over the raw weighted score
    int idxAttention = _findCol(header, ['attention_percentage', 'attention_pct']);
    if (idxAttention < 0) {
      idxAttention = _findCol(header, ['attention_score', 'attention']);
    }
    int idxPhone = _findCol(header, ['usingphone_frames', 'phone_frames', 'phone_usage', 'phone']);
    int idxCamera = _findCol(header, ['camera_id', 'camera', 'cam']);

    final records = <StudentRecord>[];
    int presentCount = 0;
    double totalAttention = 0;
    int attentionRows = 0;
    int phoneUsageCount = 0;
    String cameraId = 'cam_01';

    for (int i = 1; i < lines.length; i++) {
      final cols = _splitCsv(lines[i]);
      if (cols.isEmpty) continue;

      final id = idxId >= 0 && idxId < cols.length ? cols[idxId] : 'S$i';
      // Use the global student ID as display name if no name column
      final name = idxName >= 0 && idxName < cols.length
          ? cols[idxName]
          : id; // show STU_001 etc. instead of generic "Student 1"
      final statusStr = idxStatus >= 0 && idxStatus < cols.length
          ? cols[idxStatus].toLowerCase().trim()
          : '';
      // Model outputs "Yes"/"No" for the Present column
      final isPresent = statusStr == 'yes' ||
          statusStr == 'present' ||
          statusStr == 'true' ||
          statusStr == '1';

      final attnStr = idxAttention >= 0 && idxAttention < cols.length
          ? cols[idxAttention]
          : '';
      final rawAttention = double.tryParse(attnStr) ?? 0.0;
      // Attention_Percentage is already 0-100; cap at 100 to handle edge cases
      final attentionPct = rawAttention.clamp(0.0, 100.0);

      if (idxCamera >= 0 && idxCamera < cols.length) {
        cameraId = cols[idxCamera];
      }

      final phoneFrames = idxPhone >= 0 && idxPhone < cols.length
          ? int.tryParse(cols[idxPhone]) ?? 0
          : 0;

      if (isPresent) presentCount++;
      if (attentionPct > 0) {
        totalAttention += attentionPct;
        attentionRows++;
      }
      if (phoneFrames > 0) {
        phoneUsageCount++;
      }

      records.add(StudentRecord(
        studentId: id,
        name: name,
        isPresent: isPresent,
        attentionScore: attentionPct,
        cameraId: cameraId,
      ));
    }

    return _AttendanceParse(
      records: records,
      presentCount: presentCount,
      totalAttention: totalAttention,
      attentionRows: attentionRows,
      phoneUsageCount: phoneUsageCount,
      cameraId: cameraId,
    );
  }

  Map<String, int> _parseActivityCsv(String csv) {
    final lines = csv.trim().split('\n');
    if (lines.isEmpty) return {};

    final header = _splitCsv(lines[0]);
    int idxActivity = _findCol(header, ['activity', 'action', 'label', 'class']);
    int idxCount = _findCol(header, ['count', 'total', 'frames', 'n']);

    final counts = <String, int>{};
    for (int i = 1; i < lines.length; i++) {
      final cols = _splitCsv(lines[i]);
      if (cols.isEmpty) continue;
      final act = idxActivity >= 0 && idxActivity < cols.length
          ? cols[idxActivity].toLowerCase().trim()
          : '';
      final cnt = idxCount >= 0 && idxCount < cols.length
          ? int.tryParse(cols[idxCount]) ?? 1
          : 1;
      counts[act] = (counts[act] ?? 0) + cnt;
    }
    return counts;
  }

  // ── Utilities ─────────────────────────────────────────────────────────────

  List<String> _splitCsv(String line) {
    final result = <String>[];
    final buffer = StringBuffer();
    bool inQuotes = false;
    for (int i = 0; i < line.length; i++) {
      final ch = line[i];
      if (ch == '"') {
        inQuotes = !inQuotes;
      } else if (ch == ',' && !inQuotes) {
        result.add(buffer.toString().trim());
        buffer.clear();
      } else {
        buffer.write(ch);
      }
    }
    result.add(buffer.toString().trim());
    return result;
  }

  int _findCol(List<String> header, List<String> candidates) {
    for (final candidate in candidates) {
      for (int i = 0; i < header.length; i++) {
        if (header[i].toLowerCase().trim() == candidate) return i;
      }
    }
    // Partial match fallback
    for (final candidate in candidates) {
      for (int i = 0; i < header.length; i++) {
        if (header[i].toLowerCase().trim().contains(candidate)) return i;
      }
    }
    return -1;
  }

  Future<String> _fetchText(String url) async {
    final response =
        await http.get(Uri.parse(url)).timeout(const Duration(seconds: 30));
    if (response.statusCode != 200) {
      throw Exception('Failed to download CSV: ${response.statusCode}');
    }
    return utf8.decode(response.bodyBytes);
  }
}

class _AttendanceParse {
  final List<StudentRecord> records;
  final int presentCount;
  final double totalAttention;
  final int attentionRows;
  final int phoneUsageCount;
  final String cameraId;

  const _AttendanceParse({
    required this.records,
    required this.presentCount,
    required this.totalAttention,
    required this.attentionRows,
    required this.phoneUsageCount,
    required this.cameraId,
  });

  factory _AttendanceParse.empty() => const _AttendanceParse(
        records: [],
        presentCount: 0,
        totalAttention: 0,
        attentionRows: 0,
        phoneUsageCount: 0,
        cameraId: 'cam_01',
      );
}
