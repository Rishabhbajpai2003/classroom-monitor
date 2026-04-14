/// Attendance status enum
enum AttendanceStatus { present, absent, late, excused }

/// Individual student attendance record
class Attendance {
  final String id;
  final String studentId;
  final String studentName;
  final String rollNumber;
  final AttendanceStatus status;
  final DateTime date;
  final String subjectId;
  final String? remarks;

  const Attendance({
    required this.id,
    required this.studentId,
    required this.studentName,
    required this.rollNumber,
    required this.status,
    required this.date,
    required this.subjectId,
    this.remarks,
  });

  /// Get status display string
  String get statusDisplayName {
    switch (status) {
      case AttendanceStatus.present:
        return 'Present';
      case AttendanceStatus.absent:
        return 'Absent';
      case AttendanceStatus.late:
        return 'Late';
      case AttendanceStatus.excused:
        return 'Excused';
    }
  }

  /// Check if student was present (including late)
  bool get wasPresent => 
      status == AttendanceStatus.present || status == AttendanceStatus.late;
}

/// Attendance summary for a class or period
class AttendanceSummary {
  final int presentCount;
  final int absentCount;
  final int lateCount;
  final int excusedCount;
  final int totalStudents;
  final DateTime date;
  final String subjectId;

  const AttendanceSummary({
    required this.presentCount,
    required this.absentCount,
    this.lateCount = 0,
    this.excusedCount = 0,
    required this.totalStudents,
    required this.date,
    required this.subjectId,
  });

  /// Calculate attendance percentage
  double get attendancePercentage {
    if (totalStudents == 0) return 0.0;
    return ((presentCount + lateCount) / totalStudents) * 100;
  }

  /// Get formatted percentage string
  String get formattedPercentage => '${attendancePercentage.toStringAsFixed(1)}%';
}

/// Attendance analytics data for charts
class AttendanceAnalytics {
  final DateTime date;
  final double attendancePercentage;
  final int presentCount;
  final int totalStudents;

  const AttendanceAnalytics({
    required this.date,
    required this.attendancePercentage,
    required this.presentCount,
    required this.totalStudents,
  });
}
