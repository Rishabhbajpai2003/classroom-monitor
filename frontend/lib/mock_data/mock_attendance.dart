import 'dart:math';
import '../models/attendance.dart';
import 'mock_subjects.dart';

/// Mock attendance data for development and testing
class MockAttendance {
  static final _random = Random(42); // Fixed seed for consistent data

  /// Student name pools for generating realistic data
  static final _firstNames = [
    'Alex',
    'Emily',
    'Michael',
    'Jessica',
    'David',
    'Amanda',
    'Ryan',
    'Sophia',
    'Daniel',
    'Olivia',
    'James',
    'Emma',
    'William',
    'Isabella',
    'Benjamin',
    'Mia',
    'Lucas',
    'Charlotte',
    'Henry',
    'Amelia',
    'Alexander',
    'Harper',
    'Sebastian',
    'Evelyn',
    'Jack',
    'Abigail',
    'Aiden',
    'Emily',
    'Matthew',
    'Elizabeth',
    'Samuel',
    'Sofia',
    'Joseph',
    'Avery',
    'John',
    'Ella',
    'Owen',
    'Scarlett',
    'Dylan',
    'Grace',
    'Luke',
    'Chloe',
    'Gabriel',
    'Victoria',
    'Anthony',
    'Riley',
    'Isaac',
    'Aria',
    'Jayden',
    'Lily',
    'Levi',
    'Aurora'
  ];

  static final _lastNames = [
    'Thompson',
    'Chen',
    'Brown',
    'Williams',
    'Martinez',
    'Lee',
    'Taylor',
    'Garcia',
    'Kim',
    'Anderson',
    'Wilson',
    'Davis',
    'Johnson',
    'Smith',
    'Rodriguez',
    'Patel',
    'Nguyen',
    'Jackson',
    'White',
    'Harris',
    'Clark',
    'Lewis',
    'Robinson',
    'Walker',
    'Young',
    'Allen',
    'King',
    'Wright',
    'Scott',
    'Torres',
    'Hill',
    'Green',
    'Adams',
    'Baker',
    'Nelson',
    'Carter',
    'Mitchell',
    'Perez',
    'Roberts',
    'Turner',
    'Phillips',
    'Campbell',
    'Parker',
    'Evans',
    'Edwards',
    'Collins',
    'Stewart',
    'Sanchez',
    'Morris',
    'Rogers'
  ];

  /// Get attendance records for a specific subject and date
  static List<Attendance> getAttendanceForSubject(
      String subjectId, DateTime date) {
    // Get the subject to find the total students
    final subject = MockSubjects.getSubjectById(subjectId);
    final totalStudents = subject?.totalStudents ?? 40;

    // Use date and subjectId to create consistent but varied data
    final seed = subjectId.hashCode + date.day + date.month;
    final random = Random(seed);

    final records = <Attendance>[];

    for (int i = 0; i < totalStudents; i++) {
      final firstName = _firstNames[i % _firstNames.length];
      final lastName = _lastNames[(i + seed) % _lastNames.length];

      // Generate attendance status with realistic distribution
      // ~80% present, ~10% absent, ~5% late, ~5% excused
      final statusRoll = random.nextDouble();
      AttendanceStatus status;
      String? remarks;

      if (statusRoll < 0.80) {
        status = AttendanceStatus.present;
      } else if (statusRoll < 0.90) {
        status = AttendanceStatus.absent;
        remarks = _getAbsentRemark(random);
      } else if (statusRoll < 0.95) {
        status = AttendanceStatus.late;
        remarks = 'Arrived ${random.nextInt(15) + 5} minutes late';
      } else {
        status = AttendanceStatus.excused;
        remarks = _getExcusedRemark(random);
      }

      records.add(Attendance(
        id: 'att_${subjectId}_${date.millisecondsSinceEpoch}_$i',
        studentId: 'stu_${subjectId}_$i',
        studentName: '$firstName $lastName',
        rollNumber:
            '${subject?.code ?? "CS"}${2021 + (i ~/ 100)}${(i % 100).toString().padLeft(3, '0')}',
        status: status,
        date: date,
        subjectId: subjectId,
        remarks: remarks,
      ));
    }

    return records;
  }

  static String _getAbsentRemark(Random random) {
    final remarks = [
      'Medical leave',
      'Personal reasons',
      'No information',
      'Family emergency',
      'Transportation issues',
    ];
    return remarks[random.nextInt(remarks.length)];
  }

  static String _getExcusedRemark(Random random) {
    final remarks = [
      'Family emergency',
      'Medical appointment',
      'University event',
      'Prior approval obtained',
      'Religious observance',
    ];
    return remarks[random.nextInt(remarks.length)];
  }

  /// Get attendance summary for a subject
  static AttendanceSummary getSummaryForSubject(
      String subjectId, DateTime date) {
    final records = getAttendanceForSubject(subjectId, date);

    int present = 0;
    int absent = 0;
    int late = 0;
    int excused = 0;

    for (final record in records) {
      switch (record.status) {
        case AttendanceStatus.present:
          present++;
          break;
        case AttendanceStatus.absent:
          absent++;
          break;
        case AttendanceStatus.late:
          late++;
          break;
        case AttendanceStatus.excused:
          excused++;
          break;
      }
    }

    return AttendanceSummary(
      presentCount: present,
      absentCount: absent,
      lateCount: late,
      excusedCount: excused,
      totalStudents: records.length,
      date: date,
      subjectId: subjectId,
    );
  }

  /// Get attendance analytics for last N classes
  static List<AttendanceAnalytics> getAnalytics(String subjectId,
      {int count = 10}) {
    final analytics = <AttendanceAnalytics>[];
    final now = DateTime.now();
    final subject = MockSubjects.getSubjectById(subjectId);
    final totalStudents = subject?.totalStudents ?? 40;

    // Generate mock analytics data for last N classes
    for (int i = 0; i < count; i++) {
      final date = now.subtract(Duration(days: i * 2 + 1));
      // Generate varying attendance percentages (75-95%)
      final seed = subjectId.hashCode + date.day;
      final random = Random(seed);
      final percentage = 75.0 + random.nextDouble() * 20.0;
      final presentCount = (totalStudents * percentage / 100).round();

      analytics.add(AttendanceAnalytics(
        date: date,
        attendancePercentage: percentage,
        presentCount: presentCount,
        totalStudents: totalStudents,
      ));
    }

    return analytics.reversed.toList();
  }

  /// Get available class dates for a subject
  static List<DateTime> getClassDates(String subjectId) {
    final dates = <DateTime>[];
    final now = DateTime.now();

    // Generate last 10 class dates
    for (int i = 0; i < 10; i++) {
      dates.add(now.subtract(Duration(days: i * 2 + 1)));
    }

    return dates;
  }

  /// Get mock video metadata for a subject on a given date
  static Map<String, dynamic>? getVideoMetadata(
      String subjectId, DateTime date) {
    final subject = MockSubjects.getSubjectById(subjectId);
    if (subject == null) return null;

    final seed = subjectId.hashCode + date.day + date.month;
    final random = Random(seed);

    // 80% chance a video exists for a given date
    if (random.nextDouble() > 0.8) return null;

    final durationMin = subject.lastClassDuration ?? 60;
    final durationSec = durationMin * 60 + random.nextInt(120) - 60;
    final fileSizeMb =
        (durationSec * 0.4 + random.nextInt(50)).round(); // ~0.4 MB/sec
    final resolutions = ['1920x1080', '1280x720', '1920x1080'];
    final fps = [24, 30, 30];

    return {
      'videoId': 'vid_${subjectId}_${date.millisecondsSinceEpoch}',
      'filename': '${subject.code}_Lecture_${date.day}_${date.month}.mp4',
      'durationSeconds': durationSec,
      'fileSizeMb': fileSizeMb,
      'resolution': resolutions[random.nextInt(resolutions.length)],
      'fps': fps[random.nextInt(fps.length)],
      'uploadedAt': date.add(Duration(hours: random.nextInt(4) + 1)),
      'processingStatus': 'completed',
      'facesDetected': subject.totalStudents + random.nextInt(6) - 2,
      'studentsRecognized': subject.totalStudents - random.nextInt(5),
      'recordedBy': subject.instructorName,
    };
  }
}
