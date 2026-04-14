import 'package:flutter/material.dart';

/// Subject model representing a course/class in the system.
/// Contains all metadata about the subject including instructor info and class statistics.
class Subject {
  final String id;
  final String name;
  final String code;
  final IconData icon;
  final String instructorName;
  final String instructorId;
  final DateTime? lastClassDate;
  final String? lastTopic;
  final int? lastClassDuration; // in minutes
  final double attendancePercentage;
  final int totalClasses;
  final int totalStudents;
  final String? description;

  const Subject({
    required this.id,
    required this.name,
    required this.code,
    required this.icon,
    required this.instructorName,
    required this.instructorId,
    this.lastClassDate,
    this.lastTopic,
    this.lastClassDuration,
    this.attendancePercentage = 0.0,
    this.totalClasses = 0,
    this.totalStudents = 0,
    this.description,
  });

  /// Get formatted duration string
  String get formattedDuration {
    if (lastClassDuration == null) return 'N/A';
    final hours = lastClassDuration! ~/ 60;
    final minutes = lastClassDuration! % 60;
    if (hours > 0) {
      return '${hours}h ${minutes}m';
    }
    return '$minutes minutes';
  }

  /// Get attendance percentage as formatted string
  String get formattedAttendance => '${attendancePercentage.toStringAsFixed(1)}%';

  /// Create a copy with modified fields
  Subject copyWith({
    String? id,
    String? name,
    String? code,
    IconData? icon,
    String? instructorName,
    String? instructorId,
    DateTime? lastClassDate,
    String? lastTopic,
    int? lastClassDuration,
    double? attendancePercentage,
    int? totalClasses,
    int? totalStudents,
    String? description,
  }) {
    return Subject(
      id: id ?? this.id,
      name: name ?? this.name,
      code: code ?? this.code,
      icon: icon ?? this.icon,
      instructorName: instructorName ?? this.instructorName,
      instructorId: instructorId ?? this.instructorId,
      lastClassDate: lastClassDate ?? this.lastClassDate,
      lastTopic: lastTopic ?? this.lastTopic,
      lastClassDuration: lastClassDuration ?? this.lastClassDuration,
      attendancePercentage: attendancePercentage ?? this.attendancePercentage,
      totalClasses: totalClasses ?? this.totalClasses,
      totalStudents: totalStudents ?? this.totalStudents,
      description: description ?? this.description,
    );
  }
}
