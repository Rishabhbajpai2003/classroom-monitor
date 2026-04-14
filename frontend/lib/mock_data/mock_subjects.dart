import 'package:flutter/material.dart';
import '../models/subject.dart';

/// Mock subjects data for development and testing
class MockSubjects {
  static List<Subject> getSubjects() {
    return [
      Subject(
        id: 'sub_001',
        name: 'Data Structures & Algorithms',
        code: 'CS201',
        icon: Icons.account_tree_rounded,
        instructorName: 'Dr. Sarah Johnson',
        instructorId: 'usr_001',
        lastClassDate: DateTime.now().subtract(const Duration(days: 1)),
        lastTopic: 'Binary Search Trees',
        lastClassDuration: 90,
        attendancePercentage: 87.5,
        totalClasses: 24,
        totalStudents: 45,
        description: 'Study of fundamental data structures and algorithm design techniques.',
      ),
      Subject(
        id: 'sub_002',
        name: 'Machine Learning',
        code: 'CS401',
        icon: Icons.psychology_rounded,
        instructorName: 'Prof. Robert Lee',
        instructorId: 'usr_003',
        lastClassDate: DateTime.now().subtract(const Duration(days: 2)),
        lastTopic: 'Neural Networks Introduction',
        lastClassDuration: 120,
        attendancePercentage: 92.3,
        totalClasses: 18,
        totalStudents: 38,
        description: 'Introduction to machine learning algorithms and applications.',
      ),
      Subject(
        id: 'sub_003',
        name: 'Database Systems',
        code: 'CS301',
        icon: Icons.storage_rounded,
        instructorName: 'Dr. Maria Garcia',
        instructorId: 'usr_004',
        lastClassDate: DateTime.now().subtract(const Duration(days: 3)),
        lastTopic: 'SQL Optimization',
        lastClassDuration: 75,
        attendancePercentage: 78.9,
        totalClasses: 20,
        totalStudents: 52,
        description: 'Design and implementation of database management systems.',
      ),
      Subject(
        id: 'sub_004',
        name: 'Computer Networks',
        code: 'CS302',
        icon: Icons.hub_rounded,
        instructorName: 'Dr. James Wilson',
        instructorId: 'usr_005',
        lastClassDate: DateTime.now().subtract(const Duration(days: 1)),
        lastTopic: 'TCP/IP Protocol Stack',
        lastClassDuration: 90,
        attendancePercentage: 84.2,
        totalClasses: 22,
        totalStudents: 40,
        description: 'Study of computer network architectures and protocols.',
      ),
      Subject(
        id: 'sub_005',
        name: 'Software Engineering',
        code: 'CS303',
        icon: Icons.engineering_rounded,
        instructorName: 'Prof. Emily Davis',
        instructorId: 'usr_006',
        lastClassDate: DateTime.now().subtract(const Duration(days: 4)),
        lastTopic: 'Agile Methodologies',
        lastClassDuration: 60,
        attendancePercentage: 91.0,
        totalClasses: 16,
        totalStudents: 35,
        description: 'Software development methodologies and best practices.',
      ),
      Subject(
        id: 'sub_006',
        name: 'Operating Systems',
        code: 'CS304',
        icon: Icons.computer_rounded,
        instructorName: 'Dr. Michael Chen',
        instructorId: 'usr_007',
        lastClassDate: DateTime.now().subtract(const Duration(days: 2)),
        lastTopic: 'Process Scheduling',
        lastClassDuration: 90,
        attendancePercentage: 82.5,
        totalClasses: 19,
        totalStudents: 42,
        description: 'Fundamentals of operating system design and implementation.',
      ),
    ];
  }

  /// Get a subject by ID
  static Subject? getSubjectById(String id) {
    try {
      return getSubjects().firstWhere((s) => s.id == id);
    } catch (e) {
      return null;
    }
  }
}
