import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/subject.dart';
import 'auth_service.dart';

/// Manages data fetching and writing to Supabase for Subjects and Enrollments
class DatabaseService {
  static final DatabaseService _instance = DatabaseService._internal();
  factory DatabaseService() => _instance;
  DatabaseService._internal();

  final _supabase = Supabase.instance.client;

  /// Fetch subjects for the current authenticated instructor
  Future<List<Subject>> getSubjects() async {
    final uid = AuthService().currentUser?.id;
    if (uid == null) return [];

    try {
      final res = await _supabase
          .from('subjects')
          .select('*')
          .order('created_at', ascending: false);

      return (res as List<dynamic>).map((e) => _mapSubject(e)).toList();
    } catch (e) {
      print('Database error getting subjects: $e');
      return [];
    }
  }

  /// Create a new subject in Supabase
  Future<Subject?> createSubject({
    required String name,
    required String code,
    required String description,
    required int iconIndex,
    required int totalStudents,
  }) async {
    final uid = Supabase.instance.client.auth.currentUser?.id;
    if (uid == null) return null;

    try {
      final res = await _supabase.from('subjects').insert({
        'name': name,
        'code': code.toUpperCase(),
        'description': description.isEmpty ? null : description,
        'icon_index': iconIndex,
        'total_students': totalStudents,
        'instructor_id': uid,
      }).select().single();

      return _mapSubject(res);
    } catch (e) {
      print('Database error creating subject: $e');
      return null;
    }
  }

  /// Delete a subject from Supabase
  Future<bool> deleteSubject(String subjectId) async {
    try {
      await _supabase.from('subjects').delete().eq('id', subjectId);
      return true;
    } catch (e) {
      print('Database error deleting subject: $e');
      return false;
    }
  }

  /// Fetch enrolled students for a subject
  Future<List<Map<String, dynamic>>> getEnrolledStudents(String subjectId) async {
    try {
      final res = await _supabase
          .from('enrollments')
          .select('''
            student:student_id (
              id,
              name
            )
          ''')
          .eq('subject_id', subjectId);
      
      return (res as List<dynamic>).map((e) {
        final st = e['student'] as Map<String, dynamic>;
        return {
          'id': st['id'],
          'name': st['name'],
        };
      }).toList();
    } catch (e) {
      print('Database error getting enrollments: $e');
      return [];
    }
  }

  Subject _mapSubject(Map<String, dynamic> data) {
    // Map the icon index back to a standard set (matching SubjectService if it existed)
    // Here we'll just mock it or map directly if you imported it.
    // We use a simplified mapping for now.
    return Subject(
      id: data['id'] as String,
      name: data['name'] as String,
      code: data['code'] as String,
      icon: _getIconByIndex(data['icon_index'] as int? ?? 0),
      instructorName: AuthService().currentUser?.email ?? 'Instructor',
      instructorId: data['instructor_id'] as String,
      description: data['description'] as String?,
      totalStudents: data['total_students'] as int? ?? 0,
      attendancePercentage: 0.0, // This would be calculated via logs later
      // We don't map dates from the db for this UI mock, but we could
    );
  }

  static IconData _getIconByIndex(int index) {
    const icons = [
      Icons.account_tree_rounded, // 0
      Icons.psychology_rounded,   // 1 (DL26)
      Icons.storage_rounded,
      Icons.hub_rounded,
      Icons.engineering_rounded,
      Icons.computer_rounded,
      Icons.code_rounded,
      Icons.calculate_rounded,
      Icons.science_rounded,
      Icons.architecture_rounded,
      Icons.school_rounded,
      Icons.analytics_rounded,
    ];
    if (index >= 0 && index < icons.length) return icons[index];
    return Icons.school_rounded;
  }
}
