import 'dart:convert';
import 'dart:html' as html;
import 'package:flutter/material.dart';
import '../models/subject.dart';
import 'auth_service.dart';

/// Manages user-specific subjects stored in localStorage.
/// Instructors can add/remove their own subjects; students see all shared subjects.
class SubjectService {
  static final SubjectService _instance = SubjectService._internal();
  factory SubjectService() => _instance;
  SubjectService._internal();

  static const _storageKey = 'classroom_monitor_subjects';

  // Icon options available when creating a subject
  static const List<Map<String, dynamic>> iconOptions = [
    {'icon': Icons.account_tree_rounded, 'label': 'Algorithms'},
    {'icon': Icons.psychology_rounded, 'label': 'AI/ML'},
    {'icon': Icons.storage_rounded, 'label': 'Database'},
    {'icon': Icons.hub_rounded, 'label': 'Networks'},
    {'icon': Icons.engineering_rounded, 'label': 'Engineering'},
    {'icon': Icons.computer_rounded, 'label': 'OS'},
    {'icon': Icons.code_rounded, 'label': 'Programming'},
    {'icon': Icons.calculate_rounded, 'label': 'Math'},
    {'icon': Icons.science_rounded, 'label': 'Science'},
    {'icon': Icons.architecture_rounded, 'label': 'Architecture'},
    {'icon': Icons.school_rounded, 'label': 'General'},
    {'icon': Icons.analytics_rounded, 'label': 'Analytics'},
  ];

  String get _scopedKey {
    final uid = AuthService().currentUser?.id ?? 'anonymous';
    return '${_storageKey}_$uid';
  }

  /// Load subjects from localStorage for the current user.
  List<Subject> loadSubjects() {
    try {
      final raw = html.window.localStorage[_scopedKey];
      if (raw == null || raw.isEmpty) return [];
      final list = json.decode(raw) as List<dynamic>;
      return list.map((e) => _fromMap(e as Map<String, dynamic>)).toList();
    } catch (_) {
      return [];
    }
  }

  /// Save the full list of subjects to localStorage.
  void saveSubjects(List<Subject> subjects) {
    final list = subjects.map(_toMap).toList();
    html.window.localStorage[_scopedKey] = json.encode(list);
  }

  /// Add a new subject and persist.
  List<Subject> addSubject(List<Subject> current, Subject subject) {
    final updated = [...current, subject];
    saveSubjects(updated);
    return updated;
  }

  /// Remove a subject by ID and persist.
  List<Subject> removeSubject(List<Subject> current, String subjectId) {
    final updated = current.where((s) => s.id != subjectId).toList();
    saveSubjects(updated);
    return updated;
  }

  // ── Serialization ──────────────────────────────────────────────────────────

  Map<String, dynamic> _toMap(Subject s) => {
        'id': s.id,
        'name': s.name,
        'code': s.code,
        'iconIndex': _iconIndex(s.icon),
        'instructorName': s.instructorName,
        'instructorId': s.instructorId,
        'lastTopic': s.lastTopic,
        'lastClassDuration': s.lastClassDuration,
        'description': s.description,
        'totalStudents': s.totalStudents,
      };

  Subject _fromMap(Map<String, dynamic> m) {
    final iconIdx = (m['iconIndex'] as int?) ?? 0;
    final iconData =
        iconOptions[iconIdx.clamp(0, iconOptions.length - 1)]['icon'] as IconData;
    return Subject(
      id: m['id'] as String,
      name: m['name'] as String,
      code: m['code'] as String,
      icon: iconData,
      instructorName: m['instructorName'] as String,
      instructorId: m['instructorId'] as String,
      lastTopic: m['lastTopic'] as String?,
      lastClassDuration: m['lastClassDuration'] as int?,
      description: m['description'] as String?,
      totalStudents: (m['totalStudents'] as int?) ?? 0,
    );
  }

  int _iconIndex(IconData icon) {
    for (int i = 0; i < iconOptions.length; i++) {
      if (iconOptions[i]['icon'] == icon) return i;
    }
    return 0;
  }
}
