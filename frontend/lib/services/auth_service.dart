import 'dart:convert';
import 'dart:html' as html;
import 'package:supabase_flutter/supabase_flutter.dart' as sp;
import '../models/user.dart';
import '../utils/env.dart';

/// Persists user session to browser localStorage so the user stays
/// logged in across page refreshes and the correct user-scoped data
/// is restored automatically.
class AuthService {
  static final AuthService _instance = AuthService._internal();
  factory AuthService() => _instance;
  AuthService._internal() {
    _restore();
  }

  static const _sessionKey = 'classroom_monitor_session';

  User? _currentUser;
  
  bool get _useSupabase => !Env.supabaseUrl.contains('YOUR_SUPABASE');

  User? get currentUser {
    if (_useSupabase && sp.Supabase.instance.client.auth.currentSession != null) {
      final supaUser = sp.Supabase.instance.client.auth.currentUser;
      if (supaUser == null) return null;
      // Extract custom metadata if you stored it in Supabase auth metadata
      final roleStr = supaUser.userMetadata?['role'] ?? 'instructor';
      final department = supaUser.userMetadata?['department'] ?? 'Computer Science';
      return User(
        id: supaUser.id,
        name: supaUser.userMetadata?['full_name'] ?? supaUser.email?.split('@')[0] ?? 'User',
        email: supaUser.email ?? '',
        role: roleStr == 'student' ? UserRole.student : UserRole.instructor,
        department: department,
      );
    }
    return _currentUser;
  }

  bool get isLoggedIn => currentUser != null;

  // ── Credentials (demo/mock — replace with real API calls later) ────────────

  static const _demoInstructor = {
    'id': 'usr_001',
    'name': 'Dr. Sarah Johnson',
    'email': 'sarah.johnson@university.edu',
    'role': 'instructor',
    'department': 'Computer Science',
  };

  static const _demoStudent = {
    'id': 'usr_002',
    'name': 'Alex Thompson',
    'email': 'alex.thompson@university.edu',
    'role': 'student',
    'department': 'Computer Science',
  };

  // ── Login / Logout ─────────────────────────────────────────────────────────

  /// Validate credentials and persist session.
  /// Returns null on success, or an error string on failure.
  Future<String?> login(String username, String password, bool asInstructor) async {
    if (_useSupabase) {
      try {
        await sp.Supabase.instance.client.auth.signInWithPassword(
          email: username,
          password: password,
        );
        return null; // success
      } catch (e) {
        return e.toString();
      }
    }

    // Mock fallback
    await Future.delayed(const Duration(milliseconds: 800));

    if (username.trim().isEmpty || password.trim().isEmpty) {
      return 'Please enter username and password';
    }

    final data = asInstructor ? _demoInstructor : _demoStudent;
    _currentUser = _userFromMap(data);

    _saveSession(_currentUser!);
    return null; // success
  }

  Future<void> logout() async {
    if (_useSupabase) {
      await sp.Supabase.instance.client.auth.signOut();
    }
    // NOTE: We intentionally do NOT clear user-scoped analysis data here.
    // The data is keyed to the user's ID, so a different user can never see it.
    // When the same user logs back in, their data will be restored from localStorage.
    _currentUser = null;
    html.window.localStorage.remove(_sessionKey);
  }

  // ── Session persistence ────────────────────────────────────────────────────

  void _saveSession(User user) {
    final map = {
      'id': user.id,
      'name': user.name,
      'email': user.email,
      'role': user.role == UserRole.instructor ? 'instructor' : 'student',
      'department': user.department,
    };
    html.window.localStorage[_sessionKey] = json.encode(map);
  }

  void _restore() {
    try {
      final jsonStr = html.window.localStorage[_sessionKey];
      if (jsonStr != null && jsonStr.isNotEmpty) {
        final map = json.decode(jsonStr) as Map<String, dynamic>;
        _currentUser = _userFromMap(map);
      }
    } catch (_) {
      _currentUser = null;
    }
  }

  User _userFromMap(Map<String, dynamic> m) {
    return User(
      id: m['id'] as String,
      name: m['name'] as String,
      email: m['email'] as String,
      role: (m['role'] as String) == 'instructor'
          ? UserRole.instructor
          : UserRole.student,
      department: m['department'] as String,
    );
  }

  // ── User-scoped localStorage key ───────────────────────────────────────────

  /// Returns a localStorage key scoped to the current user's ID.
  /// This ensures different users get different analysis data.
  String scopedKey(String base) {
    final uid = _currentUser?.id ?? 'anonymous';
    return '${base}_$uid';
  }
}
