/// User model representing a user of the classroom monitoring system.
/// Users can be either Instructors or Students with different permissions.
enum UserRole { instructor, student }

class User {
  final String id;
  final String name;
  final String email;
  final UserRole role;
  final String? avatarUrl;
  final String department;

  const User({
    required this.id,
    required this.name,
    required this.email,
    required this.role,
    this.avatarUrl,
    required this.department,
  });

  /// Check if user is an instructor
  bool get isInstructor => role == UserRole.instructor;

  /// Check if user is a student
  bool get isStudent => role == UserRole.student;

  /// Get role display name
  String get roleDisplayName => role == UserRole.instructor ? 'Instructor' : 'Student';

  /// Create a copy with modified fields
  User copyWith({
    String? id,
    String? name,
    String? email,
    UserRole? role,
    String? avatarUrl,
    String? department,
  }) {
    return User(
      id: id ?? this.id,
      name: name ?? this.name,
      email: email ?? this.email,
      role: role ?? this.role,
      avatarUrl: avatarUrl ?? this.avatarUrl,
      department: department ?? this.department,
    );
  }
}
