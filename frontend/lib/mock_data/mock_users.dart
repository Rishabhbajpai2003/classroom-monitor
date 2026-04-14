import '../models/user.dart';

/// Mock users for development and testing
class MockUsers {
  /// Current logged in user - can be changed to test different roles
  static User currentUser = instructorUser;

  /// Mock instructor user
  static const User instructorUser = User(
    id: 'usr_001',
    name: 'Dr. Sarah Johnson',
    email: 'sarah.johnson@university.edu',
    role: UserRole.instructor,
    avatarUrl: null,
    department: 'Computer Science',
  );

  /// Mock student user
  static const User studentUser = User(
    id: 'usr_002',
    name: 'Alex Thompson',
    email: 'alex.thompson@university.edu',
    role: UserRole.student,
    avatarUrl: null,
    department: 'Computer Science',
  );

  /// List of all mock students
  static const List<User> students = [
    User(
      id: 'stu_001',
      name: 'Alex Thompson',
      email: 'alex.thompson@university.edu',
      role: UserRole.student,
      department: 'Computer Science',
    ),
    User(
      id: 'stu_002',
      name: 'Emily Chen',
      email: 'emily.chen@university.edu',
      role: UserRole.student,
      department: 'Computer Science',
    ),
    User(
      id: 'stu_003',
      name: 'Michael Brown',
      email: 'michael.brown@university.edu',
      role: UserRole.student,
      department: 'Computer Science',
    ),
    User(
      id: 'stu_004',
      name: 'Jessica Williams',
      email: 'jessica.williams@university.edu',
      role: UserRole.student,
      department: 'Computer Science',
    ),
    User(
      id: 'stu_005',
      name: 'David Martinez',
      email: 'david.martinez@university.edu',
      role: UserRole.student,
      department: 'Computer Science',
    ),
    User(
      id: 'stu_006',
      name: 'Amanda Lee',
      email: 'amanda.lee@university.edu',
      role: UserRole.student,
      department: 'Computer Science',
    ),
    User(
      id: 'stu_007',
      name: 'Ryan Taylor',
      email: 'ryan.taylor@university.edu',
      role: UserRole.student,
      department: 'Computer Science',
    ),
    User(
      id: 'stu_008',
      name: 'Sophia Garcia',
      email: 'sophia.garcia@university.edu',
      role: UserRole.student,
      department: 'Computer Science',
    ),
  ];

  /// Toggle between instructor and student role for testing
  static void toggleRole() {
    if (currentUser.isInstructor) {
      currentUser = studentUser;
    } else {
      currentUser = instructorUser;
    }
  }

  /// Set current user to instructor
  static void setInstructor() {
    currentUser = instructorUser;
  }

  /// Set current user to student
  static void setStudent() {
    currentUser = studentUser;
  }
}
