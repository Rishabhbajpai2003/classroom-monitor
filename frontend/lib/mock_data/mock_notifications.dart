import '../models/notification.dart';

/// Mock notifications data for development and testing
class MockNotifications {
  static List<AppNotification> getNotifications() {
    final now = DateTime.now();
    
    return [
      AppNotification(
        id: 'notif_001',
        title: 'Attendance Generated',
        message: 'Attendance for Data Structures class has been generated from video analysis.',
        type: NotificationType.attendanceGenerated,
        timestamp: now.subtract(const Duration(minutes: 15)),
        isRead: false,
        actionRoute: '/attendance',
        metadata: {'subjectId': 'sub_001'},
      ),
      AppNotification(
        id: 'notif_002',
        title: 'Video Processing Complete',
        message: 'Your class recording has been processed. 38 students were detected.',
        type: NotificationType.videoProcessed,
        timestamp: now.subtract(const Duration(hours: 1)),
        isRead: false,
        actionRoute: '/processing',
        metadata: {'videoId': 'vid_001'},
      ),
      AppNotification(
        id: 'notif_003',
        title: 'Class Reminder',
        message: 'Machine Learning class starts in 30 minutes (Room 301)',
        type: NotificationType.classReminder,
        timestamp: now.subtract(const Duration(hours: 2)),
        isRead: true,
        actionRoute: '/subject/sub_002',
      ),
      AppNotification(
        id: 'notif_004',
        title: 'Low Attendance Alert',
        message: 'Database Systems class had only 65% attendance yesterday.',
        type: NotificationType.alert,
        timestamp: now.subtract(const Duration(hours: 5)),
        isRead: true,
        actionRoute: '/attendance',
        metadata: {'subjectId': 'sub_003'},
      ),
      AppNotification(
        id: 'notif_005',
        title: 'New Announcement',
        message: 'Mid-semester exams scheduled for next week. Check the timetable.',
        type: NotificationType.announcement,
        timestamp: now.subtract(const Duration(days: 1)),
        isRead: true,
      ),
      AppNotification(
        id: 'notif_006',
        title: 'Attendance Generated',
        message: 'Attendance for Computer Networks class has been generated.',
        type: NotificationType.attendanceGenerated,
        timestamp: now.subtract(const Duration(days: 1, hours: 3)),
        isRead: true,
        actionRoute: '/attendance',
        metadata: {'subjectId': 'sub_004'},
      ),
      AppNotification(
        id: 'notif_007',
        title: 'Class Reminder',
        message: 'Software Engineering class tomorrow at 10:00 AM',
        type: NotificationType.classReminder,
        timestamp: now.subtract(const Duration(days: 2)),
        isRead: true,
      ),
      AppNotification(
        id: 'notif_008',
        title: 'Video Processing Complete',
        message: 'Operating Systems lecture video has been processed successfully.',
        type: NotificationType.videoProcessed,
        timestamp: now.subtract(const Duration(days: 3)),
        isRead: true,
        metadata: {'videoId': 'vid_002'},
      ),
    ];
  }

  /// Get unread notifications count
  static int getUnreadCount() {
    return getNotifications().where((n) => !n.isRead).length;
  }
}
