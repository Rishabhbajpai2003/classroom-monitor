import 'package:flutter/animation.dart';

/// App-wide constants
class AppConstants {
  AppConstants._();

  // App info
  static const String appName = 'Classroom Monitor';
  static const String appVersion = '1.0.0';

  // Spacing
  static const double spacingXxs = 2.0;
  static const double spacingXs = 4.0;
  static const double spacingSm = 8.0;
  static const double spacingMd = 16.0;
  static const double spacingLg = 24.0;
  static const double spacingXl = 32.0;
  static const double spacingXxl = 48.0;

  // Border radius
  static const double radiusSm = 8.0;
  static const double radiusMd = 12.0;
  static const double radiusLg = 14.0;
  static const double radiusXl = 20.0;
  static const double radiusCircle = 100.0;

  // Animation durations
  static const Duration animationFast = Duration(milliseconds: 150);
  static const Duration animationNormal = Duration(milliseconds: 300);
  static const Duration animationSlow = Duration(milliseconds: 500);
  static const Duration animationStagger = Duration(milliseconds: 60);

  // Animation curves
  static const Curve curveDefault = Curves.easeOutCubic;
  static const Curve curveEmphasized = Curves.easeInOutCubicEmphasized;
  static const Curve curveSpring = Curves.elasticOut;

  // Mock processing delays
  static const Duration mockProcessingDelay = Duration(seconds: 3);
  static const Duration mockLoadingDelay = Duration(milliseconds: 800);
}

/// Route names for navigation
class Routes {
  Routes._();

  static const String login = '/login';
  static const String dashboard = '/dashboard';
  static const String subjectDetail = '/subject';
  static const String attendanceSummary = '/attendance-summary';
  static const String attendanceList = '/attendance-list';
  static const String uploadVideo = '/upload-video';
  static const String processingStatus = '/processing-status';
  static const String analytics = '/analytics';
  static const String notifications = '/notifications';
  static const String profile = '/profile';
  static const String privacyConsent = '/privacy-consent';
}
