import 'package:flutter/material.dart';
import '../../utils/theme.dart';

/// Compact attendance stat card with animated counter
class AttendanceCard extends StatelessWidget {
  final String title;
  final String value;
  final IconData icon;
  final AttendanceCardType type;

  const AttendanceCard({
    super.key,
    required this.title,
    required this.value,
    required this.icon,
    this.type = AttendanceCardType.neutral,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final cardColor = _getIconColor();

    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: cardColor.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: cardColor.withOpacity(0.15),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Icon
          Container(
            padding: const EdgeInsets.all(6),
            decoration: BoxDecoration(
              color: cardColor.withOpacity(0.15),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(
              icon,
              size: 16,
              color: cardColor,
            ),
          ),
          const SizedBox(height: 8),
          // Animated value
          TweenAnimationBuilder<double>(
            tween: Tween(begin: 0, end: 1),
            duration: const Duration(milliseconds: 600),
            curve: Curves.easeOutCubic,
            builder: (context, animValue, child) {
              return Opacity(
                opacity: animValue,
                child: Transform.translate(
                  offset: Offset(0, 4 * (1 - animValue)),
                  child: child,
                ),
              );
            },
            child: Text(
              value,
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w700,
                color: cardColor,
                fontSize: 22,
              ),
            ),
          ),
          const SizedBox(height: 2),
          // Title
          Text(
            title,
            style: theme.textTheme.bodySmall?.copyWith(
              color: colorScheme.onSurfaceVariant,
              fontSize: 11,
            ),
          ),
        ],
      ),
    );
  }

  Color _getIconColor() {
    switch (type) {
      case AttendanceCardType.present:
        return AppTheme.presentColor;
      case AttendanceCardType.absent:
        return AppTheme.absentColor;
      case AttendanceCardType.late:
        return AppTheme.lateColor;
      case AttendanceCardType.percentage:
        return AppTheme.primaryColor;
      case AttendanceCardType.neutral:
        return Colors.grey;
    }
  }
}

enum AttendanceCardType {
  present,
  absent,
  late,
  percentage,
  neutral,
}
