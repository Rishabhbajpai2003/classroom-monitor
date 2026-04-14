import 'package:flutter/material.dart';
import '../../mock_data/mock_users.dart';
import '../../mock_data/mock_notifications.dart';
import '../../services/auth_service.dart';
import '../../services/database_service.dart';
import '../../services/subject_service.dart'; // Keep for iconOptions
import '../../models/subject.dart';
import '../../utils/theme.dart';
import '../../widgets/cards/subject_card.dart';
import '../../widgets/common/empty_state.dart';
import '../../widgets/common/animated_nav_bar.dart';
import '../subject/subject_detail_screen.dart';
import '../notifications/notifications_screen.dart';
import '../profile/profile_settings_screen.dart';
import '../analytics/attendance_analytics_screen.dart';
import '../study_assistant/study_assistant_screen.dart';

/// Subject Dashboard Screen - Main screen after login
/// Modern compact dashboard with analytics-style cards and animated navigation
class SubjectDashboardScreen extends StatefulWidget {
  const SubjectDashboardScreen({super.key});

  @override
  State<SubjectDashboardScreen> createState() => _SubjectDashboardScreenState();
}

class _SubjectDashboardScreenState extends State<SubjectDashboardScreen> {
  int _currentIndex = 0;
  List<Subject> _subjects = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadSubjects();
  }

  Future<void> _loadSubjects() async {
    setState(() => _isLoading = true);
    final saved = await DatabaseService().getSubjects();
    if (!mounted) return;
    setState(() {
      _subjects = saved;
      _isLoading = false;
    });
  }

  void _showAddSubjectDialog() {
    final nameCtrl = TextEditingController();
    final codeCtrl = TextEditingController();
    final descCtrl = TextEditingController();
    final studentsCtrl = TextEditingController();
    int selectedIconIndex = 0;
    final user = AuthService().currentUser ?? MockUsers.instructorUser;

    showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) => AlertDialog(
          title: const Row(
            children: [
              Icon(Icons.add_circle_outline_rounded, color: AppTheme.primaryColor),
              SizedBox(width: 8),
              Text('Add Subject'),
            ],
          ),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: nameCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Subject Name *',
                    hintText: 'e.g. Data Structures',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: codeCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Subject Code *',
                    hintText: 'e.g. CS201',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: studentsCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Number of Students',
                    hintText: 'e.g. 40',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: descCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Description (optional)',
                    border: OutlineInputBorder(),
                  ),
                  maxLines: 2,
                ),
                const SizedBox(height: 16),
                const Text('Icon', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: List.generate(SubjectService.iconOptions.length, (i) {
                    final opt = SubjectService.iconOptions[i];
                    final isSelected = i == selectedIconIndex;
                    return GestureDetector(
                      onTap: () => setDialogState(() => selectedIconIndex = i),
                      child: Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: isSelected
                              ? AppTheme.primaryColor
                              : Colors.grey.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(10),
                          border: isSelected
                              ? Border.all(color: AppTheme.primaryColor, width: 2)
                              : null,
                        ),
                        child: Icon(
                          opt['icon'] as IconData,
                          size: 22,
                          color: isSelected ? Colors.white : Colors.grey.shade600,
                        ),
                      ),
                    );
                  }),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              style: ElevatedButton.styleFrom(backgroundColor: AppTheme.primaryColor),
              onPressed: () async {
                if (nameCtrl.text.trim().isEmpty || codeCtrl.text.trim().isEmpty) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Subject name and code are required')),
                  );
                  return;
                }
                
                final newSubject = await DatabaseService().createSubject(
                  name: nameCtrl.text.trim(),
                  code: codeCtrl.text.trim(),
                  description: descCtrl.text.trim(),
                  iconIndex: selectedIconIndex,
                  totalStudents: int.tryParse(studentsCtrl.text.trim()) ?? 0,
                );
                
                if (newSubject != null) {
                  setState(() {
                    _subjects.insert(0, newSubject);
                  });
                  if (context.mounted) Navigator.pop(ctx);
                  if (context.mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text('"${newSubject.name}" added successfully!'),
                        backgroundColor: AppTheme.presentColor,
                      ),
                    );
                  }
                } else {
                  if (context.mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Failed to add subject. Are you logged in?')),
                    );
                  }
                }
              },
              child: const Text('Add Subject', style: TextStyle(color: Colors.white)),
            ),
          ],
        ),
      ),
    );
  }

  void _deleteSubject(Subject subject) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Remove Subject'),
        content: Text('Remove "${subject.name}" from your dashboard?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () async {
              final success = await DatabaseService().deleteSubject(subject.id);
              if (success && context.mounted) {
                setState(() {
                  _subjects.removeWhere((s) => s.id == subject.id);
                });
                Navigator.pop(ctx);
              }
            },
            child: const Text('Remove', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final user = AuthService().currentUser ?? MockUsers.instructorUser;
    final isInstructor = user.isInstructor;

    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: [
          _buildDashboardContent(),
          const AttendanceAnalyticsScreen(),
          const StudyAssistantScreen(),
          const NotificationsScreen(),
          const ProfileSettingsScreen(),
        ],
      ),
      // Only instructors can add subjects
      floatingActionButton: (_currentIndex == 0 && isInstructor)
          ? FloatingActionButton.extended(
              onPressed: _showAddSubjectDialog,
              backgroundColor: AppTheme.primaryColor,
              icon: const Icon(Icons.add_rounded, color: Colors.white),
              label: const Text('Add Subject',
                  style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
            )
          : null,
      bottomNavigationBar: AnimatedNavBar(
        currentIndex: _currentIndex,
        onTap: (index) => setState(() => _currentIndex = index),
        items: const [
          AnimatedNavItem(
            icon: Icons.dashboard_outlined,
            selectedIcon: Icons.dashboard_rounded,
            label: 'Dashboard',
          ),
          AnimatedNavItem(
            icon: Icons.analytics_outlined,
            selectedIcon: Icons.analytics_rounded,
            label: 'Analytics',
          ),
          AnimatedNavItem(
            icon: Icons.psychology_outlined,
            selectedIcon: Icons.psychology_rounded,
            label: 'Study AI',
          ),
          AnimatedNavItem(
            icon: Icons.notifications_outlined,
            selectedIcon: Icons.notifications_rounded,
            label: 'Alerts',
          ),
          AnimatedNavItem(
            icon: Icons.person_outlined,
            selectedIcon: Icons.person_rounded,
            label: 'Profile',
          ),
        ],
      ),
    );
  }

  Widget _buildDashboardContent() {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final user = AuthService().currentUser ?? MockUsers.instructorUser;

    return CustomScrollView(
      slivers: [
        // Compact gradient app bar
        SliverAppBar(
          expandedHeight: 120,
          floating: false,
          pinned: true,
          automaticallyImplyLeading: false,
          backgroundColor: AppTheme.primaryColor,
          flexibleSpace: FlexibleSpaceBar(
            background: Container(
              decoration: const BoxDecoration(
                gradient: AppTheme.primaryGradient,
              ),
              child: SafeArea(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 8, 20, 16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      Row(
                        children: [
                          // Avatar
                          Container(
                            width: 40,
                            height: 40,
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Center(
                              child: Text(
                                user.name.substring(0, 1).toUpperCase(),
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 18,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  _getGreeting(),
                                  style: theme.textTheme.bodySmall?.copyWith(
                                    color: Colors.white.withOpacity(0.7),
                                    fontSize: 12,
                                  ),
                                ),
                                Text(
                                  user.name,
                                  style: theme.textTheme.titleMedium?.copyWith(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          _buildNotificationBadge(colorScheme),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          title: Text(
            'Dashboard',
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),

        // Quick overview stats row
        if (!_isLoading && _subjects.isNotEmpty)
          SliverToBoxAdapter(
            child: _buildOverviewStats(theme, colorScheme),
          ),

        // Content
        if (_isLoading)
          const SliverFillRemaining(
            child: Center(child: CircularProgressIndicator()),
          )
        else if (_subjects.isEmpty)
          SliverFillRemaining(
            child: EmptyState(
              icon: Icons.school_outlined,
              title: 'No Subjects Yet',
              subtitle:
                  user.isInstructor
                  ? 'Tap the "+" button below to add your first subject.'
                  : 'No subjects have been added to your account yet.',
              actionLabel: user.isInstructor ? 'Add Subject' : 'Refresh',
              onAction: user.isInstructor ? _showAddSubjectDialog : _loadSubjects,
            ),
          )
        else
          _buildSubjectGrid(),
      ],
    );
  }

  Widget _buildOverviewStats(ThemeData theme, ColorScheme colorScheme) {
    final avgAttendance = _subjects.isNotEmpty
        ? _subjects.map((s) => s.attendancePercentage).reduce((a, b) => a + b) /
            _subjects.length
        : 0.0;
    final totalStudents =
        _subjects.fold<int>(0, (sum, s) => sum + s.totalStudents);
    final unreadCount = MockNotifications.getUnreadCount();

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
      child: Row(
        children: [
          _MiniStat(
            icon: Icons.school_rounded,
            value: '${_subjects.length}',
            label: 'Courses',
            color: AppTheme.primaryColor,
          ),
          const SizedBox(width: 8),
          _MiniStat(
            icon: Icons.trending_up_rounded,
            value: '${avgAttendance.toStringAsFixed(0)}%',
            label: 'Avg Attend.',
            color: AppTheme.presentColor,
          ),
          const SizedBox(width: 8),
          _MiniStat(
            icon: Icons.people_outline_rounded,
            value: '$totalStudents',
            label: 'Students',
            color: AppTheme.tertiaryColor,
          ),
          const SizedBox(width: 8),
          _MiniStat(
            icon: Icons.notifications_none_rounded,
            value: '$unreadCount',
            label: 'Alerts',
            color: AppTheme.warningColor,
          ),
        ],
      ),
    );
  }

  Widget _buildNotificationBadge(ColorScheme colorScheme) {
    final unreadCount = MockNotifications.getUnreadCount();

    return Stack(
      children: [
        Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.15),
            borderRadius: BorderRadius.circular(12),
          ),
          child: IconButton(
            padding: EdgeInsets.zero,
            icon: const Icon(
              Icons.notifications_outlined,
              color: Colors.white,
              size: 20,
            ),
            onPressed: () {
              setState(() => _currentIndex = 2);
            },
          ),
        ),
        if (unreadCount > 0)
          Positioned(
            right: 4,
            top: 4,
            child: Container(
              width: 16,
              height: 16,
              decoration: BoxDecoration(
                color: AppTheme.absentColor,
                shape: BoxShape.circle,
                border: Border.all(
                  color: AppTheme.primaryColor,
                  width: 1.5,
                ),
              ),
              child: Center(
                child: Text(
                  unreadCount.toString(),
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 8,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildSubjectGrid() {
    return SliverLayoutBuilder(
      builder: (context, constraints) {
        final maxWidth = constraints.crossAxisExtent.clamp(0, 600).toDouble();
        final cardWidth = (maxWidth - 32 - 10) / 2; // padding + spacing
        const cardHeight = 160.0;
        final aspectRatio = cardWidth / cardHeight;

        return SliverPadding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          sliver: SliverGrid(
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              childAspectRatio: aspectRatio.clamp(0.8, 2.5),
              crossAxisSpacing: 10,
              mainAxisSpacing: 10,
            ),
            delegate: SliverChildBuilderDelegate(
              (context, index) {
                final subject = _subjects[index];
                return GestureDetector(
                  onLongPress: () => _deleteSubject(subject),
                  child: SubjectCard(
                    subject: subject,
                    animationIndex: index,
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) =>
                              SubjectDetailScreen(subject: subject),
                        ),
                      );
                    },
                  ),
                );
              },
              childCount: _subjects.length,
            ),
          ),
        );
      },
    );
  }

  String _getGreeting() {
    final hour = DateTime.now().hour;
    if (hour < 12) return 'Good morning,';
    if (hour < 17) return 'Good afternoon,';
    return 'Good evening,';
  }
}

/// Mini stat widget for the overview row
class _MiniStat extends StatelessWidget {
  final IconData icon;
  final String value;
  final String label;
  final Color color;

  const _MiniStat({
    required this.icon,
    required this.value,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: color.withOpacity(0.15),
          ),
        ),
        child: Column(
          children: [
            Icon(icon, size: 16, color: color),
            const SizedBox(height: 4),
            Text(
              value,
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w700,
                fontSize: 14,
                color: colorScheme.onSurface,
              ),
            ),
            Text(
              label,
              style: theme.textTheme.labelSmall?.copyWith(
                color: colorScheme.onSurfaceVariant,
                fontSize: 9,
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }
}
