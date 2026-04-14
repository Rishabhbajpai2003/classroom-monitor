import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../../models/subject.dart';
import '../../models/attendance.dart';
import '../../services/analysis_data_service.dart';
import '../../widgets/common/empty_state.dart';
import '../../utils/theme.dart';

/// Attendance List Screen — tighter spacing, compact rows, staggered animation
class AttendanceListScreen extends StatefulWidget {
  final Subject subject;
  final DateTime date;

  const AttendanceListScreen({
    super.key,
    required this.subject,
    required this.date,
  });

  @override
  State<AttendanceListScreen> createState() => _AttendanceListScreenState();
}

class _AttendanceListScreenState extends State<AttendanceListScreen> {
  List<Attendance> _allRecords = [];
  List<Attendance> _filteredRecords = [];
  final TextEditingController _searchController = TextEditingController();
  AttendanceStatus? _statusFilter;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadAttendance();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _loadAttendance() async {
    setState(() => _isLoading = true);
    
    // OFFLINE MODE: Using CSV data from the model directly instead of Supabase
    // final enrolled = await DatabaseService().getEnrolledStudents(widget.subject.id);
    
    // Get live data if a video was analyzed
    final liveData = AnalysisDataService().latestData;

    // Simulate network delay for UI consistency
    await Future.delayed(const Duration(milliseconds: 600));

    if (!mounted) return;
    
    setState(() {
      if (liveData == null || liveData.records.isEmpty) {
        _allRecords = [];
      } else {
        // Build the records from the actual AI model's CSV output
        _allRecords = liveData.records.map((r) {
          return Attendance(
            id: 'att_${r.studentId}',
            studentId: r.studentId,
            subjectId: widget.subject.id,
            date: widget.date,
            status: r.isPresent ? AttendanceStatus.present : AttendanceStatus.absent,
            studentName: r.name,
            rollNumber: r.studentId, // From CSV
          );
        }).toList();
      }
      _filteredRecords = _allRecords;
      _isLoading = false;
    });
  }

  void _applyFilters() {
    setState(() {
      _filteredRecords = _allRecords.where((record) {
        final searchQuery = _searchController.text.toLowerCase();
        final matchesSearch = searchQuery.isEmpty ||
            record.studentName.toLowerCase().contains(searchQuery) ||
            record.rollNumber.toLowerCase().contains(searchQuery);
        final matchesStatus =
            _statusFilter == null || record.status == _statusFilter;
        return matchesSearch && matchesStatus;
      }).toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final dateFormat = DateFormat('EEE, MMM dd, yyyy');

    return Scaffold(
      appBar: AppBar(
        title: const Text('Attendance List'),
        actions: [
          IconButton(
            icon: const Icon(Icons.download_rounded, size: 20),
            onPressed: () => _showExportSnackbar(context),
            tooltip: 'Export CSV',
          ),
        ],
      ),
      body: Column(
        children: [
          // Compact header
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: colorScheme.surface,
              border: Border(
                bottom: BorderSide(
                  color: colorScheme.outlineVariant.withOpacity(0.2),
                ),
              ),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    widget.subject.name,
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w600,
                      fontSize: 13,
                    ),
                  ),
                ),
                Icon(
                  Icons.calendar_today_rounded,
                  size: 12,
                  color: colorScheme.onSurfaceVariant,
                ),
                const SizedBox(width: 4),
                Text(
                  dateFormat.format(widget.date),
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),

          // Search and filters
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: Column(
              children: [
                // Search
                SizedBox(
                  height: 42,
                  child: TextField(
                    controller: _searchController,
                    style: theme.textTheme.bodySmall?.copyWith(fontSize: 13),
                    decoration: InputDecoration(
                      hintText: 'Search by name or roll number',
                      hintStyle: theme.textTheme.bodySmall?.copyWith(
                        color: colorScheme.onSurfaceVariant,
                        fontSize: 12,
                      ),
                      prefixIcon: const Icon(Icons.search_rounded, size: 18),
                      contentPadding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 0),
                      suffixIcon: _searchController.text.isNotEmpty
                          ? IconButton(
                              icon: const Icon(Icons.clear_rounded, size: 16),
                              onPressed: () {
                                _searchController.clear();
                                _applyFilters();
                              },
                            )
                          : null,
                    ),
                    onChanged: (_) => _applyFilters(),
                  ),
                ),
                const SizedBox(height: 8),
                _buildFilterChips(theme, colorScheme),
              ],
            ),
          ),

          // Results count
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                Text(
                  '${_filteredRecords.length} of ${_allRecords.length} students',
                  style: theme.textTheme.labelSmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 6),

          // List
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : _filteredRecords.isEmpty
                    ? const EmptyState(
                        icon: Icons.search_off_rounded,
                        title: 'No Results Found',
                        subtitle: 'Try adjusting your search or filters',
                      )
                    : _buildAttendanceList(theme, colorScheme),
          ),
        ],
      ),
    );
  }

  Widget _buildFilterChips(ThemeData theme, ColorScheme colorScheme) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: [
          _FilterChip(
            label: 'All',
            isSelected: _statusFilter == null,
            onTap: () {
              setState(() => _statusFilter = null);
              _applyFilters();
            },
          ),
          const SizedBox(width: 6),
          _FilterChip(
            label: 'Present',
            isSelected: _statusFilter == AttendanceStatus.present,
            color: AppTheme.presentColor,
            onTap: () {
              setState(() => _statusFilter = AttendanceStatus.present);
              _applyFilters();
            },
          ),
          const SizedBox(width: 6),
          _FilterChip(
            label: 'Absent',
            isSelected: _statusFilter == AttendanceStatus.absent,
            color: AppTheme.absentColor,
            onTap: () {
              setState(() => _statusFilter = AttendanceStatus.absent);
              _applyFilters();
            },
          ),
          const SizedBox(width: 6),
          _FilterChip(
            label: 'Late',
            isSelected: _statusFilter == AttendanceStatus.late,
            color: AppTheme.lateColor,
            onTap: () {
              setState(() => _statusFilter = AttendanceStatus.late);
              _applyFilters();
            },
          ),
          const SizedBox(width: 6),
          _FilterChip(
            label: 'Excused',
            isSelected: _statusFilter == AttendanceStatus.excused,
            color: AppTheme.excusedColor,
            onTap: () {
              setState(() => _statusFilter = AttendanceStatus.excused);
              _applyFilters();
            },
          ),
        ],
      ),
    );
  }

  Widget _buildAttendanceList(ThemeData theme, ColorScheme colorScheme) {
    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      itemCount: _filteredRecords.length,
      itemBuilder: (context, index) {
        final record = _filteredRecords[index];
        return TweenAnimationBuilder<double>(
          tween: Tween(begin: 0, end: 1),
          duration: Duration(milliseconds: 300 + (index * 30).clamp(0, 300)),
          curve: Curves.easeOutCubic,
          builder: (context, value, child) {
            return Opacity(
              opacity: value,
              child: Transform.translate(
                offset: Offset(0, 8 * (1 - value)),
                child: child,
              ),
            );
          },
          child: _AttendanceRow(
            record: record,
            index: index + 1,
          ),
        );
      },
    );
  }

  void _showExportSnackbar(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Row(
          children: [
            Icon(Icons.check_circle, color: Colors.white, size: 16),
            SizedBox(width: 8),
            Text('CSV export available with backend integration'),
          ],
        ),
        behavior: SnackBarBehavior.floating,
        duration: Duration(seconds: 3),
      ),
    );
  }
}

/// Compact filter chip
class _FilterChip extends StatelessWidget {
  final String label;
  final bool isSelected;
  final Color? color;
  final VoidCallback onTap;

  const _FilterChip({
    required this.label,
    required this.isSelected,
    this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final chipColor = color ?? AppTheme.primaryColor;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected ? chipColor.withOpacity(0.15) : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isSelected
                ? chipColor
                : colorScheme.outlineVariant.withOpacity(0.4),
            width: 1,
          ),
        ),
        child: Text(
          label,
          style: theme.textTheme.labelSmall?.copyWith(
            color: isSelected ? chipColor : colorScheme.onSurfaceVariant,
            fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
            fontSize: 11,
          ),
        ),
      ),
    );
  }
}

/// Compact attendance row
class _AttendanceRow extends StatelessWidget {
  final Attendance record;
  final int index;

  const _AttendanceRow({
    required this.record,
    required this.index,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: colorScheme.outlineVariant.withOpacity(0.2),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.02),
            blurRadius: 4,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Row(
        children: [
          // Index
          Container(
            width: 28,
            height: 28,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: colorScheme.surfaceContainerHighest.withOpacity(0.5),
              borderRadius: BorderRadius.circular(7),
            ),
            child: Text(
              index.toString(),
              style: theme.textTheme.labelSmall?.copyWith(
                fontWeight: FontWeight.w600,
                fontSize: 11,
              ),
            ),
          ),
          const SizedBox(width: 10),
          // Student info
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  record.studentName,
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w500,
                    fontSize: 13,
                  ),
                ),
                Text(
                  record.rollNumber,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
          // Status badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: _getStatusColor().withOpacity(0.12),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  _getStatusIcon(),
                  size: 12,
                  color: _getStatusColor(),
                ),
                const SizedBox(width: 3),
                Text(
                  record.statusDisplayName,
                  style: theme.textTheme.labelSmall?.copyWith(
                    color: _getStatusColor(),
                    fontWeight: FontWeight.w600,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _getStatusColor() {
    switch (record.status) {
      case AttendanceStatus.present:
        return AppTheme.presentColor;
      case AttendanceStatus.absent:
        return AppTheme.absentColor;
      case AttendanceStatus.late:
        return AppTheme.lateColor;
      case AttendanceStatus.excused:
        return AppTheme.excusedColor;
    }
  }

  IconData _getStatusIcon() {
    switch (record.status) {
      case AttendanceStatus.present:
        return Icons.check_circle_rounded;
      case AttendanceStatus.absent:
        return Icons.cancel_rounded;
      case AttendanceStatus.late:
        return Icons.schedule_rounded;
      case AttendanceStatus.excused:
        return Icons.info_rounded;
    }
  }
}
