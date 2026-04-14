import 'dart:html' as html;
import 'package:flutter/material.dart';
import '../../models/subject.dart';
import '../../services/api_service.dart';
import '../../services/analysis_data_service.dart';
import '../../widgets/video/video_preview_widget.dart';
import '../../widgets/tables/csv_data_table_widget.dart';

/// Shows analysis results from the FastAPI backend and live attendance stats parsed from CSV.
class AnalysisResultsScreen extends StatefulWidget {
  final Subject subject;
  final String videoName;
  final GradioAnalysisResult? result; // null = demo mode

  const AnalysisResultsScreen({
    super.key,
    required this.subject,
    required this.videoName,
    required this.result,
  });

  @override
  State<AnalysisResultsScreen> createState() => _AnalysisResultsScreenState();
}

class _AnalysisResultsScreenState extends State<AnalysisResultsScreen> {
  final _dataService = AnalysisDataService();
  LiveAttendanceData? _liveData;
  bool _parsingCsv = false;
  String? _parseError;

  GradioAnalysisResult? _currentResult;
  String _selectedCsvType = 'attendance'; // 'attendance', 'activity', 'speech', 'timeline', 'events'

  @override
  void initState() {
    super.initState();
    _currentResult = widget.result ?? _dataService.latestResult;
    
    if (_currentResult != null) {
      _parseCsvs();
    }
  }

  Future<void> _parseCsvs() async {
    setState(() => _parsingCsv = true);
    try {
      await _dataService.parseFromResult(_currentResult!);
      if (mounted) {
        setState(() {
          _liveData = _dataService.latestData;
          _parsingCsv = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _parseError = e.toString();
          _parsingCsv = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final isDemo = _currentResult == null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Analysis Results'),
        leading: BackButton(onPressed: () => Navigator.pop(context)),
        actions: [
          if (!isDemo)
            IconButton(
              icon: const Icon(Icons.description_outlined),
              tooltip: 'View Technical Logs',
              onPressed: _showLogViewer,
            ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildHeaderCard(theme, colorScheme, isDemo),
            const SizedBox(height: 20),

            if (!isDemo && _currentResult != null) ...[
              _buildVideoPreviewSection(theme, colorScheme),
              const SizedBox(height: 24),
            ],

            if (isDemo) ...[
              _buildDemoBanner(theme, colorScheme),
              const SizedBox(height: 20),
            ],

            // ── Live Stats from CSV ─────────────────────────────────────────
            if (!isDemo) ...[
              _buildLiveStatsSection(theme, colorScheme),
              const SizedBox(height: 24),
              _buildCsvExplorerSection(theme, colorScheme),
              const SizedBox(height: 24),
            ],

            // ── Downloads ──────────────────────────────────────────────────
            Text('Downloads',
                style: theme.textTheme.titleMedium
                    ?.copyWith(fontWeight: FontWeight.w600)),
            const SizedBox(height: 12),

            _buildDownloadTile(context,
                icon: Icons.ondemand_video_rounded,
                color: Colors.purple,
                title: 'Attention-Annotated Video',
                subtitle: 'Face & attention tracking overlay',
                url: isDemo ? null : _currentResult!.attentionVideoUrl,
                filename: 'attention_video.mp4'),
            const SizedBox(height: 10),

            _buildDownloadTile(context,
                icon: Icons.table_chart_rounded,
                color: Colors.green,
                title: 'Attendance Report (CSV)',
                subtitle: 'attendance_report.csv',
                url: isDemo ? null : _currentResult!.attendanceCsvUrl,
                filename: 'attendance_report.csv'),
            const SizedBox(height: 10),

            _buildDownloadTile(context,
                icon: Icons.directions_run_rounded,
                color: Colors.orange,
                title: 'Activity-Tracked Video',
                subtitle: 'Pose & activity tracking overlay',
                url: isDemo ? null : _currentResult!.activityVideoUrl,
                filename: 'activity_tracking.mp4'),
            const SizedBox(height: 10),

            _buildDownloadTile(context,
                icon: Icons.bar_chart_rounded,
                color: Colors.blue,
                title: 'Activity Summary (CSV)',
                subtitle: 'person_activity_summary.csv',
                url: isDemo ? null : _currentResult!.activityCsvUrl,
                filename: 'person_activity_summary.csv'),

            // ── NEW: Speech Topic Results ───────────────────────────────────
            if (!isDemo && (_currentResult!.hasSpeechVideo || _currentResult!.hasSpeechCsv)) ...[
              const SizedBox(height: 20),
              _buildSectionHeader(theme, '🎙️ Speech Topic Analysis'),
              const SizedBox(height: 10),
              if (_currentResult!.hasSpeechVideo)
                _buildDownloadTile(context,
                    icon: Icons.record_voice_over_rounded,
                    color: Colors.teal,
                    title: 'Speech Topic Video',
                    subtitle: 'Annotated video with class-related / off-topic labels',
                    url: _currentResult!.speechVideoUrl,
                    filename: 'speech_topics.mp4'),
              if (_currentResult!.hasSpeechVideo && _currentResult!.hasSpeechCsv)
                const SizedBox(height: 10),
              if (_currentResult!.hasSpeechCsv)
                _buildDownloadTile(context,
                    icon: Icons.text_snippet_outlined,
                    color: Colors.teal,
                    title: 'Speech Topic Segments (CSV)',
                    subtitle: 'Timestamped class_related / off_topic / unknown segments',
                    url: _currentResult!.speechCsvUrl,
                    filename: 'speech_topic_segments.csv'),
            ],

            // ── NEW: Seat Map ───────────────────────────────────────────────
            if (!isDemo && (_currentResult!.hasSeatMapPng || _currentResult!.hasSeatMapJson)) ...[
              const SizedBox(height: 20),
              _buildSectionHeader(theme, '🗺️ Seat Map'),
              const SizedBox(height: 10),
              if (_currentResult!.hasSeatMapPng) ...[
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.network(
                    _currentResult!.seatMapPngUrl!,
                    fit: BoxFit.contain,
                    errorBuilder: (_, __, ___) => const SizedBox.shrink(),
                  ),
                ),
                const SizedBox(height: 10),
              ],
              if (_currentResult!.hasSeatMapJson)
                _buildDownloadTile(context,
                    icon: Icons.event_seat_rounded,
                    color: Colors.indigo,
                    title: 'Seat Map (JSON)',
                    subtitle: 'Student-to-seat assignment data',
                    url: _currentResult!.seatMapJsonUrl,
                    filename: 'seat_map.json'),
            ],

            // ── NEW: Seating Timeline & Attendance Events ───────────────────
            if (!isDemo && (_currentResult!.hasSeatingTimeline || _currentResult!.hasAttendanceEvents)) ...[
              const SizedBox(height: 20),
              _buildSectionHeader(theme, '📋 Seating & Events'),
              const SizedBox(height: 10),
              if (_currentResult!.hasSeatingTimeline)
                _buildDownloadTile(context,
                    icon: Icons.timeline_rounded,
                    color: Colors.deepPurple,
                    title: 'Student Seating Timeline (CSV)',
                    subtitle: 'Which seat each student occupied over time',
                    url: _currentResult!.seatingTimelineUrl,
                    filename: 'student_seating_timeline.csv'),
              if (_currentResult!.hasSeatingTimeline && _currentResult!.hasAttendanceEvents)
                const SizedBox(height: 10),
              if (_currentResult!.hasAttendanceEvents)
                _buildDownloadTile(context,
                    icon: Icons.event_note_rounded,
                    color: Colors.deepOrange,
                    title: 'Attendance Events (CSV)',
                    subtitle: 'Timestamped entry, exit & seat-shift events per student',
                    url: _currentResult!.attendanceEventsUrl,
                    filename: 'attendance_events.csv'),
            ],

            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionHeader(ThemeData theme, String title, {IconData? icon}) {
    return Row(
      children: [
        if (icon != null) ...[
          Icon(icon, size: 20, color: theme.colorScheme.primary),
          const SizedBox(width: 8),
        ],
        Text(
          title,
          style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
        ),
      ],
    );
  }

  // ── Video Preview Section ─────────────────────────────────────────────────

  Widget _buildVideoPreviewSection(ThemeData theme, ColorScheme colorScheme) {
    final videoUrl = _currentResult!.attentionVideoUrl;
    if (videoUrl == null) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildSectionHeader(theme, 'Analysis Preview', icon: Icons.play_circle_outline),
        const SizedBox(height: 12),
        VideoPreviewWidget(
          videoUrl: videoUrl,
          aspectRatio: 16 / 9,
          autoPlay: false,
        ),
        const SizedBox(height: 8),
        Text(
          'Primary Result: Attention & Seating Visualization',
          style: theme.textTheme.bodySmall?.copyWith(color: colorScheme.onSurfaceVariant),
        ),
      ],
    );
  }

  // ── CSV Explorer Section ──────────────────────────────────────────────────

  Widget _buildCsvExplorerSection(ThemeData theme, ColorScheme colorScheme) {
    final availableCsvs = <String, String?>{
      'attendance': _currentResult!.attendanceCsvUrl,
      'activity': _currentResult!.activityCsvUrl,
      'speech': _currentResult!.speechCsvUrl,
      'timeline': _currentResult!.seatingTimelineUrl,
      'events': _currentResult!.attendanceEventsUrl,
    };

    // Filter out nulls
    final validCsvs = availableCsvs.entries.where((e) => e.value != null).toList();
    if (validCsvs.isEmpty) return const SizedBox.shrink();

    // Ensure selected CSV is actually available
    if (availableCsvs[_selectedCsvType] == null) {
      _selectedCsvType = validCsvs.first.key;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildSectionHeader(theme, 'Detailed Reports Explorer', icon: Icons.analytics_outlined),
        const SizedBox(height: 12),
        
        // Report Selector
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            children: validCsvs.map((e) {
              final isSelected = _selectedCsvType == e.key;
              return Padding(
                padding: const EdgeInsets.only(right: 8.0),
                child: ChoiceChip(
                  label: Text(_formatCsvLabel(e.key)),
                  selected: isSelected,
                  onSelected: (selected) {
                    if (selected) setState(() => _selectedCsvType = e.key);
                  },
                ),
              );
            }).toList(),
          ),
        ),
        
        const SizedBox(height: 16),
        
        // Data Table
        CsvDataTableWidget(
          csvUrl: availableCsvs[_selectedCsvType]!,
          title: _formatCsvTitle(_selectedCsvType),
        ),
      ],
    );
  }

  String _formatCsvLabel(String type) {
    switch (type) {
      case 'attendance': return 'Attendance';
      case 'activity': return 'Activity';
      case 'speech': return 'Speech';
      case 'timeline': return 'Seating';
      case 'events': return 'Events';
      default: return type;
    }
  }

  String _formatCsvTitle(String type) {
    switch (type) {
      case 'attendance': return 'Student Attendance & Attention Report';
      case 'activity': return 'Person-Wise Activity Summary';
      case 'speech': return 'Class-Related Speech Segments';
      case 'timeline': return 'Student Seating Occupation Timeline';
      case 'events': return 'Timeline of Significant Classroom Events';
      default: return 'Report Data';
    }
  }

  // ── Live Stats Section ────────────────────────────────────────────────────

  Widget _buildLiveStatsSection(ThemeData theme, ColorScheme colorScheme) {
    if (_parsingCsv) {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            const SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            const SizedBox(width: 12),
            Text('Parsing attendance data from CSV…',
                style: theme.textTheme.bodySmall),
          ],
        ),
      );
    }

    if (_parseError != null) {
      return Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.orange.withAlpha(20),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: Colors.orange.withAlpha(60)),
        ),
        child: Row(
          children: [
            const Icon(Icons.warning_amber_rounded,
                color: Colors.orange, size: 18),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                'Could not parse CSV stats. Download the file to view manually.',
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: Colors.orange.shade700),
              ),
            ),
          ],
        ),
      );
    }

    if (_liveData == null) return const SizedBox.shrink();

    final d = _liveData!;
    final pct = d.attendancePercentage;
    final pctColor = pct >= 75
        ? Colors.green
        : pct >= 50
            ? Colors.orange
            : Colors.red;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Section header with "Live from CSV" badge
        Row(
          children: [
            Text('Attendance Summary',
                style: theme.textTheme.titleMedium
                    ?.copyWith(fontWeight: FontWeight.w600)),
            const SizedBox(width: 8),
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.green.withAlpha(25),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.green.withAlpha(60)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.check_circle_outline_rounded,
                      size: 10, color: Colors.green),
                  const SizedBox(width: 4),
                  Text('Live from CSV',
                      style: TextStyle(
                          fontSize: 10,
                          color: Colors.green.shade700,
                          fontWeight: FontWeight.w600)),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),

        // Stat cards row
        Row(
          children: [
            _statCard(theme, '${d.presentCount}', 'Present',
                Colors.green, Icons.check_circle_rounded),
            const SizedBox(width: 10),
            _statCard(theme, '${d.absentCount}', 'Absent',
                Colors.red, Icons.cancel_rounded),
            const SizedBox(width: 10),
            _statCard(theme, '${d.totalStudents}', 'Total',
                Colors.blue, Icons.people_rounded),
            const SizedBox(width: 10),
            _statCard(theme, '${pct.toStringAsFixed(0)}%', 'Rate',
                pctColor, Icons.pie_chart_rounded),
          ],
        ),

        // Attention & Activity row (if available)
        if (d.averageAttention > 0 || d.sittingCount > 0) ...[
          const SizedBox(height: 10),
          Row(
            children: [
              if (d.averageAttention > 0)
                Expanded(
                  child: _metricTile(
                    theme,
                    colorScheme,
                    icon: Icons.psychology_rounded,
                    color: Colors.purple,
                    label: 'Avg Attention',
                    value: '${d.averageAttention.toStringAsFixed(1)}%',
                  ),
                ),
              if (d.averageAttention > 0 && d.sittingCount > 0)
                const SizedBox(width: 10),
              if (d.sittingCount > 0)
                Expanded(
                  child: _metricTile(
                    theme,
                    colorScheme,
                    icon: Icons.event_seat_rounded,
                    color: Colors.teal,
                    label: 'Sitting',
                    value: '${d.sittingCount}',
                  ),
                ),
              if (d.standingCount > 0) ...[
                const SizedBox(width: 10),
                Expanded(
                  child: _metricTile(
                    theme,
                    colorScheme,
                    icon: Icons.accessibility_new_rounded,
                    color: Colors.orange,
                    label: 'Standing',
                    value: '${d.standingCount}',
                  ),
                ),
              ],
            ],
          ),
        ],

        // Student list (if records available)
        if (d.records.isNotEmpty) ...[
          const SizedBox(height: 12),
          _buildStudentList(theme, colorScheme, d.records),
        ],
      ],
    );
  }

  Widget _statCard(ThemeData theme, String value, String label, Color color,
      IconData icon) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
        decoration: BoxDecoration(
          color: color.withAlpha(18),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withAlpha(50)),
        ),
        child: Column(
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(height: 4),
            Text(value,
                style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold, color: color)),
            Text(label,
                style: theme.textTheme.labelSmall
                    ?.copyWith(color: color, fontSize: 10)),
          ],
        ),
      ),
    );
  }

  Widget _metricTile(ThemeData theme, ColorScheme colorScheme,
      {required IconData icon,
      required Color color,
      required String label,
      required String value}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withAlpha(40)),
      ),
      child: Row(
        children: [
          Icon(icon, color: color, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(value,
                    style: theme.textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.bold, color: color)),
                Text(label,
                    style: theme.textTheme.labelSmall?.copyWith(
                        color: colorScheme.onSurfaceVariant, fontSize: 10)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStudentList(ThemeData theme, ColorScheme colorScheme,
      List<StudentRecord> records) {
    // Show max 8; offer expand if more
    const maxVisible = 8;
    final visible = records.take(maxVisible).toList();

    return Container(
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 12, 14, 6),
            child: Text('Students (${records.length})',
                style: theme.textTheme.titleSmall
                    ?.copyWith(fontWeight: FontWeight.w600)),
          ),
          const Divider(height: 1),
          ...visible.map((r) => ListTile(
                dense: true,
                leading: CircleAvatar(
                  radius: 14,
                  backgroundColor:
                      (r.isPresent ? Colors.green : Colors.red).withAlpha(25),
                  child: Text(
                    r.name.isNotEmpty ? r.name[0].toUpperCase() : '?',
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: r.isPresent ? Colors.green : Colors.red),
                  ),
                ),
                title: Text(r.name,
                    style: theme.textTheme.bodySmall
                        ?.copyWith(fontWeight: FontWeight.w500)),
                subtitle: r.studentId.isNotEmpty
                    ? Text(r.studentId,
                        style: theme.textTheme.labelSmall?.copyWith(
                            color: colorScheme.onSurfaceVariant))
                    : null,
                trailing: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (r.attentionScore > 0)
                      Text(
                        '${r.attentionScore.toStringAsFixed(0)}%',
                        style: TextStyle(
                            fontSize: 11,
                            color: colorScheme.onSurfaceVariant),
                      ),
                    const SizedBox(width: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 7, vertical: 2),
                      decoration: BoxDecoration(
                        color: (r.isPresent ? Colors.green : Colors.red)
                            .withAlpha(20),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        r.isPresent ? 'Present' : 'Absent',
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                          color:
                              r.isPresent ? Colors.green.shade700 : Colors.red.shade700,
                        ),
                      ),
                    ),
                  ],
                ),
              )),
          if (records.length > maxVisible)
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 4, 14, 10),
              child: Text(
                '+ ${records.length - maxVisible} more (download CSV for full list)',
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: colorScheme.onSurfaceVariant),
              ),
            ),
        ],
      ),
    );
  }

  // ── Header / Banner / Download tiles (unchanged) ──────────────────────────

  Widget _buildHeaderCard(
      ThemeData theme, ColorScheme colorScheme, bool isDemo) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: isDemo
              ? [Colors.orange.shade700, Colors.orange.shade400]
              : [Colors.green.shade700, Colors.green.shade500],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: (isDemo ? Colors.orange : Colors.green).withOpacity(0.3),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(
              isDemo ? Icons.science_rounded : Icons.check_circle_rounded,
              size: 32,
              color: Colors.white,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  isDemo ? 'Demo Results' : 'Analysis Complete!',
                  style: theme.textTheme.titleMedium?.copyWith(
                      color: Colors.white, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 4),
                Text(widget.videoName,
                    style: theme.textTheme.bodySmall
                        ?.copyWith(color: Colors.white.withOpacity(0.85)),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis),
                const SizedBox(height: 2),
                Text(widget.subject.name,
                    style: theme.textTheme.bodySmall
                        ?.copyWith(color: Colors.white.withOpacity(0.7))),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _showLogViewer() {
    final logText = _currentResult?.logText ?? 'No logs available.';
    final logUrl = _currentResult?.pipelineLogUrl;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.terminal, size: 20),
            const SizedBox(width: 8),
            const Text('Pipeline Technical Logs'),
          ],
        ),
        content: SizedBox(
          width: double.maxFinite,
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.black,
              borderRadius: BorderRadius.circular(8),
            ),
            child: SingleChildScrollView(
              child: Text(
                logText,
                style: const TextStyle(
                  color: Colors.greenAccent,
                  fontFamily: 'monospace',
                  fontSize: 12,
                ),
              ),
            ),
          ),
        ),
        actions: [
          if (logUrl != null)
            TextButton.icon(
              onPressed: () => html.window.open(logUrl, '_blank'),
              icon: const Icon(Icons.download),
              label: const Text('Download Log'),
            ),
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  Widget _buildDemoBanner(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.orange.withAlpha(25),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.orange.withAlpha(80)),
      ),
      child: Row(
        children: [
          const Icon(Icons.info_outline, color: Colors.orange, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'Demo mode — API was offline. Tap "Set API URL" on the upload screen and retry.',
              style: theme.textTheme.bodySmall
                  ?.copyWith(color: colorScheme.onSurfaceVariant),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDownloadTile(BuildContext context,
      {required IconData icon,
      required Color color,
      required String title,
      required String subtitle,
      required String? url,
      required String filename}) {
    final colorScheme = Theme.of(context).colorScheme;
    final theme = Theme.of(context);
    final isAvailable = url != null;

    return Container(
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        border:
            isAvailable ? Border.all(color: color.withOpacity(0.3)) : null,
      ),
      child: ListTile(
        leading: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: color.withOpacity(isAvailable ? 0.12 : 0.06),
            borderRadius: BorderRadius.circular(10),
          ),
          child:
              Icon(icon, color: isAvailable ? color : Colors.grey, size: 22),
        ),
        title: Text(title,
            style: theme.textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
                color: isAvailable ? null : colorScheme.onSurfaceVariant)),
        subtitle: Text(isAvailable ? subtitle : 'Not generated',
            style: theme.textTheme.bodySmall
                ?.copyWith(color: colorScheme.onSurfaceVariant)),
        trailing: isAvailable
            ? ElevatedButton.icon(
                onPressed: () => _downloadFile(url, filename),
                icon: const Icon(Icons.download_rounded, size: 18),
                label: const Text('Download'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: color,
                  foregroundColor: Colors.white,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  textStyle: const TextStyle(fontSize: 13),
                ),
              )
            : const Icon(Icons.remove_circle_outline, color: Colors.grey),
      ),
    );
  }

  void _downloadFile(String url, String filename) {
    html.AnchorElement(href: url)
      ..setAttribute('download', filename)
      ..click();
  }
}
