import 'dart:async';
import 'dart:convert';
import 'dart:html' as html;

import 'package:flutter/material.dart';

import '../../models/subject.dart';
import '../../services/api_service.dart';
import '../../widgets/buttons/primary_button.dart';
import '../../widgets/dialogs/api_url_dialog.dart';
import 'analysis_results_screen.dart';

/// Uploads a classroom recording and tracks backend progress through
/// the monorepo FastAPI job endpoints.
class UploadVideoScreen extends StatefulWidget {
  final Subject subject;

  const UploadVideoScreen({
    super.key,
    required this.subject,
  });

  @override
  State<UploadVideoScreen> createState() => _UploadVideoScreenState();
}

class _UploadVideoScreenState extends State<UploadVideoScreen> {
  final ApiService _apiService = ApiService();

  html.File? _selectedFile;
  String? _selectedVideoName;
  String? _fileSizeDisplay;

  bool _isUploading = false;
  bool _isCheckingBackend = true;
  bool _backendAvailable = false;
  String? _errorMessage;
  String _statusMessage = '';
  double _uploadProgress = 0;
  double _analysisProgress = 0;
  String _analysisLabel = '';

  String _device = 'auto';
  String _cameraId = 'cam_01';
  bool _runAttention = true;
  bool _runActivity = true;
  bool _runSpeech = false;
  String _courseProfile = 'default';
  DateTime _selectedDate = DateTime.now();
  late final TextEditingController _cameraIdController;
  late final TextEditingController _topicController;

  @override
  void initState() {
    super.initState();
    _cameraIdController = TextEditingController(text: _cameraId);
    _topicController = TextEditingController(text: widget.subject.name);
    _checkBackendStatus();
  }

  @override
  void dispose() {
    _cameraIdController.dispose();
    _topicController.dispose();
    super.dispose();
  }

  Future<void> _checkBackendStatus() async {
    setState(() => _isCheckingBackend = true);
    try {
      _backendAvailable = await _apiService.healthCheck();
    } catch (_) {
      _backendAvailable = false;
    }
    if (mounted) {
      setState(() => _isCheckingBackend = false);
    }
  }

  void _selectVideo() {
    final uploadInput = html.FileUploadInputElement()
      ..accept = 'video/*'
      ..click();

    uploadInput.onChange.listen((_) {
      final files = uploadInput.files;
      if (files != null && files.isNotEmpty) {
        final file = files.first;
        if (mounted) {
          setState(() {
            _selectedFile = file;
            _selectedVideoName = file.name;
            _fileSizeDisplay = _formatBytes(file.size);
            _errorMessage = null;
            _statusMessage = '';
            _uploadProgress = 0;
            _analysisProgress = 0;
            _analysisLabel = '';
          });
        }
      }
    });
  }

  void _removeVideo() {
    setState(() {
      _selectedFile = null;
      _selectedVideoName = null;
      _fileSizeDisplay = null;
      _errorMessage = null;
      _statusMessage = '';
      _uploadProgress = 0;
      _analysisProgress = 0;
      _analysisLabel = '';
    });
  }

  String _formatBytes(int bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
      return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
    }
    if (bytes >= 1024 * 1024) {
      return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
    }
    return '${(bytes / 1024).toStringAsFixed(1)} KB';
  }

  Future<void> _uploadAndAnalyze() async {
    if (_selectedFile == null || _selectedVideoName == null) {
      return;
    }

    final isLarge = _selectedFile!.size > 100 * 1024 * 1024;
    setState(() {
      _isUploading = true;
      _errorMessage = null;
      _uploadProgress = 0;
      _analysisProgress = 0;
      _analysisLabel = '';
      _statusMessage = isLarge
          ? 'Uploading ${_fileSizeDisplay ?? "video"} to the backend. Keep this tab open.'
          : 'Uploading video to the backend.';
    });

    try {
      if (!_backendAvailable) {
        await Future.delayed(const Duration(seconds: 2));
        if (!mounted) {
          return;
        }
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => AnalysisResultsScreen(
              subject: widget.subject,
              videoName: _selectedVideoName!,
              result: null,
            ),
          ),
        );
        return;
      }

      final jobId = await _createJobViaXHR(
        _selectedFile!,
        onProgress: (value) {
          if (!mounted) {
            return;
          }
          setState(() => _uploadProgress = value);
        },
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _uploadProgress = 1.0;
        _analysisProgress = 0.02;
        _analysisLabel = 'Queued';
        _statusMessage = 'Upload finished. Analysis has started.';
      });

      final result = await _pollAnalysisJob(jobId);

      if (!mounted) {
        return;
      }

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => AnalysisResultsScreen(
            subject: widget.subject,
            videoName: _selectedVideoName!,
            result: result,
          ),
        ),
      );
    } catch (error) {
      if (mounted) {
        setState(() {
          _isUploading = false;
          _uploadProgress = 0;
          _analysisProgress = 0;
          _analysisLabel = '';
          _statusMessage = '';
          _errorMessage = 'Error: $error';
        });
      }
    }
  }

  Future<GradioAnalysisResult> _pollAnalysisJob(String jobId) async {
    final startedAt = DateTime.now();

    while (true) {
      final status = await _apiService.getAnalysisJobStatus(jobId);

      if (!mounted) {
        throw Exception('Upload screen was closed before analysis finished.');
      }

      setState(() {
        _analysisProgress = status.progress.clamp(0.0, 1.0);
        _analysisLabel =
            status.currentStep.isNotEmpty ? status.currentStep : 'Processing';
      });

      if (status.isCompleted) {
        if (status.result == null) {
          throw Exception('Analysis completed without a result payload.');
        }
        setState(() {
          _analysisProgress = 1.0;
          _analysisLabel = 'Completed';
        });
        return status.result!;
      }

      if (status.isFailed) {
        throw Exception(status.errorMessage ?? 'Analysis job failed.');
      }

      if (DateTime.now().difference(startedAt) > const Duration(hours: 2)) {
        throw Exception('Analysis timed out after 2 hours.');
      }

      await Future.delayed(const Duration(seconds: 2));
    }
  }

  Future<String> _createJobViaXHR(
    html.File file, {
    required void Function(double value) onProgress,
  }) {
    final completer = _XHRCompleter<String>();
    final xhr = html.HttpRequest();
    final formData = html.FormData();

    formData.appendBlob('video', file, file.name);
    formData.append('device', _device);
    formData.append('camera_id', _cameraId);
    formData.append('run_attention', _runAttention.toString());
    formData.append('run_activity', _runActivity.toString());
    formData.append('run_speech', _runSpeech.toString());
    formData.append('course_profile', _courseProfile);
    formData.append(
      'class_date',
      '${_selectedDate.year}-${_selectedDate.month.toString().padLeft(2, '0')}-${_selectedDate.day.toString().padLeft(2, '0')}',
    );
    formData.append('class_topic', _topicController.text.trim());
    formData.append('subject_name', widget.subject.name);

    xhr.open('POST', '${ApiService.baseUrl}/api/analysis/jobs');
    xhr.setRequestHeader('Accept', 'application/json');

    xhr.upload.onProgress.listen((event) {
      if (event.lengthComputable) {
        onProgress(event.loaded! / event.total!);
      }
    });

    xhr.onLoad.listen((_) {
      if (xhr.status == 200 || xhr.status == 201) {
        try {
          final raw = xhr.responseText ?? '{}';
          final decoded = json.decode(raw);
          if (decoded is Map<String, dynamic> && decoded['job_id'] != null) {
            completer.complete(decoded['job_id'].toString());
          } else {
            completer.completeError(
              Exception('Unexpected response from backend: $raw'),
            );
          }
        } catch (error) {
          completer.completeError(error);
        }
      } else {
        completer.completeError(
          Exception('Upload failed (${xhr.status}): ${xhr.responseText}'),
        );
      }
    });

    xhr.onError.listen((_) {
      completer.completeError(
        Exception('Network error during upload. Check the backend URL.'),
      );
    });

    xhr.onTimeout.listen((_) {
      completer.completeError(Exception('Upload timed out.'));
    });

    xhr.send(formData);
    return completer.future;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Upload Class Video'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: _buildBackendStatus(colorScheme),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildSubjectInfo(theme, colorScheme),
            const SizedBox(height: 24),
            if (_errorMessage != null) ...[
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(Icons.error_outline, color: colorScheme.error),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _errorMessage!,
                        style: TextStyle(color: colorScheme.onErrorContainer),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
            ],
            _buildUploadArea(theme, colorScheme),
            const SizedBox(height: 24),
            if (_selectedFile != null && !_isUploading) ...[
              _buildAnalysisOptions(theme, colorScheme),
              const SizedBox(height: 24),
            ],
            if (_isUploading)
              _buildProgressSection(theme, colorScheme)
            else
              PrimaryButton(
                label: 'Upload & Analyze',
                isLoading: false,
                onPressed: _selectedFile != null ? _uploadAndAnalyze : null,
                icon: Icons.cloud_upload_rounded,
              ),
            const SizedBox(height: 16),
            _buildInstructions(theme, colorScheme),
          ],
        ),
      ),
    );
  }

  Widget _buildProgressSection(ThemeData theme, ColorScheme colorScheme) {
    final isAnalyzing = _uploadProgress >= 1.0;
    final uploadPct = (_uploadProgress * 100).toInt();
    final analysisPct = (_analysisProgress * 100).toInt();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _ProgressRow(
          label: 'Upload',
          pct: isAnalyzing ? 100 : uploadPct,
          done: isAnalyzing,
          color: colorScheme.primary,
          theme: theme,
        ),
        const SizedBox(height: 10),
        _ProgressRow(
          label: 'Analysis',
          pct: isAnalyzing ? analysisPct : 0,
          running: isAnalyzing && analysisPct < 100,
          done: analysisPct >= 100,
          color: Colors.orange,
          theme: theme,
          sublabel: isAnalyzing && _analysisLabel.isNotEmpty ? _analysisLabel : null,
        ),
        const SizedBox(height: 10),
        Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: Colors.orange.withAlpha(18),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            children: [
              const Icon(Icons.info_outline, size: 16, color: Colors.orange),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  isAnalyzing
                      ? 'The backend is processing your video. Keep this tab open until the result screen appears.'
                      : _statusMessage,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: Colors.orange.shade700,
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildBackendStatus(ColorScheme colorScheme) {
    if (_isCheckingBackend) {
      return const SizedBox(
        width: 20,
        height: 20,
        child: CircularProgressIndicator(strokeWidth: 2),
      );
    }

    return GestureDetector(
      onTap: () async {
        final saved = await ApiUrlDialog.show(context);
        if (saved && mounted) {
          _checkBackendStatus();
        }
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: _backendAvailable
              ? Colors.green.withAlpha(25)
              : Colors.orange.withAlpha(25),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              _backendAvailable ? Icons.cloud_done : Icons.cloud_off,
              size: 16,
              color: _backendAvailable ? Colors.green : Colors.orange,
            ),
            const SizedBox(width: 4),
            Text(
              _backendAvailable ? 'Backend Online' : 'Set API URL',
              style: TextStyle(
                fontSize: 12,
                color: _backendAvailable ? Colors.green : Colors.orange,
              ),
            ),
            const SizedBox(width: 2),
            Icon(
              Icons.edit_rounded,
              size: 11,
              color: _backendAvailable ? Colors.green : Colors.orange,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSubjectInfo(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(widget.subject.icon, size: 24, color: colorScheme.primary),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.subject.name,
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  widget.subject.code,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildUploadArea(ThemeData theme, ColorScheme colorScheme) {
    if (_selectedFile != null) {
      return Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Column(
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: colorScheme.primaryContainer,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                Icons.movie_rounded,
                size: 40,
                color: colorScheme.primary,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              _selectedVideoName!,
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w500,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 4),
            Text(
              _fileSizeDisplay ?? '',
              style: theme.textTheme.bodySmall?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
            if (_selectedFile!.size > 200 * 1024 * 1024) ...[
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.orange.withAlpha(25),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.warning_amber_rounded,
                      size: 14,
                      color: Colors.orange,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      'Large file: upload may take several minutes.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: Colors.orange.shade700,
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ),
            ],
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                OutlinedButton.icon(
                  onPressed: _isUploading ? null : _selectVideo,
                  icon: const Icon(Icons.swap_horiz_rounded, size: 18),
                  label: const Text('Change'),
                ),
                const SizedBox(width: 12),
                OutlinedButton.icon(
                  onPressed: _isUploading ? null : _removeVideo,
                  icon: const Icon(Icons.delete_outline_rounded, size: 18),
                  label: const Text('Remove'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: colorScheme.error,
                  ),
                ),
              ],
            ),
          ],
        ),
      );
    }

    return GestureDetector(
      onTap: _selectVideo,
      child: Container(
        height: 200,
        decoration: BoxDecoration(
          color: colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: colorScheme.outline.withAlpha(120),
            width: 2,
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 72,
                height: 72,
                decoration: BoxDecoration(
                  color: colorScheme.primaryContainer,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.videocam_rounded,
                  size: 36,
                  color: colorScheme.primary,
                ),
              ),
              const SizedBox(height: 16),
              Text(
                'Tap to Select Video',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Supported: MP4, MOV, AVI',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAnalysisOptions(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Analysis Options',
            style: theme.textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            decoration: const InputDecoration(
              labelText: 'Camera ID',
              hintText: 'cam_01',
              prefixIcon: Icon(Icons.videocam_outlined),
              isDense: true,
            ),
            controller: _cameraIdController,
            onChanged: (value) {
              setState(() => _cameraId = value.isEmpty ? 'cam_01' : value);
            },
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _device,
            decoration: const InputDecoration(
              labelText: 'Compute Device',
              prefixIcon: Icon(Icons.memory_outlined),
              isDense: true,
            ),
            items: const [
              DropdownMenuItem(value: 'auto', child: Text('Auto')),
              DropdownMenuItem(value: 'cuda', child: Text('GPU (CUDA)')),
              DropdownMenuItem(value: 'mps', child: Text('Apple MPS')),
              DropdownMenuItem(value: 'cpu', child: Text('CPU')),
            ],
            onChanged: (value) => setState(() => _device = value ?? 'auto'),
          ),
          const SizedBox(height: 8),
          CheckboxListTile(
            title: const Text('Attention & Attendance'),
            subtitle: const Text('Face recognition and attendance outputs'),
            value: _runAttention,
            onChanged: (value) => setState(() => _runAttention = value ?? true),
            contentPadding: EdgeInsets.zero,
            dense: true,
          ),
          CheckboxListTile(
            title: const Text('Activity Detection'),
            subtitle: const Text('Pose and activity tracking outputs'),
            value: _runActivity,
            onChanged: (value) => setState(() => _runActivity = value ?? true),
            contentPadding: EdgeInsets.zero,
            dense: true,
          ),
          CheckboxListTile(
            title: const Text('Speech Analysis'),
            subtitle: const Text('Speech topic outputs (slower)'),
            value: _runSpeech,
            onChanged: (value) => setState(() => _runSpeech = value ?? false),
            contentPadding: EdgeInsets.zero,
            dense: true,
          ),
          AnimatedOpacity(
            opacity: _runSpeech ? 1.0 : 0.4,
            duration: const Duration(milliseconds: 250),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 4),
                const Text(
                  'Course Profile',
                  style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
                ),
                const SizedBox(height: 4),
                DropdownButtonFormField<String>(
                  value: _courseProfile,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.school_outlined),
                    isDense: true,
                    contentPadding: EdgeInsets.symmetric(
                      vertical: 8,
                      horizontal: 12,
                    ),
                  ),
                  items: const [
                    DropdownMenuItem(value: 'default', child: Text('Default')),
                    DropdownMenuItem(value: 'math', child: Text('Mathematics')),
                    DropdownMenuItem(value: 'science', child: Text('Science')),
                    DropdownMenuItem(value: 'dl', child: Text('Deep Learning')),
                    DropdownMenuItem(value: 'cs', child: Text('Computer Science')),
                  ],
                  onChanged: _runSpeech
                      ? (value) => setState(
                            () => _courseProfile = value ?? 'default',
                          )
                      : null,
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          const Divider(),
          const SizedBox(height: 8),
          Text(
            'Session Details',
            style: theme.textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 12),
          InkWell(
            onTap: () async {
              final picked = await showDatePicker(
                context: context,
                initialDate: _selectedDate,
                firstDate: DateTime(2023),
                lastDate: DateTime.now(),
              );
              if (picked != null) {
                setState(() => _selectedDate = picked);
              }
            },
            child: InputDecorator(
              decoration: const InputDecoration(
                labelText: 'Class Date',
                prefixIcon: Icon(Icons.calendar_today_outlined),
                isDense: true,
              ),
              child: Text(
                '${_selectedDate.day}/${_selectedDate.month}/${_selectedDate.year}',
                style: theme.textTheme.bodyMedium,
              ),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _topicController,
            readOnly: true,
            decoration: InputDecoration(
              labelText: 'Class Topic / Title',
              prefixIcon: const Icon(Icons.topic_outlined),
              suffixIcon: Icon(
                Icons.lock_outline,
                size: 16,
                color: theme.colorScheme.primary.withAlpha(128),
              ),
              isDense: true,
              helperText: 'Locked to the current subject profile',
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInstructions(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colorScheme.primaryContainer.withAlpha(76),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            Icons.info_outline_rounded,
            size: 20,
            color: colorScheme.primary,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _backendAvailable
                  ? 'Videos upload directly to the classroom-monitor FastAPI backend. The result page will open automatically when the job completes.'
                  : 'Demo mode is active because the backend is offline. Configure the backend URL to run real analysis.',
              style: theme.textTheme.bodySmall?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ProgressRow extends StatelessWidget {
  final String label;
  final int pct;
  final bool done;
  final bool running;
  final Color color;
  final ThemeData theme;
  final String? sublabel;

  const _ProgressRow({
    required this.label,
    required this.pct,
    required this.color,
    required this.theme,
    this.done = false,
    this.running = false,
    this.sublabel,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = theme.colorScheme;
    final clampedPct = pct.clamp(0, 100);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            SizedBox(
              width: 96,
              child: Text(
                label,
                style: theme.textTheme.bodySmall?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: colorScheme.onSurfaceVariant,
                ),
              ),
            ),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: LinearProgressIndicator(
                  value: running ? null : clampedPct / 100.0,
                  minHeight: 8,
                  backgroundColor: colorScheme.surfaceContainerHighest,
                  valueColor: AlwaysStoppedAnimation<Color>(
                    done ? Colors.green : color,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),
            SizedBox(
              width: 40,
              child: done
                  ? const Icon(
                      Icons.check_circle_rounded,
                      size: 18,
                      color: Colors.green,
                    )
                  : Text(
                      '$clampedPct%',
                      textAlign: TextAlign.end,
                      style: theme.textTheme.labelMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: pct > 0 ? color : colorScheme.onSurfaceVariant,
                      ),
                    ),
            ),
          ],
        ),
        if (sublabel != null && sublabel!.isNotEmpty) ...[
          const SizedBox(height: 3),
          Padding(
            padding: const EdgeInsets.only(left: 96),
            child: Text(
              sublabel!,
              style: theme.textTheme.bodySmall?.copyWith(
                fontSize: 11,
                color: color.withAlpha(200),
              ),
            ),
          ),
        ],
      ],
    );
  }
}

class _XHRCompleter<T> {
  final Completer<T> _completer = Completer<T>();

  Future<T> get future => _completer.future;

  void complete(T value) {
    if (!_completer.isCompleted) {
      _completer.complete(value);
    }
  }

  void completeError(Object error) {
    if (!_completer.isCompleted) {
      _completer.completeError(error);
    }
  }
}
