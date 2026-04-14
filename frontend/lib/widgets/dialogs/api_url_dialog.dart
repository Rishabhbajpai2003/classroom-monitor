import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../services/url_config_service.dart';

class ApiUrlDialog extends StatefulWidget {
  const ApiUrlDialog({super.key});

  static Future<bool> show(BuildContext context) async {
    final saved = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (_) => const ApiUrlDialog(),
    );
    return saved ?? false;
  }

  @override
  State<ApiUrlDialog> createState() => _ApiUrlDialogState();
}

class _ApiUrlDialogState extends State<ApiUrlDialog> {
  final _controller = TextEditingController();
  final _service = UrlConfigService();

  _DialogState _state = _DialogState.idle;
  String _statusMessage = '';
  bool _isOnline = false;

  @override
  void initState() {
    super.initState();
    _controller.text = _service.currentUrl;
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _testAndSave() async {
    final url = _controller.text.trim();
    if (url.isEmpty) {
      return;
    }

    setState(() {
      _state = _DialogState.testing;
      _statusMessage = 'Testing backend connection...';
      _isOnline = false;
    });

    final result = await _service.validate(url);
    if (!mounted) {
      return;
    }

    setState(() {
      _isOnline = result.isOnline;
      _statusMessage = result.message;
      _state = result.isValid
          ? (result.isOnline ? _DialogState.success : _DialogState.warning)
          : _DialogState.error;
    });

    if (result.isValid) {
      _service.saveUrl(url);
    }
  }

  void _clear() {
    _service.clearUrl();
    setState(() {
      _controller.text = 'http://localhost:8000';
      _state = _DialogState.idle;
      _statusMessage = '';
      _isOnline = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 480),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: colorScheme.primaryContainer,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Icon(
                      Icons.cloud_sync_rounded,
                      color: colorScheme.primary,
                      size: 22,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Connect to Backend',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          'Point the app to the classroom-monitor FastAPI service',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    onPressed: () => Navigator.pop(context, false),
                    icon: const Icon(Icons.close_rounded),
                    padding: EdgeInsets.zero,
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, size: 16, color: colorScheme.primary),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Local development usually runs at http://localhost:8000.\nYou can also paste a remote backend URL here.',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: colorScheme.onSurfaceVariant,
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              Text(
                'Backend URL',
                style: theme.textTheme.labelLarge?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 8),
              TextField(
                controller: _controller,
                enabled: _state != _DialogState.testing,
                keyboardType: TextInputType.url,
                autocorrect: false,
                decoration: InputDecoration(
                  hintText: 'http://localhost:8000',
                  prefixIcon: const Icon(Icons.link_rounded),
                  suffixIcon: IconButton(
                    icon: const Icon(Icons.content_paste_rounded, size: 18),
                    tooltip: 'Paste from clipboard',
                    onPressed: () async {
                      final data = await Clipboard.getData(Clipboard.kTextPlain);
                      if (data?.text != null) {
                        _controller.text = data!.text!.trim();
                        setState(() {
                          _state = _DialogState.idle;
                          _statusMessage = '';
                          _isOnline = false;
                        });
                      }
                    },
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                onChanged: (_) => setState(() {
                  _state = _DialogState.idle;
                  _statusMessage = '';
                  _isOnline = false;
                }),
                onSubmitted: (_) => _testAndSave(),
              ),
              const SizedBox(height: 12),
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 250),
                child: _statusMessage.isEmpty
                    ? const SizedBox.shrink()
                    : _StatusBanner(
                        key: ValueKey(_statusMessage),
                        message: _statusMessage,
                        state: _state,
                      ),
              ),
              const SizedBox(height: 20),
              Row(
                children: [
                  TextButton.icon(
                    onPressed: _state == _DialogState.testing ? null : _clear,
                    icon: const Icon(Icons.clear_rounded, size: 16),
                    label: const Text('Reset'),
                    style: TextButton.styleFrom(
                      foregroundColor: colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const Spacer(),
                  TextButton(
                    onPressed: _state == _DialogState.testing
                        ? null
                        : () => Navigator.pop(context, false),
                    child: const Text('Cancel'),
                  ),
                  const SizedBox(width: 8),
                  FilledButton.icon(
                    onPressed: _state == _DialogState.testing ||
                            _controller.text.trim().isEmpty
                        ? null
                        : _isOnline
                            ? () => Navigator.pop(context, true)
                            : _testAndSave,
                    icon: _state == _DialogState.testing
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : Icon(_isOnline ? Icons.check_rounded : Icons.cloud_done),
                    label: Text(_isOnline ? 'Done' : 'Test & Save'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

enum _DialogState {
  idle,
  testing,
  success,
  warning,
  error,
}

class _StatusBanner extends StatelessWidget {
  final String message;
  final _DialogState state;

  const _StatusBanner({
    super.key,
    required this.message,
    required this.state,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    Color background;
    Color foreground;
    IconData icon;

    switch (state) {
      case _DialogState.success:
        background = Colors.green.withAlpha(16);
        foreground = Colors.green.shade700;
        icon = Icons.check_circle_outline;
        break;
      case _DialogState.warning:
        background = Colors.orange.withAlpha(16);
        foreground = Colors.orange.shade700;
        icon = Icons.warning_amber_rounded;
        break;
      case _DialogState.error:
        background = colorScheme.errorContainer;
        foreground = colorScheme.error;
        icon = Icons.error_outline;
        break;
      case _DialogState.testing:
        background = colorScheme.primaryContainer.withAlpha(32);
        foreground = colorScheme.primary;
        icon = Icons.sync;
        break;
      case _DialogState.idle:
        background = colorScheme.surfaceContainerHighest;
        foreground = colorScheme.onSurfaceVariant;
        icon = Icons.info_outline;
        break;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 18, color: foreground),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style: theme.textTheme.bodySmall?.copyWith(color: foreground),
            ),
          ),
        ],
      ),
    );
  }
}
