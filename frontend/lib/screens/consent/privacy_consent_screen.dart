import 'package:flutter/material.dart';
import '../../widgets/buttons/primary_button.dart';

/// Privacy and Consent Screen - Explains video monitoring and data collection
/// Allows users to accept or decline consent
class PrivacyConsentScreen extends StatefulWidget {
  final bool isInitialSetup;

  const PrivacyConsentScreen({
    super.key,
    this.isInitialSetup = false,
  });

  @override
  State<PrivacyConsentScreen> createState() => _PrivacyConsentScreenState();
}

class _PrivacyConsentScreenState extends State<PrivacyConsentScreen> {
  bool _hasReadPolicy = false;
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _onScroll() {
    if (_scrollController.position.pixels >=
        _scrollController.position.maxScrollExtent - 50) {
      if (!_hasReadPolicy) {
        setState(() => _hasReadPolicy = true);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Privacy & Consent'),
      ),
      body: Column(
        children: [
          // Scrollable content
          Expanded(
            child: SingleChildScrollView(
              controller: _scrollController,
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Header
                  _buildHeader(theme, colorScheme),
                  const SizedBox(height: 24),
                  
                  // Introduction
                  _buildSection(
                    theme,
                    colorScheme,
                    'Introduction',
                    'This application uses video recording and facial recognition technology to automate classroom attendance tracking. We are committed to protecting your privacy and handling your data responsibly.',
                  ),
                  
                  // What we collect
                  _buildSection(
                    theme,
                    colorScheme,
                    'What We Collect',
                    null,
                    bulletPoints: [
                      'Facial recognition data for attendance verification',
                      'Class video recordings for processing',
                      'Attendance records and timestamps',
                      'Basic profile information',
                    ],
                  ),
                  
                  // How we use data
                  _buildSection(
                    theme,
                    colorScheme,
                    'How We Use Your Data',
                    null,
                    bulletPoints: [
                      'Generate automated attendance records',
                      'Match faces with registered student profiles',
                      'Provide attendance analytics and reports',
                      'Improve recognition accuracy over time',
                    ],
                  ),
                  
                  // Data protection
                  _buildSection(
                    theme,
                    colorScheme,
                    'Data Protection',
                    null,
                    bulletPoints: [
                      'All data is encrypted in transit and at rest',
                      'Video recordings are deleted after processing',
                      'Facial data is stored securely and never shared',
                      'Access is restricted to authorized personnel only',
                    ],
                  ),
                  
                  // Your rights
                  _buildSection(
                    theme,
                    colorScheme,
                    'Your Rights',
                    null,
                    bulletPoints: [
                      'Request access to your personal data',
                      'Request deletion of your data',
                      'Opt out of video-based attendance tracking',
                      'Receive manual attendance as an alternative',
                    ],
                  ),
                  
                  // Important notice
                  _buildNotice(theme, colorScheme),
                  
                  const SizedBox(height: 24),
                  
                  // Scroll indicator
                  if (!_hasReadPolicy)
                    Center(
                      child: Column(
                        children: [
                          Icon(
                            Icons.keyboard_double_arrow_down_rounded,
                            color: colorScheme.onSurfaceVariant,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Scroll to read full policy',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: colorScheme.onSurfaceVariant,
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          ),
          
          // Action buttons
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: colorScheme.surface,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 10,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            child: Column(
              children: [
                // Consent checkbox
                Row(
                  children: [
                    Checkbox(
                      value: _hasReadPolicy,
                      onChanged: (value) {
                        setState(() => _hasReadPolicy = value ?? false);
                      },
                    ),
                    Expanded(
                      child: Text(
                        'I have read and understood the privacy policy',
                        style: theme.textTheme.bodySmall,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                // Buttons
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: () => _handleDecline(context),
                        style: OutlinedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                        ),
                        child: const Text('Decline'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      flex: 2,
                      child: PrimaryButton(
                        label: 'Accept',
                        onPressed: _hasReadPolicy
                            ? () => _handleAccept(context)
                            : null,
                        isFullWidth: true,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: colorScheme.primaryContainer.withOpacity(0.3),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: colorScheme.primaryContainer,
              shape: BoxShape.circle,
            ),
            child: Icon(
              Icons.shield_rounded,
              size: 40,
              color: colorScheme.primary,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Privacy & Data Protection',
            style: theme.textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            'Your privacy is important to us. Please review our data collection and usage practices.',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildSection(
    ThemeData theme,
    ColorScheme colorScheme,
    String title,
    String? content, {
    List<String>? bulletPoints,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 12),
          if (content != null)
            Text(
              content,
              style: theme.textTheme.bodyMedium?.copyWith(
                height: 1.6,
                color: colorScheme.onSurfaceVariant,
              ),
            ),
          if (bulletPoints != null)
            ...bulletPoints.map((point) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        margin: const EdgeInsets.only(top: 8, right: 12),
                        width: 6,
                        height: 6,
                        decoration: BoxDecoration(
                          color: colorScheme.primary,
                          shape: BoxShape.circle,
                        ),
                      ),
                      Expanded(
                        child: Text(
                          point,
                          style: theme.textTheme.bodyMedium?.copyWith(
                            height: 1.5,
                            color: colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ),
                    ],
                  ),
                )),
        ],
      ),
    );
  }

  Widget _buildNotice(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colorScheme.errorContainer.withOpacity(0.3),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: colorScheme.error.withOpacity(0.3),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            Icons.warning_amber_rounded,
            color: colorScheme.error,
            size: 24,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Important',
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: colorScheme.error,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Declining consent may result in manual attendance tracking. Contact your instructor for alternative options.',
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

  void _handleAccept(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Row(
          children: [
            Icon(Icons.check_circle, color: Colors.white),
            SizedBox(width: 12),
            Text('Consent accepted successfully'),
          ],
        ),
        backgroundColor: Colors.green,
        behavior: SnackBarBehavior.floating,
      ),
    );
    
    Navigator.pop(context);
  }

  void _handleDecline(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Decline Consent?'),
          content: const Text(
            'If you decline, video-based attendance tracking will be disabled for your account. You may need to use manual attendance methods.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.pop(context);
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Consent declined. Using manual attendance.'),
                    behavior: SnackBarBehavior.floating,
                  ),
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).colorScheme.error,
              ),
              child: const Text('Decline'),
            ),
          ],
        );
      },
    );
  }
}
