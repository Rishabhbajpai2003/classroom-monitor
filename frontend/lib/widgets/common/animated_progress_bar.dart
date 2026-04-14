import 'package:flutter/material.dart';

/// Animated progress bar with gradient fill and smooth entrance animation
class AnimatedProgressBar extends StatelessWidget {
  final double value; // 0.0 to 1.0
  final double height;
  final Color? color;
  final Color? backgroundColor;
  final Gradient? gradient;
  final Duration duration;
  final BorderRadius? borderRadius;

  const AnimatedProgressBar({
    super.key,
    required this.value,
    this.height = 6,
    this.color,
    this.backgroundColor,
    this.gradient,
    this.duration = const Duration(milliseconds: 800),
    this.borderRadius,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final effectiveBorderRadius =
        borderRadius ?? BorderRadius.circular(height / 2);

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: value.clamp(0.0, 1.0)),
      duration: duration,
      curve: Curves.easeOutCubic,
      builder: (context, animatedValue, _) {
        return Container(
          height: height,
          decoration: BoxDecoration(
            color: backgroundColor ??
                colorScheme.surfaceContainerHighest.withOpacity(0.5),
            borderRadius: effectiveBorderRadius,
          ),
          child: Align(
            alignment: Alignment.centerLeft,
            child: FractionallySizedBox(
              widthFactor: animatedValue,
              child: Container(
                decoration: BoxDecoration(
                  color:
                      gradient == null ? (color ?? colorScheme.primary) : null,
                  gradient: gradient,
                  borderRadius: effectiveBorderRadius,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
