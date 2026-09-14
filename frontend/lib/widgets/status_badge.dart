import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';
import '../../models/detection_result.dart';

enum StatusBadgeState { active, inactive, analyzing, safe, suspicious, highRisk, insufficient }

class StatusBadge extends StatelessWidget {
  final String label;
  final StatusBadgeState state;

  const StatusBadge({super.key, required this.label, required this.state});

  static StatusBadgeState fromDetectionState(DetectionState s) {
    switch (s) {
      case DetectionState.ANALYZING:        return StatusBadgeState.analyzing;
      case DetectionState.SAFE:             return StatusBadgeState.safe;
      case DetectionState.SUSPICIOUS:       return StatusBadgeState.suspicious;
      case DetectionState.HIGH_RISK:        return StatusBadgeState.highRisk;
      case DetectionState.INSUFFICIENT_AUDIO: return StatusBadgeState.insufficient;
    }
  }

  Color _color() {
    switch (state) {
      case StatusBadgeState.active:       return AppColors.safe;
      case StatusBadgeState.inactive:     return AppColors.textSecondary;
      case StatusBadgeState.analyzing:    return AppColors.accent;
      case StatusBadgeState.safe:         return AppColors.safe;
      case StatusBadgeState.suspicious:   return AppColors.suspicious;
      case StatusBadgeState.highRisk:     return AppColors.highRisk;
      case StatusBadgeState.insufficient: return AppColors.insufficient;
    }
  }

  IconData _icon() {
    switch (state) {
      case StatusBadgeState.active:       return Icons.shield;
      case StatusBadgeState.inactive:     return Icons.shield_outlined;
      case StatusBadgeState.analyzing:    return Icons.hourglass_top_rounded;
      case StatusBadgeState.safe:         return Icons.check_circle_outline;
      case StatusBadgeState.suspicious:   return Icons.warning_amber_rounded;
      case StatusBadgeState.highRisk:     return Icons.error_outline;
      case StatusBadgeState.insufficient: return Icons.mic_off_outlined;
    }
  }

  @override
  Widget build(BuildContext context) {
    final color = _color();
    return Semantics(
      label: label,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
        decoration: BoxDecoration(
          color: color.withOpacity(0.12),
          border: Border.all(color: color.withOpacity(0.4)),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(_icon(), size: 14, color: color),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                color: color,
                fontWeight: FontWeight.w600,
                fontSize: 12,
                letterSpacing: 1.2,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
