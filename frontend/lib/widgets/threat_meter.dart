import 'dart:math';
import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';

// ─────────────────────────────────────────────────────────────
// ThreatMeter – circular radial gauge
// value: 0.0 (safe) → 1.0 (maximum risk)
// Animates smoothly on value change.
// ─────────────────────────────────────────────────────────────
class ThreatMeter extends StatelessWidget {
  final double value; // 0.0 – 1.0
  final String label;
  const ThreatMeter({super.key, required this.value, required this.label});

  Color get _color {
    if (value >= 0.70) return AppColors.highRisk;
    if (value >= 0.40) return AppColors.suspicious;
    return AppColors.safe;
  }

  String get _riskLabel {
    if (value >= 0.70) return 'HIGH';
    if (value >= 0.40) return 'MEDIUM';
    return 'LOW';
  }

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: value),
      duration: const Duration(milliseconds: 600),
      curve: Curves.easeInOut,
      builder: (context, animated, _) {
        return SizedBox(
          width: 200,
          height: 200,
          child: Stack(
            alignment: Alignment.center,
            children: [
              CustomPaint(
                size: const Size(200, 200),
                painter: _GaugePainter(animated, _color),
              ),
              Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    '${(animated * 100).toStringAsFixed(0)}%',
                    style: TextStyle(
                      color: _color,
                      fontSize: 36,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    _riskLabel,
                    style: TextStyle(
                      color: _color,
                      fontSize: 13,
                      letterSpacing: 2.0,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    label,
                    style: const TextStyle(
                      color: AppColors.textSecondary,
                      fontSize: 10,
                      letterSpacing: 1.2,
                    ),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}

class _GaugePainter extends CustomPainter {
  final double value;
  final Color color;
  _GaugePainter(this.value, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2 - 12;
    const startAngle = 3 * pi / 4;
    const sweepFull = 3 * pi / 2;

    final trackPaint = Paint()
      ..color = AppColors.border
      ..strokeWidth = 14
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final fillPaint = Paint()
      ..shader = SweepGradient(
        startAngle: startAngle,
        endAngle: startAngle + sweepFull * value,
        colors: [color.withOpacity(0.6), color],
        tileMode: TileMode.clamp,
      ).createShader(Rect.fromCircle(center: center, radius: radius))
      ..strokeWidth = 14
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    // Track
    canvas.drawArc(Rect.fromCircle(center: center, radius: radius),
        startAngle, sweepFull, false, trackPaint);

    // Fill
    if (value > 0) {
      canvas.drawArc(Rect.fromCircle(center: center, radius: radius),
          startAngle, sweepFull * value, false, fillPaint);
    }

    // Glow
    if (value > 0) {
      final glowPaint = Paint()
        ..color = color.withOpacity(0.15)
        ..strokeWidth = 28
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6);
      canvas.drawArc(Rect.fromCircle(center: center, radius: radius),
          startAngle, sweepFull * value, false, glowPaint);
    }
  }

  @override
  bool shouldRepaint(_GaugePainter old) => old.value != value || old.color != color;
}
