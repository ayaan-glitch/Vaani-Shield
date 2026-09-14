import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';
import '../../models/detection_result.dart';

class WarningScreen extends StatelessWidget {
  const WarningScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final result = ModalRoute.of(context)?.settings.arguments as DetectionResult?;
    final riskPct = result != null ? (result.probability * 100).toStringAsFixed(0) : '—';
    final confidence = result != null
        ? (result.probability >= 0.85 ? 'High' : result.probability >= 0.70 ? 'Medium' : 'Elevated')
        : '—';

    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // ── Warning Icon ──────────────────────────────────────────
              Container(
                width: 90,
                height: 90,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: AppColors.highRisk.withOpacity(0.12),
                  border: Border.all(color: AppColors.highRisk.withOpacity(0.5), width: 2),
                ),
                child: const Icon(Icons.warning_amber_rounded,
                    color: AppColors.highRisk, size: 48),
              ),
              const SizedBox(height: 24),

              const Text(
                '⚠ POSSIBLE VOICE CLONING DETECTED',
                style: TextStyle(
                    color: AppColors.highRisk,
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 1.5),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),

              // ── Risk Stats ────────────────────────────────────────────
              _InfoRow(label: 'Risk', value: '$riskPct%', color: AppColors.highRisk),
              _InfoRow(label: 'Confidence', value: confidence, color: AppColors.suspicious),
              const SizedBox(height: 20),

              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.card,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.border),
                ),
                child: const Text(
                  'Multiple suspicious voice segments detected.\n\nThis is a probabilistic assessment — not a definitive classification. '
                  'Possible synthetic or voice-cloned audio was indicated.',
                  style: TextStyle(
                      color: AppColors.textSecondary, fontSize: 13, height: 1.6),
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(height: 32),

              // ── Buttons ───────────────────────────────────────────────
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => Navigator.pop(context),
                      child: const Text('DISMISS'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.pop(context);
                        Navigator.pushNamed(context, '/evidence');
                      },
                      child: const Text('VIEW DETAILS'),
                    ),
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

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _InfoRow({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(color: AppColors.textSecondary, fontSize: 14)),
          Text(value,
              style: TextStyle(
                  color: color, fontWeight: FontWeight.w700, fontSize: 20)),
        ],
      ),
    );
  }
}
