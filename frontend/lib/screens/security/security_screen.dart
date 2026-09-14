import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';

class SecurityScreen extends StatelessWidget {
  const SecurityScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(title: const Text('SECURITY & PRIVACY')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            // ── Hero ────────────────────────────────────────────────────
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    AppColors.accent.withOpacity(0.12),
                    AppColors.accentDim.withOpacity(0.06),
                  ],
                ),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.accent.withOpacity(0.3)),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.memory_outlined, color: AppColors.accent, size: 28),
                      SizedBox(width: 12),
                      Text(
                        'ON-DEVICE AI',
                        style: TextStyle(
                            color: AppColors.accent,
                            fontSize: 18,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 2),
                      ),
                    ],
                  ),
                  SizedBox(height: 12),
                  Text(
                    'VAANI‑SHIELD is designed to perform voice-clone analysis locally on the device, using an on-device ONNX model.',
                    style: TextStyle(
                        color: AppColors.textSecondary, fontSize: 13, height: 1.6),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // ── Feature List ─────────────────────────────────────────────
            const _FeatureItem(
                icon: Icons.check_circle_outline,
                title: 'Local Inference',
                desc: 'Detection runs on the device without sending raw audio to a remote server.'),
            const _FeatureItem(
                icon: Icons.check_circle_outline,
                title: 'No Raw Audio for Blockchain',
                desc: 'Only session metadata, risk scores and evidence hashes are recorded.'),
            const _FeatureItem(
                icon: Icons.check_circle_outline,
                title: 'Model Version Tracking',
                desc: 'The model version is embedded in every session record for auditability.'),
            const _FeatureItem(
                icon: Icons.check_circle_outline,
                title: 'Session Risk Tracking',
                desc: 'Aggregated risk data is retained per session for evidence purposes.'),
            const SizedBox(height: 24),

            // ── Disclaimer ───────────────────────────────────────────────
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppColors.card,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppColors.border),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.info_outline, color: AppColors.textSecondary, size: 16),
                      SizedBox(width: 8),
                      Text('Disclaimer',
                          style: TextStyle(
                              color: AppColors.textSecondary,
                              fontSize: 12,
                              fontWeight: FontWeight.w600)),
                    ],
                  ),
                  SizedBox(height: 10),
                  Text(
                    'The final privacy behaviour of this application depends on the actual Android audio pipeline implementation and the specific call technology used. '
                    'VAANI‑SHIELD cannot guarantee that audio never leaves the device, as that depends on the complete system integration.',
                    style: TextStyle(
                        color: AppColors.textSecondary, fontSize: 12, height: 1.6),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _FeatureItem extends StatelessWidget {
  final IconData icon;
  final String title;
  final String desc;
  const _FeatureItem({required this.icon, required this.title, required this.desc});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.card,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppColors.safe, size: 20),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(
                        color: AppColors.textPrimary,
                        fontWeight: FontWeight.w600,
                        fontSize: 13)),
                const SizedBox(height: 4),
                Text(desc,
                    style: const TextStyle(
                        color: AppColors.textSecondary, fontSize: 12, height: 1.5)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
