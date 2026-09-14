import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_theme.dart';
import '../../core/constants/app_constants.dart';
import '../../providers/detection_provider.dart';
import '../../providers/session_provider.dart';
import '../../widgets/status_badge.dart';
import '../../widgets/detection_card.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final dp = context.watch<DetectionProvider>();

    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('VAANI‑SHIELD'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () => Navigator.pushNamed(context, '/settings'),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: 8),
              // ── Logo / Title ──────────────────────────────────────────
              Container(
                width: 72,
                height: 72,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(colors: [
                    AppColors.accent.withOpacity(0.3),
                    AppColors.bg,
                  ]),
                  border: Border.all(color: AppColors.accent.withOpacity(0.4), width: 2),
                ),
                child: const Icon(Icons.security, color: AppColors.accent, size: 36),
              ),
              const SizedBox(height: 16),
              const Text(
                AppConstants.appName,
                style: TextStyle(
                  color: AppColors.accent,
                  fontSize: 26,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 4,
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                AppConstants.appSubtitle,
                style: TextStyle(color: AppColors.textSecondary, fontSize: 13, letterSpacing: 0.5),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 28),

              // ── Protection Status ─────────────────────────────────────
              StatusBadge(
                label: dp.isRunning ? 'PROTECTION ACTIVE' : '● PROTECTION OFF',
                state: dp.isRunning ? StatusBadgeState.active : StatusBadgeState.inactive,
              ),
              const SizedBox(height: 28),

              // ── Start Protection Button ───────────────────────────────
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: dp.isRunning
                      ? null
                      : () async {
                          context.read<SessionProvider>().startSession();
                          await context.read<DetectionProvider>().startDetection();
                          if (context.mounted) {
                            Navigator.pushNamed(context, '/monitoring');
                          }
                        },
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('START PROTECTION'),
                ),
              ),
              const SizedBox(height: 28),

              // ── Feature Cards ─────────────────────────────────────────
              GridView.count(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                crossAxisCount: 2,
                childAspectRatio: 1.5,
                children: [
                  DetectionCard(
                    title: 'AI Detection',
                    icon: Icons.memory_outlined,
                    subtitle: 'On-device model',
                    onTap: () => Navigator.pushNamed(context, '/security'),
                  ),
                  DetectionCard(
                    title: 'Privacy',
                    icon: Icons.lock_outline,
                    subtitle: 'No raw audio upload',
                    onTap: () => Navigator.pushNamed(context, '/security'),
                  ),
                  DetectionCard(
                    title: 'Security',
                    icon: Icons.shield_outlined,
                    subtitle: 'Risk assessment',
                    onTap: () => Navigator.pushNamed(context, '/security'),
                  ),
                  DetectionCard(
                    title: 'Blockchain Evidence',
                    icon: Icons.link_outlined,
                    subtitle: 'Session records',
                    onTap: () => Navigator.pushNamed(context, '/evidence'),
                  ),
                ],
              ),
              const SizedBox(height: 24),

              // ── Privacy Note ──────────────────────────────────────────
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: AppColors.card,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.border),
                ),
                child: const Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.info_outline, color: AppColors.accent, size: 16),
                    SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'Voice analysis is designed to run on-device. Raw audio should not be uploaded to a server for the core detection process.',
                        style: TextStyle(color: AppColors.textSecondary, fontSize: 12, height: 1.5),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
