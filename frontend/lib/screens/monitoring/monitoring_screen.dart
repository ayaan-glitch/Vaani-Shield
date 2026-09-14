import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_theme.dart';
import '../../core/constants/app_constants.dart';
import '../../models/detection_result.dart';
import '../../providers/detection_provider.dart';
import '../../providers/session_provider.dart';
import '../../widgets/status_badge.dart';
import '../../widgets/threat_meter.dart';
import '../../widgets/probability_graph.dart';

class MonitoringScreen extends StatefulWidget {
  const MonitoringScreen({super.key});

  @override
  State<MonitoringScreen> createState() => _MonitoringScreenState();
}

class _MonitoringScreenState extends State<MonitoringScreen> {
  final List<DetectionResult> _history = [];
  bool _warningShown = false;

  String _stateLabel(DetectionState s) {
    switch (s) {
      case DetectionState.ANALYZING:        return 'ANALYZING';
      case DetectionState.SAFE:             return 'SAFE';
      case DetectionState.SUSPICIOUS:       return 'SUSPICIOUS';
      case DetectionState.HIGH_RISK:        return 'HIGH RISK';
      case DetectionState.INSUFFICIENT_AUDIO: return 'INSUFFICIENT AUDIO';
    }
  }

  @override
  Widget build(BuildContext context) {
    final dp = context.watch<DetectionProvider>();
    final sp = context.watch<SessionProvider>();
    final result = dp.currentResult;

    // Record chunk stats
    if (result != null && result.state != DetectionState.INSUFFICIENT_AUDIO) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        sp.recordChunk(
          probability: result.probability,
          suspicious: result.state == DetectionState.SUSPICIOUS ||
              result.state == DetectionState.HIGH_RISK,
        );
      });
    }

    // Maintain history list
    if (result != null && (_history.isEmpty || _history.last != result)) {
      _history.add(result);
      if (_history.length > 60) _history.removeAt(0);
    }

    // Show warning when HIGH_RISK
    if (result != null &&
        result.state == DetectionState.HIGH_RISK &&
        !_warningShown) {
      _warningShown = true;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        Navigator.pushNamed(context, '/warning', arguments: result);
        Future.delayed(const Duration(seconds: 10), () => _warningShown = false);
      });
    }

    final stateLabel = result != null ? _stateLabel(result.state) : 'WAITING';
    final badgeState = result != null
        ? StatusBadge.fromDetectionState(result.state)
        : StatusBadgeState.analyzing;
    final prob = result?.probability ?? 0.0;
    final windowRisk = dp.windowRisk;
    final isDemo = result?.modelVersion == AppConstants.demoModelVersion;

    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('MONITORING'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new),
          onPressed: () async {
            sp.endSession();
            await dp.stopDetection();
            if (context.mounted) {
              Navigator.pushReplacementNamed(context, '/summary');
            }
          },
        ),
        actions: [
          if (isDemo)
            Container(
              margin: const EdgeInsets.only(right: 12),
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: AppColors.suspicious.withOpacity(0.2),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: AppColors.suspicious.withOpacity(0.5)),
              ),
              child: const Text(
                'DEMO MODE',
                style: TextStyle(
                    color: AppColors.suspicious, fontSize: 10, fontWeight: FontWeight.w700, letterSpacing: 1),
              ),
            ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              // ── Status Header ────────────────────────────────────────
              const StatusBadge(label: 'PROTECTION ACTIVE', state: StatusBadgeState.active),
              const SizedBox(height: 16),
              StatusBadge(label: stateLabel, state: badgeState),
              const SizedBox(height: 28),

              // ── Threat Meter ─────────────────────────────────────────
              ThreatMeter(value: prob, label: 'CURRENT CHUNK'),
              const SizedBox(height: 28),

              // ── 5-Second Window Stats ────────────────────────────────
              _GlassCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('5‑SECOND RISK WINDOW',
                        style: TextStyle(
                            color: AppColors.textSecondary, fontSize: 11, letterSpacing: 1.5)),
                    const SizedBox(height: 12),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _StatBlock(
                            label: 'WINDOW RISK',
                            value: '${(windowRisk * 100).toStringAsFixed(0)}%',
                            color: windowRisk >= AppConstants.highRiskThreshold
                                ? AppColors.highRisk
                                : windowRisk >= AppConstants.suspiciousThreshold
                                    ? AppColors.suspicious
                                    : AppColors.safe),
                        _StatBlock(
                            label: 'CHUNKS',
                            value: '${dp.windowCount} / ${AppConstants.windowSize}',
                            color: AppColors.textPrimary),
                        _StatBlock(
                            label: 'TOTAL',
                            value: '${sp.chunksAnalyzed}',
                            color: AppColors.textPrimary),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // ── Current Chunk Info ───────────────────────────────────
              _GlassCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('CURRENT CHUNK',
                        style: TextStyle(
                            color: AppColors.textSecondary, fontSize: 11, letterSpacing: 1.5)),
                    const SizedBox(height: 10),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _StatBlock(
                            label: 'PROBABILITY',
                            value: result?.state == DetectionState.INSUFFICIENT_AUDIO
                                ? 'N/A'
                                : '${(prob * 100).toStringAsFixed(0)}%',
                            color: AppColors.accent),
                        _StatBlock(
                            label: 'INFERENCE',
                            value: result != null ? '${result.inferenceTimeMs}ms' : '—',
                            color: AppColors.textPrimary),
                        _StatBlock(
                            label: 'MODEL',
                            value: result?.modelVersion ?? AppConstants.modelNotConnected,
                            color: isDemo ? AppColors.suspicious : AppColors.safe),
                      ],
                    ),
                    if (result?.state == DetectionState.INSUFFICIENT_AUDIO) ...[
                      const SizedBox(height: 12),
                      const Row(
                        children: [
                          Icon(Icons.mic_off_outlined, color: AppColors.insufficient, size: 16),
                          SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'Not enough usable speech for reliable analysis.',
                              style: TextStyle(color: AppColors.insufficient, fontSize: 12),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // ── Probability Graph ────────────────────────────────────
              _GlassCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('SPOOF PROBABILITY (LAST 60s)',
                            style: TextStyle(
                                color: AppColors.textSecondary, fontSize: 11, letterSpacing: 1.5)),
                        Text('TIME →',
                            style: TextStyle(
                                color: AppColors.textSecondary, fontSize: 9, letterSpacing: 1.0)),
                      ],
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      height: 160,
                      child: ProbabilityGraph(history: _history),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // ── Stop Button ──────────────────────────────────────────
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: () async {
                    sp.endSession();
                    await dp.stopDetection();
                    if (context.mounted) {
                      Navigator.pushReplacementNamed(context, '/summary');
                    }
                  },
                  icon: const Icon(Icons.stop_circle_outlined),
                  label: const Text('STOP PROTECTION'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppColors.highRisk,
                    side: const BorderSide(color: AppColors.highRisk),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _GlassCard extends StatelessWidget {
  final Widget child;
  const _GlassCard({required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.card,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: child,
    );
  }
}

class _StatBlock extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _StatBlock({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            style: const TextStyle(
                color: AppColors.textSecondary, fontSize: 9, letterSpacing: 1.2)),
        const SizedBox(height: 4),
        Text(value,
            style: TextStyle(
                color: color, fontWeight: FontWeight.w800, fontSize: 18)),
      ],
    );
  }
}
