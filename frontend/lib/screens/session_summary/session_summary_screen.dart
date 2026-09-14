import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_theme.dart';
import '../../providers/session_provider.dart';
import '../../widgets/session_stats.dart';

class SessionSummaryScreen extends StatelessWidget {
  const SessionSummaryScreen({super.key});

  String _fmt(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '${d.inHours > 0 ? '${d.inHours}:' : ''}$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    final sp = context.read<SessionProvider>();
    final result = sp.buildResult();
    final riskColor = result.finalRisk == 'HIGH'
        ? AppColors.highRisk
        : result.finalRisk == 'SUSPICIOUS'
            ? AppColors.suspicious
            : AppColors.safe;
    final isDemo = result.modelVersion == 'DEMO';

    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('SESSION SUMMARY'),
        leading: IconButton(
          icon: const Icon(Icons.home_outlined),
          onPressed: () => Navigator.pushNamedAndRemoveUntil(context, '/', (_) => false),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              // ── Result Banner ────────────────────────────────────────
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: riskColor.withOpacity(0.10),
                  border: Border.all(color: riskColor.withOpacity(0.4)),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  children: [
                    Text(
                      'FINAL RISK: ${result.finalRisk}',
                      style: TextStyle(
                          color: riskColor,
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 2),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Peak: ${(result.peakProbability * 100).toStringAsFixed(1)}%',
                      style: TextStyle(color: riskColor.withOpacity(0.7), fontSize: 14),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // ── Stats Card ────────────────────────────────────────────
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: AppColors.card,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: AppColors.border),
                ),
                child: Column(
                  children: [
                    SessionStats(label: 'Duration', value: _fmt(result.duration)),
                    const Divider(color: AppColors.border),
                    SessionStats(
                        label: 'Speech Analysed', value: _fmt(result.speechAnalyzed)),
                    const Divider(color: AppColors.border),
                    SessionStats(
                        label: 'Chunks Analysed', value: '${result.chunksAnalyzed}'),
                    const Divider(color: AppColors.border),
                    SessionStats(
                        label: 'Suspicious Chunks',
                        value: '${result.suspiciousChunks}',
                        valueColor: result.suspiciousChunks > 0 ? AppColors.suspicious : null),
                    const Divider(color: AppColors.border),
                    SessionStats(
                        label: 'Peak Spoof Prob.',
                        value: '${(result.peakProbability * 100).toStringAsFixed(1)}%',
                        valueColor: riskColor),
                    const Divider(color: AppColors.border),
                    SessionStats(label: 'Final Risk', value: result.finalRisk, valueColor: riskColor),
                    const Divider(color: AppColors.border),
                    SessionStats(
                        label: 'Model',
                        value: result.modelVersion,
                        valueColor: isDemo ? AppColors.suspicious : AppColors.safe),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // ── Privacy Note ──────────────────────────────────────────
              if (isDemo)
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: AppColors.suspicious.withOpacity(0.08),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppColors.suspicious.withOpacity(0.3)),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.science_outlined, color: AppColors.suspicious, size: 16),
                      SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          'This session used DEMO MODE. Results are simulated and do not represent real detection.',
                          style: TextStyle(color: AppColors.suspicious, fontSize: 12, height: 1.5),
                        ),
                      ),
                    ],
                  ),
                )
              else
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: AppColors.safe.withOpacity(0.08),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppColors.safe.withOpacity(0.3)),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.lock_outline, color: AppColors.safe, size: 16),
                      SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          'Raw audio was processed locally on this device.',
                          style: TextStyle(color: AppColors.safe, fontSize: 12, height: 1.5),
                        ),
                      ),
                    ],
                  ),
                ),
              const SizedBox(height: 24),

              // ── Action Buttons ────────────────────────────────────────
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => Navigator.pushNamedAndRemoveUntil(context, '/', (_) => false),
                      icon: const Icon(Icons.home_outlined),
                      label: const Text('HOME'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: () => Navigator.pushNamed(context, '/evidence'),
                      icon: const Icon(Icons.receipt_long_outlined),
                      label: const Text('EVIDENCE'),
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
