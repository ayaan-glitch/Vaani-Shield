import 'dart:math';
import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';
import '../../models/evidence_record.dart';

// ─────────────────────────────────────────────────────────────
// EvidenceScreen – DEMO / NOT CONNECTED
// Shows mock evidence records. Blockchain is not yet integrated.
// ─────────────────────────────────────────────────────────────
class EvidenceScreen extends StatelessWidget {
  const EvidenceScreen({super.key});

  // Generate a plausible-looking demo record
  static EvidenceRecord _demoRecord() {
    final rng = Random();
    final id = 'VS-${1000 + rng.nextInt(8999)}';
    final hash = List.generate(8, (_) => rng.nextInt(256).toRadixString(16).padLeft(2, '0')).join('');
    return EvidenceRecord(
      sessionId: id,
      riskScore: 0.87,
      modelVersion: 'DEMO',
      chunksAnalyzed: 42,
      result: 'SUSPICIOUS',
      evidenceHash: '$hash...${hash.substring(0, 4)}',
      blockchainStatus: 'READY',
      timestamp: DateTime.now().subtract(const Duration(minutes: 5)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final record = _demoRecord();

    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(title: const Text('BLOCKCHAIN EVIDENCE')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            // ── Demo Banner ──────────────────────────────────────────────
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: AppColors.suspicious.withOpacity(0.12),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppColors.suspicious.withOpacity(0.4)),
              ),
              child: const Row(
                children: [
                  Icon(Icons.link_off, color: AppColors.suspicious, size: 16),
                  SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'DEMO / NOT CONNECTED — blockchain backend is not yet implemented.',
                      style: TextStyle(color: AppColors.suspicious, fontSize: 12),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // ── Evidence Card ────────────────────────────────────────────
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: AppColors.card,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppColors.border),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _EvidRow(label: 'SESSION ID', value: record.sessionId, valueColor: AppColors.accent),
                  const Divider(color: AppColors.border),
                  _EvidRow(
                      label: 'RISK SCORE',
                      value: '${(record.riskScore * 100).toStringAsFixed(0)}%',
                      valueColor: AppColors.highRisk),
                  const Divider(color: AppColors.border),
                  _EvidRow(label: 'MODEL', value: record.modelVersion, valueColor: AppColors.suspicious),
                  const Divider(color: AppColors.border),
                  _EvidRow(label: 'CHUNKS', value: '${record.chunksAnalyzed}'),
                  const Divider(color: AppColors.border),
                  _EvidRow(label: 'RESULT', value: record.result, valueColor: AppColors.suspicious),
                  const Divider(color: AppColors.border),
                  _EvidRow(
                      label: 'EVIDENCE HASH',
                      value: record.evidenceHash,
                      valueColor: AppColors.textSecondary),
                  const Divider(color: AppColors.border),
                  _EvidRow(
                      label: 'BLOCKCHAIN STATUS',
                      value: record.blockchainStatus,
                      valueColor: AppColors.safe),
                  const Divider(color: AppColors.border),
                  _EvidRow(
                      label: 'TIMESTAMP',
                      value: record.timestamp.toString().substring(0, 19)),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // ── Architecture Note ────────────────────────────────────────
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
                  Text('Evidence Architecture',
                      style: TextStyle(
                          color: AppColors.textSecondary,
                          fontSize: 12,
                          fontWeight: FontWeight.w600)),
                  SizedBox(height: 10),
                  Text(
                    'In production, the evidence record will contain:\n'
                    '• Session metadata\n'
                    '• Risk result and model version\n'
                    '• Timestamp\n'
                    '• Evidence hash (SHA-256)\n\n'
                    'Raw audio will NOT be stored on the blockchain.',
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

class _EvidRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;
  const _EvidRow({required this.label, required this.value, this.valueColor});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(
                  color: AppColors.textSecondary, fontSize: 11, letterSpacing: 1.0)),
          Flexible(
            child: Text(
              value,
              style: TextStyle(
                  color: valueColor ?? AppColors.textPrimary,
                  fontWeight: FontWeight.w700,
                  fontSize: 13),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}
