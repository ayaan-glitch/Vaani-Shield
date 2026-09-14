class EvidenceRecord {
  final String sessionId;
  final double riskScore;       // 0.0 - 1.0
  final String modelVersion;
  final int chunksAnalyzed;
  final String result;          // SAFE / SUSPICIOUS / HIGH_RISK
  final String evidenceHash;    // SHA-256 stub
  final String blockchainStatus; // READY / NOT CONNECTED
  final DateTime timestamp;

  const EvidenceRecord({
    required this.sessionId,
    required this.riskScore,
    required this.modelVersion,
    required this.chunksAnalyzed,
    required this.result,
    required this.evidenceHash,
    required this.blockchainStatus,
    required this.timestamp,
  });
}
