class SessionResult {
  final Duration duration;
  final Duration speechAnalyzed;
  final int chunksAnalyzed;
  final int suspiciousChunks;
  final double peakProbability; // 0.0 – 1.0
  final String finalRisk;       // SAFE, SUSPICIOUS, HIGH
  final String modelVersion;

  const SessionResult({
    required this.duration,
    required this.speechAnalyzed,
    required this.chunksAnalyzed,
    required this.suspiciousChunks,
    required this.peakProbability,
    required this.finalRisk,
    required this.modelVersion,
  });
}
