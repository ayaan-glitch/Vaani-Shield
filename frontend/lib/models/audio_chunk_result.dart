class AudioChunkResult {
  final int chunkIndex;
  final double probability; // 0.0 - 1.0
  final DateTime timestamp;

  const AudioChunkResult({
    required this.chunkIndex,
    required this.probability,
    required this.timestamp,
  });
}
