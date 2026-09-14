enum DetectionState {
  ANALYZING,
  SAFE,
  SUSPICIOUS,
  HIGH_RISK,
  INSUFFICIENT_AUDIO,
}

class DetectionResult {
  final double probability; // 0.0 – 1.0
  final DetectionState state;
  final String modelVersion; // "DEMO" | "MODEL NOT CONNECTED" | actual version
  final int inferenceTimeMs;

  const DetectionResult({
    required this.probability,
    required this.state,
    required this.modelVersion,
    required this.inferenceTimeMs,
  });
}
