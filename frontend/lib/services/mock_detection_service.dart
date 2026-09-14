import 'dart:async';
import 'dart:math';

import '../core/constants/app_constants.dart';
import '../models/detection_result.dart';
import 'detection_service.dart';

// ─────────────────────────────────────────────────────────────
// MOCK IMPLEMENTATION – FOR UI DEVELOPMENT ONLY
// Never presents results as real voice‑clone detection.
// Clearly labelled: modelVersion = "DEMO"
// ─────────────────────────────────────────────────────────────
class MockDetectionService implements DetectionService {
  final _controller = StreamController<DetectionResult>.broadcast();
  Timer? _timer;
  int _tick = 0;
  final _rng = Random(42); // deterministic seed for reproducibility

  // Demo sequence: each entry is (state, probability)
  static const _sequence = [
    (DetectionState.ANALYZING, 0.0),
    (DetectionState.SAFE, 0.12),
    (DetectionState.SAFE, 0.18),
    (DetectionState.SAFE, 0.09),
    (DetectionState.INSUFFICIENT_AUDIO, 0.0),
    (DetectionState.SAFE, 0.14),
    (DetectionState.SUSPICIOUS, 0.47),
    (DetectionState.SUSPICIOUS, 0.61),
    (DetectionState.HIGH_RISK, 0.82),
    (DetectionState.HIGH_RISK, 0.77),
    (DetectionState.SUSPICIOUS, 0.55),
    (DetectionState.SAFE, 0.21),
    (DetectionState.INSUFFICIENT_AUDIO, 0.0),
    (DetectionState.SAFE, 0.11),
    (DetectionState.SAFE, 0.07),
  ];

  @override
  Future<void> initialize() async {
    _tick = 0;
    _timer = Timer.periodic(const Duration(seconds: 1), (_) => _emit());
  }

  void _emit() {
    final entry = _sequence[_tick % _sequence.length];
    _controller.add(DetectionResult(
      probability: entry.$2,
      state: entry.$1,
      modelVersion: AppConstants.demoModelVersion,
      inferenceTimeMs: 18 + _rng.nextInt(10),
    ));
    _tick++;
  }

  @override
  Stream<DetectionResult> get results => _controller.stream;

  @override
  Future<void> dispose() async {
    _timer?.cancel();
    _timer = null;
    await _controller.close();
  }
}
