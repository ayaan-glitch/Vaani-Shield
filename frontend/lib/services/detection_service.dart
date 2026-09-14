import '../models/detection_result.dart';

// ─────────────────────────────────────────────────────────────
// DetectionService – abstract interface
// The UI always depends on this, never on a concrete impl.
// ─────────────────────────────────────────────────────────────
abstract class DetectionService {
  /// Initialise resources (load model, open audio source, etc.)
  Future<void> initialize();

  /// Continuous stream of results – one per analysed audio chunk.
  Stream<DetectionResult> get results;

  /// Release all resources.
  Future<void> dispose();
}
