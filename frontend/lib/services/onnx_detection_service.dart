import '../models/detection_result.dart';
import 'detection_service.dart';

// ─────────────────────────────────────────────────────────────
// OnnxDetectionService – STUB
// Will be implemented once the ML team delivers the ONNX model.
//
// Production flow:
//   Flutter → MethodChannel("vaanishield/detection")
//     → Kotlin DetectionChannel
//     → OnnxRuntimeHelper
//     → model.onnx
//     → { probability, state, modelVersion, inferenceTimeMs }
// ─────────────────────────────────────────────────────────────
class OnnxDetectionService implements DetectionService {
  @override
  Future<void> initialize() {
    throw UnimplementedError(
      '[OnnxDetectionService] Not yet implemented. '
      'Awaiting the ONNX model and Kotlin native bridge.',
    );
  }

  @override
  Stream<DetectionResult> get results => throw UnimplementedError(
        '[OnnxDetectionService] Not yet implemented.',
      );

  @override
  Future<void> dispose() async {}
}
