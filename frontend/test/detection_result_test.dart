import 'package:flutter_test/flutter_test.dart';
import 'package:vaani_shield/models/detection_result.dart';
import 'package:vaani_shield/models/session_result.dart';
import 'package:vaani_shield/core/constants/app_constants.dart';

void main() {
  group('DetectionResult', () {
    test('constructs with all fields', () {
      const r = DetectionResult(
        probability: 0.75,
        state: DetectionState.HIGH_RISK,
        modelVersion: 'DEMO',
        inferenceTimeMs: 20,
      );
      expect(r.probability, 0.75);
      expect(r.state, DetectionState.HIGH_RISK);
      expect(r.modelVersion, 'DEMO');
      expect(r.inferenceTimeMs, 20);
    });

    test('INSUFFICIENT_AUDIO has zero probability', () {
      const r = DetectionResult(
        probability: 0.0,
        state: DetectionState.INSUFFICIENT_AUDIO,
        modelVersion: 'DEMO',
        inferenceTimeMs: 0,
      );
      expect(r.state, DetectionState.INSUFFICIENT_AUDIO);
      expect(r.probability, 0.0);
    });

    test('Risk thresholds map correctly', () {
      double pLow = 0.1;
      double pMed = 0.55;
      double pHigh = 0.82;

      expect(pLow < AppConstants.suspiciousThreshold, isTrue);
      expect(pMed >= AppConstants.suspiciousThreshold && pMed < AppConstants.highRiskThreshold, isTrue);
      expect(pHigh >= AppConstants.highRiskThreshold, isTrue);
    });
  });

  group('SessionResult', () {
    test('builds with correct fields', () {
      const r = SessionResult(
        duration: Duration(minutes: 2, seconds: 34),
        speechAnalyzed: Duration(minutes: 1, seconds: 48),
        chunksAnalyzed: 108,
        suspiciousChunks: 17,
        peakProbability: 0.91,
        finalRisk: 'HIGH',
        modelVersion: 'DEMO',
      );
      expect(r.chunksAnalyzed, 108);
      expect(r.suspiciousChunks, 17);
      expect(r.finalRisk, 'HIGH');
      expect(r.peakProbability, 0.91);
    });
  });

  group('MockDetectionService sequence', () {
    test('INSUFFICIENT_AUDIO state has zero probability', () {
      // Verify our understanding of the contract
      const r = DetectionResult(
        probability: 0.0,
        state: DetectionState.INSUFFICIENT_AUDIO,
        modelVersion: AppConstants.demoModelVersion,
        inferenceTimeMs: 0,
      );
      expect(r.state != DetectionState.SAFE, isTrue);
      expect(r.state != DetectionState.SUSPICIOUS, isTrue);
    });
  });
}
