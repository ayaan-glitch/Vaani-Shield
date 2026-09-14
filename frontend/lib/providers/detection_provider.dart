import 'dart:async';
import 'dart:collection';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../core/constants/app_constants.dart';
import '../models/detection_result.dart';
import '../services/detection_service.dart';
import '../services/mock_detection_service.dart';
import '../services/onnx_detection_service.dart';

class DetectionProvider extends ChangeNotifier {
  DetectionService? _service;
  StreamSubscription<DetectionResult>? _sub;

  DetectionResult? _current;
  bool _isRunning = false;
  String? _errorMessage;

  // 5‑second window for temporal aggregation
  final _window = Queue<DetectionResult>();

  DetectionResult? get currentResult => _current;
  bool get isRunning => _isRunning;
  String? get errorMessage => _errorMessage;

  /// Average probability over the last 5-second window.
  double get windowRisk {
    final valid = _window
        .where((r) => r.state != DetectionState.INSUFFICIENT_AUDIO &&
            r.state != DetectionState.ANALYZING)
        .toList();
    if (valid.isEmpty) return 0.0;
    return valid.map((r) => r.probability).reduce((a, b) => a + b) / valid.length;
  }

  int get windowCount => _window.length;

  Future<void> startDetection() async {
    _errorMessage = null;
    try {
      final prefs = await SharedPreferences.getInstance();
      final demoMode = prefs.getBool(AppConstants.demoModeKey) ?? true;

      _service = demoMode ? MockDetectionService() : OnnxDetectionService();
      await _service!.initialize();

      _sub = _service!.results.listen(
        (result) {
          _current = result;
          _window.addLast(result);
          if (_window.length > AppConstants.windowSize) _window.removeFirst();
          notifyListeners();
        },
        onError: (e) {
          _errorMessage = 'Detection error: $e';
          notifyListeners();
        },
      );
      _isRunning = true;
      notifyListeners();
    } catch (e) {
      _errorMessage = 'Failed to start detection: $e';
      _isRunning = false;
      notifyListeners();
    }
  }

  Future<void> stopDetection() async {
    await _sub?.cancel();
    _sub = null;
    await _service?.dispose();
    _service = null;
    _isRunning = false;
    _current = null;
    _window.clear();
    notifyListeners();
  }

  @override
  void dispose() {
    _sub?.cancel();
    _service?.dispose();
    super.dispose();
  }
}
