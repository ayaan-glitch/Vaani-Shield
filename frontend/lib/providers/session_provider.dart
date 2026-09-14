import 'package:flutter/foundation.dart';

import '../core/constants/app_constants.dart';
import '../models/session_result.dart';

class SessionProvider extends ChangeNotifier {
  DateTime? _startTime;
  DateTime? _endTime;
  int _chunksAnalyzed = 0;
  int _suspiciousChunks = 0;
  double _peakProbability = 0.0;
  String _modelVersion = AppConstants.demoModelVersion;

  DateTime? get startTime => _startTime;
  int get chunksAnalyzed => _chunksAnalyzed;
  int get suspiciousChunks => _suspiciousChunks;
  double get peakProbability => _peakProbability;

  Duration get elapsed => _startTime == null
      ? Duration.zero
      : (DateTime.now().difference(_startTime!));

  void startSession({String modelVersion = AppConstants.demoModelVersion}) {
    _startTime = DateTime.now();
    _endTime = null;
    _chunksAnalyzed = 0;
    _suspiciousChunks = 0;
    _peakProbability = 0.0;
    _modelVersion = modelVersion;
    notifyListeners();
  }

  void endSession() {
    _endTime = DateTime.now();
    notifyListeners();
  }

  void recordChunk({required double probability, required bool suspicious}) {
    _chunksAnalyzed++;
    if (suspicious) _suspiciousChunks++;
    if (probability > _peakProbability) _peakProbability = probability;
    notifyListeners();
  }

  SessionResult buildResult() {
    final end = _endTime ?? DateTime.now();
    final duration = _startTime == null ? Duration.zero : end.difference(_startTime!);
    return SessionResult(
      duration: duration,
      speechAnalyzed: Duration(seconds: _chunksAnalyzed),
      chunksAnalyzed: _chunksAnalyzed,
      suspiciousChunks: _suspiciousChunks,
      peakProbability: _peakProbability,
      finalRisk: _riskLabel(),
      modelVersion: _modelVersion,
    );
  }

  String _riskLabel() {
    if (_peakProbability >= AppConstants.highRiskThreshold) return 'HIGH';
    if (_peakProbability >= AppConstants.suspiciousThreshold) return 'SUSPICIOUS';
    return 'SAFE';
  }
}
