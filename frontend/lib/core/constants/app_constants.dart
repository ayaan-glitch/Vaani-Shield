class AppConstants {
  // SharedPreferences key for demo mode toggle
  static const String demoModeKey = 'demo_mode_enabled';

  // Flutter↔Kotlin MethodChannel identifiers
  static const String methodChannelName = 'vaanishield/detection';
  static const String methodInitialize   = 'initialize';
  static const String methodAnalyzeChunk = 'analyzeChunk';
  static const String methodDispose      = 'dispose';

  // Model version labels
  static const String modelNotConnected = 'MODEL NOT CONNECTED';
  static const String demoModelVersion  = 'DEMO';

  // UI labels
  static const String appName = 'VAANI‑SHIELD';
  static const String appSubtitle = 'Real-Time Voice Clone Protection';

  // Risk thresholds
  static const double suspiciousThreshold = 0.40;
  static const double highRiskThreshold   = 0.70;

  // 5-second window size (chunks of 1 second each)
  static const int windowSize = 5;
}
