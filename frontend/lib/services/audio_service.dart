// ─────────────────────────────────────────────────────────────
// AudioService – abstract audio source layer
// The UI never cares which audio source is active.
// ─────────────────────────────────────────────────────────────

/// Marker interface for all audio sources.
abstract class AudioSource {
  String get label;
  Future<void> start();
  Future<void> stop();
}

/// Test source – for development only, feeds silence.
class TestAudioSource implements AudioSource {
  @override
  String get label => 'TEST SOURCE (silence)';

  @override
  Future<void> start() async {}

  @override
  Future<void> stop() async {}
}

/// Placeholder for real microphone input (prototype/dev only).
/// NOTE: microphone ≠ remote caller audio.
class MicrophoneAudioSource implements AudioSource {
  @override
  String get label => 'MICROPHONE (prototype)';

  @override
  Future<void> start() async {
    throw UnimplementedError('Microphone source not yet wired up.');
  }

  @override
  Future<void> stop() async {}
}

/// Placeholder for future VoIP/call audio interception.
class VoIPAudioSource implements AudioSource {
  @override
  String get label => 'VoIP CALL AUDIO';

  @override
  Future<void> start() async {
    throw UnimplementedError('VoIP audio source not yet implemented.');
  }

  @override
  Future<void> stop() async {}
}
