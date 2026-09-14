import '../models/evidence_record.dart';

// ─────────────────────────────────────────────────────────────
// EvidenceService – DEMO / NOT CONNECTED
// Stores evidence records locally in memory.
// Placeholder for future blockchain backend integration.
// ─────────────────────────────────────────────────────────────
class EvidenceService {
  final List<EvidenceRecord> _records = [];

  List<EvidenceRecord> get records => List.unmodifiable(_records);

  /// Add a new evidence record for a completed session.
  void addRecord(EvidenceRecord record) {
    _records.add(record);
  }

  void clear() => _records.clear();
}
