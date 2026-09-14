import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../core/theme/app_theme.dart';
import '../../core/constants/app_constants.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _demoMode = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _demoMode = prefs.getBool(AppConstants.demoModeKey) ?? true;
    });
  }

  Future<void> _save(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(AppConstants.demoModeKey, value);
    setState(() => _demoMode = value);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(title: const Text('SETTINGS')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            // ── Demo Mode ────────────────────────────────────────────────
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              decoration: BoxDecoration(
                color: AppColors.card,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
              ),
              child: SwitchListTile(
                title: const Text('Demo Mode',
                    style: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w600)),
                subtitle: Text(
                  _demoMode
                      ? 'Using simulated detection results. Results are NOT real.'
                      : 'Will attempt to use real ONNX model (not yet available).',
                  style: const TextStyle(color: AppColors.textSecondary, fontSize: 12),
                ),
                value: _demoMode,
                activeThumbColor: AppColors.accent,
                activeTrackColor: AppColors.accentDim,
                onChanged: _save,
              ),
            ),
            const SizedBox(height: 24),

            // ── About ────────────────────────────────────────────────────
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppColors.card,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('VAANI‑SHIELD',
                      style: TextStyle(
                          color: AppColors.accent,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 2,
                          fontSize: 15)),
                  SizedBox(height: 6),
                  Text(
                    'Version: 0.1.0 (Prototype)\n'
                    'UI Status: Complete\n'
                    'ML Model: Pending ONNX integration\n'
                    'Blockchain: Not connected',
                    style: TextStyle(
                        color: AppColors.textSecondary, fontSize: 12, height: 1.7),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
