import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'core/theme/app_theme.dart';
import 'providers/detection_provider.dart';
import 'providers/session_provider.dart';
import 'screens/home/home_screen.dart';
import 'screens/monitoring/monitoring_screen.dart';
import 'screens/settings/settings_screen.dart';
import 'screens/warning/warning_screen.dart';
import 'screens/session_summary/session_summary_screen.dart';
import 'screens/security/security_screen.dart';
import 'screens/evidence/evidence_screen.dart';

void main() {
  runApp(const VaaniShieldApp());
}

class VaaniShieldApp extends StatelessWidget {
  const VaaniShieldApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => DetectionProvider()),
        ChangeNotifierProvider(create: (_) => SessionProvider()),
      ],
      child: MaterialApp(
        title: 'VAANI‑SHIELD',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.lightTheme,
        darkTheme: AppTheme.darkTheme,
        themeMode: ThemeMode.dark,
        initialRoute: '/',
        routes: {
          '/':           (_) => const HomeScreen(),
          '/monitoring': (_) => const MonitoringScreen(),
          '/settings':   (_) => const SettingsScreen(),
          '/warning':    (_) => const WarningScreen(),
          '/summary':    (_) => const SessionSummaryScreen(),
          '/security':   (_) => const SecurityScreen(),
          '/evidence':   (_) => const EvidenceScreen(),
        },
      ),
    );
  }
}
