import 'package:flutter/material.dart';

import 'keystroke_home_page.dart';

void main() {
  runApp(const KeystrokeApp());
}

class KeystrokeApp extends StatelessWidget {
  const KeystrokeApp({super.key, this.home});

  final Widget? home;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Keystroke AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0C0E17),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF8B5CF6),
          secondary: Color(0xFF06B6D4),
          surface: Color(0xFF151828),
          error: Color(0xFFEF4444),
          onPrimary: Colors.white,
          onSurface: Color(0xFFF1F5F9),
        ),
        cardTheme: CardThemeData(
          color: const Color(0xFF16192B),
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
            side: const BorderSide(color: Color(0xFF262B46), width: 1),
          ),
        ),
        textTheme: ThemeData.dark().textTheme.apply(
          bodyColor: const Color(0xFFE2E8F0),
          displayColor: Colors.white,
        ),
        useMaterial3: true,
      ),
      home: home ?? const KeystrokeHomePage(),
    );
  }
}
