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
      title: 'Keystroke Analyzer',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: home ?? const KeystrokeHomePage(),
    );
  }
}
