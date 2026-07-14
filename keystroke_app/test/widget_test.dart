import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:keystroke_app/main.dart';

void main() {
  testWidgets('KeystrokeApp smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const KeystrokeApp());

    expect(find.text('Voice Recorder'), findsOneWidget);
    expect(find.text('Tap to record'), findsOneWidget);

    await tester.tap(find.byIcon(Icons.settings_outlined));
    await tester.pumpAndSettle();

    expect(find.text('Backend server URL'), findsOneWidget);
  });
}
