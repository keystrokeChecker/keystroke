import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';

import 'recording_result.dart';

class RecordingsPage extends StatelessWidget {
  final List<RecordingResult> history;
  final AudioPlayer player;
  final Future<void> Function(int) onDelete;

  const RecordingsPage({
    super.key,
    required this.history,
    required this.player,
    required this.onDelete,
  });

  Future<void> _play(String path) async {
    try {
      await player.setFilePath(path);
      await player.play();
    } catch (e) {
      // Playback errors are non-fatal here; the parent screen can keep running.
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    if (history.isEmpty) {
      return const Center(child: Text('No recordings yet.'));
    }

    return ListView.builder(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
      itemCount: history.length,
      itemBuilder: (context, index) {
        final item = history[index];
        final fileName = item.filePath.split('/').last;

        return Dismissible(
          key: ValueKey(item.filePath),
          direction: DismissDirection.endToStart,
          background: Container(
            margin: const EdgeInsets.only(bottom: 10),
            decoration: BoxDecoration(
              color: Colors.redAccent,
              borderRadius: BorderRadius.circular(16),
            ),
            alignment: Alignment.centerRight,
            padding: const EdgeInsets.only(right: 24),
            child: const Icon(Icons.delete_outline, color: Colors.white, size: 28),
          ),
          onDismissed: (_) => onDelete(index),
          child: Card(
            margin: const EdgeInsets.only(bottom: 10),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: ListTile(
              contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              leading: Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.12),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.graphic_eq_rounded, color: Color(0xFF0188E2)),
              ),
              title: Text(item.formatted),
              subtitle: Text(
                '$fileName | ${item.method.toUpperCase()}',
                style: TextStyle(
                  color: isDark ? Colors.white60 : Colors.black54,
                ),
              ),
              trailing: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  IconButton(
                    icon: const Icon(Icons.play_arrow_rounded),
                    onPressed: () => _play(item.filePath),
                  ),
                  IconButton(
                    icon: const Icon(Icons.delete_outline),
                    onPressed: () => onDelete(index),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
