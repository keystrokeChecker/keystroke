"""
Step 1 (M1): Record audio + simultaneously log keystrokes with timestamps.

Run this while typing a known phrase. It saves:
  - data/session_<name>.wav      -> the raw audio
  - data/session_<name>_log.csv  -> timestamp, key, is_word_boundary

USAGE:
    python record_dataset.py --name session1 --duration 15

Then type your test phrase (e.g. "the project") into any window while it records.
Press SPACE between words as normal — that's what marks word boundaries.

NOTE ON CONSENT/ETHICS:
Only use this to record your own typing, with your own knowledge and consent.
Do not use this to capture another person's keystrokes without permission.
"""

import argparse
import csv
import os
import threading
import time
import wave

import pyaudio
from pynput import keyboard

SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


class KeyLogger:
    """Logs every keypress with a timestamp relative to recording start."""

    def __init__(self, start_time):
        self.start_time = start_time
        self.events = []  # list of (relative_time, key_str, is_boundary)
        self.listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key):
        t = time.time() - self.start_time
        try:
            key_str = key.char
            is_boundary = False
        except AttributeError:
            key_str = str(key)
            # space, enter, and tab are treated as word boundaries
            is_boundary = key in (keyboard.Key.space, keyboard.Key.enter, keyboard.Key.tab)
        self.events.append((t, key_str, is_boundary))

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()

    def save(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_sec", "key", "is_word_boundary"])
            for t, k, b in self.events:
                writer.writerow([f"{t:.4f}", k, b])


def record_audio(duration, wav_path, start_barrier):
    """Records raw audio for `duration` seconds and writes it to wav_path."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    start_barrier.wait()  # sync start with keylogger
    start_time = time.time()
    while time.time() - start_time < duration:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    wf = wave.open(wav_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Session name, e.g. session1")
    parser.add_argument("--duration", type=int, default=15, help="Recording duration in seconds")
    args = parser.parse_args()

    wav_path = os.path.join(DATA_DIR, f"{args.name}.wav")
    log_path = os.path.join(DATA_DIR, f"{args.name}_log.csv")

    barrier = threading.Barrier(2)

    print(f"Recording will start in 2 seconds. Type your phrase for {args.duration}s...")
    time.sleep(2)

    start_time_holder = {"t": None}

    def audio_thread_fn():
        record_audio(args.duration, wav_path, barrier)

    audio_thread = threading.Thread(target=audio_thread_fn)
    audio_thread.start()

    # Wait until audio thread is about to start, then start keylogger at same instant
    barrier.wait()
    kl = KeyLogger(start_time=time.time())
    kl.start()

    audio_thread.join()
    kl.stop()
    kl.save(log_path)

    print(f"Saved audio to: {wav_path}")
    print(f"Saved keylog to: {log_path}")
    print(f"Recorded {len(kl.events)} keypress events.")


if __name__ == "__main__":
    main()
