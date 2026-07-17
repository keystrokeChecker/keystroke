"""
Step 1 (M1): Record audio + simultaneously log keystrokes with timestamps.

Run this while typing a known phrase. It saves:
  - data/session_<name>.wav      -> the raw audio
  - data/session_<name>_log.csv  -> timestamp, key, is_word_boundary
  - data/session_<name>_meta.json -> metadata including timing offset and silence RMS

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
import json
import math
import os
import struct
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

    def __init__(self):
        self.start_time = None
        self.events = []  # list of (relative_time, key_str, is_boundary)
        self.listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key):
        t = time.perf_counter() - self.start_time if self.start_time is not None else 0.0
        try:
            key_str = key.char
            is_boundary = False
        except AttributeError:
            key_str = str(key)
            # space, enter, and tab are treated as word boundaries
            is_boundary = key in (keyboard.Key.space, keyboard.Key.enter, keyboard.Key.tab)
        self.events.append((t, key_str, is_boundary))

    def start(self, start_time):
        self.start_time = start_time
        self.listener.start()

    def stop(self):
        self.listener.stop()

    def save(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_sec", "key", "is_word_boundary"])
            for t, k, b in self.events:
                writer.writerow([f"{t:.4f}", k, b])


def calculate_rms(audio_data):
    """Calculate the Root Mean Square (RMS) amplitude of 16-bit PCM audio frames."""
    count = len(audio_data) / 2
    if count == 0:
        return 0.0
    format_str = f"{int(count)}h"
    shorts = struct.unpack(format_str, audio_data)
    sum_squares = sum(x * x for x in shorts)
    return math.sqrt(sum_squares / count)


def record_session(name, duration):
    wav_path = os.path.join(DATA_DIR, f"{name}.wav")
    log_path = os.path.join(DATA_DIR, f"{name}_log.csv")
    meta_path = os.path.join(DATA_DIR, f"{name}_meta.json")

    pa = pyaudio.PyAudio()

    # Query device sample rate support
    try:
        device_info = pa.get_default_input_device_info()
        default_rate = int(device_info.get("defaultSampleRate", 44100))
        device_index = int(device_info.get("index", 0))
    except Exception:
        print("Warning: Could not query default input device info. Assuming default settings.")
        default_rate = SAMPLE_RATE
        device_index = None

    sr = SAMPLE_RATE
    if default_rate != SAMPLE_RATE and device_index is not None:
        try:
            # Test if 44100Hz is supported
            if not pa.is_format_supported(SAMPLE_RATE, input_device=device_index, input_channels=CHANNELS, input_format=FORMAT):
                sr = default_rate
                print(f"Warning: 44100Hz not supported. Falling back to default rate of {sr}Hz.")
        except Exception:
            sr = default_rate
            print(f"Warning: Error checking format support. Falling back to default rate of {sr}Hz.")

    # Open PyAudio Stream
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sr,
        input=True,
        frames_per_buffer=CHUNK,
    )

    # 2. Silence Calibration: Record 1 second of ambient noise
    print("Calibrating ambient background noise (please remain quiet for 1s)...")
    calibration_frames = []
    calibration_samples = int(sr / CHUNK)
    for _ in range(max(1, calibration_samples)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        calibration_frames.append(data)
    
    ambient_bytes = b"".join(calibration_frames)
    ambient_rms = calculate_rms(ambient_bytes)
    print(f"Ambient noise calibration RMS: {ambient_rms:.2f}")
    if ambient_rms > 1000.0:
        print("Warning: High ambient noise level detected! The environment might be too noisy.")

    # Visual countdown
    print("Prepare to type...")
    for i in [3, 2, 1]:
        print(f" {i}...")
        time.sleep(1.0)
    print("Recording! Start typing now...")

    barrier = threading.Barrier(2)
    frames = []
    
    timing_info = {
        "audio_stream_start_perf": 0.0,
        "keylogger_start_perf": 0.0,
    }

    def audio_thread_fn():
        barrier.wait()
        # Precise timestamp when audio recording loop starts
        timing_info["audio_stream_start_perf"] = time.perf_counter()
        
        loop_start = time.perf_counter()
        while time.perf_counter() - loop_start < duration:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

    audio_thread = threading.Thread(target=audio_thread_fn)
    audio_thread.start()

    kl = KeyLogger()
    barrier.wait()
    # Precise timestamp when keylogger starts listening
    logger_start = time.perf_counter()
    timing_info["keylogger_start_perf"] = logger_start
    kl.start(logger_start)

    audio_thread.join()
    kl.stop()
    
    # Save audio stream
    stream.stop_stream()
    stream.close()
    
    wf = wave.open(wav_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(sr)
    wf.writeframes(b"".join(frames))
    wf.close()
    
    pa.terminate()

    # Save log file
    kl.save(log_path)

    # Save Metadata JSON
    sync_offset_ms = (timing_info["keylogger_start_perf"] - timing_info["audio_stream_start_perf"]) * 1000.0
    meta = {
        "session_name": name,
        "duration_seconds": duration,
        "sample_rate_hz": sr,
        "ambient_rms": ambient_rms,
        "audio_start_perf": timing_info["audio_stream_start_perf"],
        "keylogger_start_perf": timing_info["keylogger_start_perf"],
        "sync_offset_ms": sync_offset_ms,
        "total_keypresses": len(kl.events),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    # Print summary statistics
    print("\n---------------- RECORDING SUMMARY ----------------")
    print(f"  Session Name         : {name}")
    print(f"  Duration             : {duration}s")
    print(f"  Total Keypresses     : {len(kl.events)}")
    print(f"  Ambient Noise RMS    : {ambient_rms:.2f}")
    print(f"  Audio/Keylog Offset  : {sync_offset_ms:.3f}ms")
    
    if len(kl.events) > 1:
        gaps = [kl.events[i][0] - kl.events[i-1][0] for i in range(1, len(kl.events))]
        avg_gap = sum(gaps) / len(gaps)
        print(f"  Avg Inter-key Gap    : {avg_gap:.3f}s")
    else:
        print("  Avg Inter-key Gap    : N/A")
    print("---------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Session name, e.g. session1")
    parser.add_argument("--duration", type=int, default=15, help="Recording duration in seconds")
    parser.add_argument("--repeat", type=int, default=1, help="Number of short sessions to record back-to-back")
    args = parser.parse_args()

    if args.repeat > 1:
        for i in range(1, args.repeat + 1):
            session_name = f"{args.name}_auto_{i}"
            input(f"Ready to record session {i}/{args.repeat} ({session_name}). Press Enter when ready...")
            record_session(session_name, args.duration)
    else:
        record_session(args.name, args.duration)


if __name__ == "__main__":
    main()

