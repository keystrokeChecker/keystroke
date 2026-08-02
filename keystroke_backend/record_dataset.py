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
import hashlib
import json
import math
import numpy as np
import os
import re
import statistics
import struct
import threading
import time
import wave

SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK = 1024

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


class KeyLogger:
    """Logs every keypress with a timestamp relative to recording start."""

    def __init__(self):
        from pynput import keyboard

        self.keyboard = keyboard
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
            is_boundary = key in (
                self.keyboard.Key.space,
                self.keyboard.Key.enter,
                self.keyboard.Key.tab,
            )
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


def valid_identifier(value):
    """Argparse validator for privacy-preserving IDs, not free-form labels."""
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "must be 1-64 characters using letters, numbers, '_' or '-'"
        )
    return value


def positive_duration(value):
    try:
        duration = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return duration


def derive_setup_id(capture_metadata):
    """Return a stable opaque ID for all leakage-relevant setup fields."""
    fields = (
        "keyboard_id",
        "microphone_id",
        "placement_id",
        "room_id",
        "typist_id",
        "capture_batch_id",
    )
    canonical = json.dumps(
        {field: capture_metadata[field] for field in fields},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "setup-" + hashlib.sha256(canonical).hexdigest()[:16]


def record_session(name, duration, capture_metadata):
    import pyaudio

    audio_format = pyaudio.paInt16
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
            if not pa.is_format_supported(SAMPLE_RATE, input_device=device_index, input_channels=CHANNELS, input_format=audio_format):
                sr = default_rate
                print(f"Warning: 44100Hz not supported. Falling back to default rate of {sr}Hz.")
        except Exception:
            sr = default_rate
            print(f"Warning: Error checking format support. Falling back to default rate of {sr}Hz.")

    # Open PyAudio Stream
    stream = pa.open(
        format=audio_format,
        channels=CHANNELS,
        rate=sr,
        input=True,
        frames_per_buffer=CHUNK,
    )

    # 2. Silence Calibration: Multi-window 3-second capture with median aggregation
    print("Calibrating ambient background noise (please remain quiet for 3s)...")
    
    # Warm up input buffer and allow hardware DC blocker / AGC to settle physically (2.0s real time)
    # We must read continuously to prevent buffer overflow and flush the initial hardware pop.
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        stream.read(CHUNK, exception_on_overflow=False)
        
    num_subwindows = 6
    subwindow_duration = 0.5  # seconds
    chunks_per_sub = max(1, int((sr * subwindow_duration) / CHUNK))
    
    sub_rms_list = []
    for _ in range(num_subwindows):
        sub_frames = []
        for _ in range(chunks_per_sub):
            data = stream.read(CHUNK, exception_on_overflow=False)
            sub_frames.append(data)
        sub_bytes = b"".join(sub_frames)
        sub_rms = calculate_rms(sub_bytes)
        sub_rms_list.append(sub_rms)
        
    # Steady-state room background noise level (25th percentile across sub-windows)
    # Transient spikes, mic pops, or rustles only INCREASE energy; 25th percentile isolates true ambient floor.
    ambient_rms = float(np.percentile(sub_rms_list, 25))
    print(f"Ambient noise calibration RMS (25th percentile of {num_subwindows} sub-windows): {ambient_rms:.2f}")

    # Sanity check for non-quiet environment or spurious noise spikes
    min_sub = min(sub_rms_list)
    max_sub = max(sub_rms_list)
    if min_sub > 0 and (max_sub / min_sub > 3.0) and max_sub > 5.0:
        print(f"Warning: Ambient noise fluctuated significantly during calibration (range: {min_sub:.2f} - {max_sub:.2f}). Ensure environment is quiet.")
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
    
    sample_width_bytes = pa.get_sample_size(audio_format)
    recorded_bytes = b"".join(frames)
    wf = wave.open(wav_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width_bytes)
    wf.setframerate(sr)
    wf.writeframes(recorded_bytes)
    wf.close()
    
    pa.terminate()

    # Save log file
    kl.save(log_path)

    # Save Metadata JSON
    sync_offset_ms = (timing_info["keylogger_start_perf"] - timing_info["audio_stream_start_perf"]) * 1000.0
    actual_duration = len(recorded_bytes) / (sr * CHANNELS * sample_width_bytes)
    setup_id = derive_setup_id(capture_metadata)
    meta = {
        "recording_id": name,
        "session_name": name,
        "duration_seconds": actual_duration,
        "requested_duration_seconds": duration,
        "sample_rate_hz": sr,
        "channels": CHANNELS,
        "sample_width_bytes": sample_width_bytes,
        "ambient_rms": ambient_rms,
        "audio_start_perf": timing_info["audio_stream_start_perf"],
        "keylogger_start_perf": timing_info["keylogger_start_perf"],
        "sync_offset_ms": sync_offset_ms,
        "total_keypresses": len(kl.events),
        "setup_id": setup_id,
        **capture_metadata,
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
    print(f"  Setup ID             : {setup_id}")
    
    if len(kl.events) > 1:
        gaps = [kl.events[i][0] - kl.events[i-1][0] for i in range(1, len(kl.events))]
        avg_gap = sum(gaps) / len(gaps)
        print(f"  Avg Inter-key Gap    : {avg_gap:.3f}s")
    else:
        print("  Avg Inter-key Gap    : N/A")
    print("---------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Record consented typing audio with leakage-safe metadata."
    )
    parser.add_argument("--name", required=True, type=valid_identifier, help="Opaque recording ID")
    parser.add_argument("--duration", type=positive_duration, default=15.0, help="Recording duration in seconds")
    parser.add_argument("--repeat", type=int, default=1, help="Number of short sessions to record back-to-back")
    parser.add_argument("--keyboard-id", required=True, type=valid_identifier)
    parser.add_argument("--microphone-id", required=True, type=valid_identifier)
    parser.add_argument("--placement-id", required=True, type=valid_identifier)
    parser.add_argument("--room-id", required=True, type=valid_identifier)
    parser.add_argument("--typist-id", required=True, type=valid_identifier)
    parser.add_argument("--capture-batch-id", required=True, type=valid_identifier)
    parser.add_argument(
        "--intended-split",
        required=True,
        choices=("train", "validation", "test", "diagnostic"),
    )
    parser.add_argument(
        "--fixture-type",
        required=True,
        choices=("typing", "silence", "non_keyboard"),
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    capture_metadata = {
        "keyboard_id": args.keyboard_id,
        "microphone_id": args.microphone_id,
        "placement_id": args.placement_id,
        "room_id": args.room_id,
        "typist_id": args.typist_id,
        "capture_batch_id": args.capture_batch_id,
        "intended_split": args.intended_split,
        "fixture_type": args.fixture_type,
    }

    if args.repeat > 1:
        for i in range(1, args.repeat + 1):
            session_name = f"{args.name}_auto_{i}"
            input(f"Ready to record session {i}/{args.repeat} ({session_name}). Press Enter when ready...")
            record_session(session_name, args.duration, capture_metadata)
    else:
        record_session(args.name, args.duration, capture_metadata)


if __name__ == "__main__":
    main()
