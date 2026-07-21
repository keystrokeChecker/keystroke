"""
Week 1 Deliverable Checker
===========================
Verifies all 5 Week 1 deliverables and produces waveform + onset plots
saved as PNG files in the data/ folder.

Run:
    python week1_check.py
"""

import os
import sys

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = {}

# ─────────────────────────────────────────────────────────
# Deliverable 1 – Android app that records audio
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Deliverable 1 – Android app that records audio")
print("=" * 60)

app_main = os.path.join(os.path.dirname(__file__), "..", "keystroke_app", "lib", "main.dart")
app_main = os.path.normpath(app_main)

if not os.path.exists(app_main):
    print(f"{FAIL} main.dart not found at {app_main}")
    results[1] = False
else:
    with open(app_main, encoding="utf-8") as f:
        src = f.read()

    checks = {
        "AudioRecorder instantiated":     "AudioRecorder()" in src,
        "WAV encoder configured":         "AudioEncoder.wav" in src,
        "Recording starts on tap":        "_recorder.start(" in src,
        "Recording stops and saves file": "_recorder.stop()" in src,
        "Files saved to documents dir":   "getApplicationDocumentsDirectory" in src,
        "Timestamped filenames":          "recording_" in src and ".wav" in src,
        "Microphone permission checked":  "hasPermission" in src,
    }

    all_ok = True
    for label, ok in checks.items():
        icon = PASS if ok else FAIL
        print(f"  {icon} {label}")
        if not ok:
            all_ok = False

    results[1] = all_ok


# ─────────────────────────────────────────────────────────
# Deliverable 2 – Audio files saved correctly
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Deliverable 2 – Audio files saved correctly")
print("=" * 60)

data_dir = os.path.join(os.path.dirname(__file__), "data")
wav_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

if not wav_files:
    print(f"{FAIL} No WAV files found in data/")
    results[2] = False
else:
    import soundfile as sf

    all_ok = True
    for fname in sorted(wav_files):
        path = os.path.join(data_dir, fname)
        try:
            data, sr = sf.read(path)
            duration = len(data) / sr
            size_kb = os.path.getsize(path) / 1024
            print(f"  {PASS} {fname}  —  {duration:.1f}s  |  {sr} Hz  |  {size_kb:.0f} KB  |  samples={len(data)}")
        except Exception as e:
            print(f"  {FAIL} {fname}  —  could not read: {e}")
            all_ok = False

    results[2] = all_ok


# ─────────────────────────────────────────────────────────
# Deliverable 3 – Python script that loads the recordings
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Deliverable 3 – Python script loads recordings")
print("=" * 60)

try:
    import soundfile as sf
    import numpy as np

    load_ok = True
    for fname in sorted(wav_files):
        path = os.path.join(data_dir, fname)
        y, sr = sf.read(path)
        y = y.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        peak = float(np.max(np.abs(y)))
        print(f"  {PASS} Loaded {fname}  —  dtype={y.dtype}  peak_amplitude={peak:.4f}")

    results[3] = load_ok
except Exception as e:
    print(f"  {FAIL} Failed to load recordings: {e}")
    results[3] = False


# ─────────────────────────────────────────────────────────
# Deliverable 4 – Basic waveform visualization
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Deliverable 4 – Waveform visualization")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use("Agg")   # headless — saves to PNG instead of opening a window
    import matplotlib.pyplot as plt
    import soundfile as sf
    import numpy as np

    saved = []
    for fname in sorted(wav_files):
        path = os.path.join(data_dir, fname)
        y, sr = sf.read(path)
        y = y.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        t = np.linspace(0, len(y) / sr, num=len(y))

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        fig.suptitle(f"Waveform — {fname}", fontsize=13, fontweight="bold")

        # Top: raw waveform
        axes[0].plot(t, y, color="#5C6BC0", linewidth=0.4, alpha=0.85)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_ylim(-1.05, 1.05)
        axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[0].set_title("Raw waveform", fontsize=10)

        # Bottom: RMS energy envelope
        frame_len = 1024
        hop = 512
        rms = []
        for i in range(0, len(y) - frame_len, hop):
            frame = y[i : i + frame_len]
            rms.append(float(np.sqrt(np.mean(frame ** 2))))
        t_rms = np.arange(len(rms)) * hop / sr
        axes[1].fill_between(t_rms, rms, color="#EF5350", alpha=0.75, linewidth=0)
        axes[1].set_ylabel("RMS Energy")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Energy envelope", fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(data_dir, fname.replace(".wav", "_waveform.png"))
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        saved.append(out_path)
        print(f"  {PASS} Saved waveform plot → {os.path.basename(out_path)}")

    results[4] = True
except Exception as e:
    print(f"  {FAIL} Visualization failed: {e}")
    results[4] = False


# ─────────────────────────────────────────────────────────
# Deliverable 5 – Detection and counting of keystroke events
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Deliverable 5 – Keystroke detection and counting")
print("=" * 60)

try:
    from onset_detector import detect_onsets
    from segmenter import segment_into_words, format_output
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf
    import numpy as np

    detect_ok = True
    for fname in sorted(wav_files):
        path = os.path.join(data_dir, fname)
        onsets, y, sr = detect_onsets(path, delta=0.07)
        counts, _ = segment_into_words(onsets, gap_threshold=0.4)
        total = sum(counts)

        print(f"  {PASS} {fname}  —  {len(onsets)} onsets detected  |  "
              f"{len(counts)} word(s)  |  per-word: {format_output(counts)}  |  total keystrokes: {total}")

        # Overlay onset markers on the waveform and save
        t = np.linspace(0, len(y) / sr, num=len(y))

        fig, ax = plt.subplots(figsize=(14, 3))
        fig.suptitle(f"Onset Detection — {fname}", fontsize=12, fontweight="bold")

        ax.plot(t, y, color="#5C6BC0", linewidth=0.3, alpha=0.7, label="waveform")
        for onset_t in onsets:
            ax.axvline(onset_t, color="#E53935", linewidth=0.7, alpha=0.7)

        # Invisible dummy line for legend
        ax.axvline(-1, color="#E53935", linewidth=1.2, label=f"onsets ({len(onsets)})")
        ax.set_xlim(0, len(y) / sr)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(data_dir, fname.replace(".wav", "_onsets.png"))
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"          → onset plot saved: {os.path.basename(out_path)}")

    results[5] = detect_ok
except Exception as e:
    print(f"  {FAIL} Detection failed: {e}")
    results[5] = False


# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("WEEK 1 SUMMARY")
print("=" * 60)

labels = {
    1: "Android app records audio",
    2: "Audio files saved correctly",
    3: "Python loads recordings",
    4: "Waveform visualization",
    5: "Keystroke detection & counting",
}

all_pass = True
for n, label in labels.items():
    ok = results.get(n, False)
    icon = PASS if ok else FAIL
    print(f"  {icon}  Deliverable {n}: {label}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"  {PASS} All Week 1 deliverables complete!")
else:
    print(f"  {WARN} Some deliverables need attention — see details above.")
print()
