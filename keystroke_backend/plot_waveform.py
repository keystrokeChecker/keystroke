import matplotlib.pyplot as plt
import numpy as np
from onset_detector import detect_onsets
from tune_and_evaluate import load_ground_truth

name = "session_calibrated_test"
onsets, y, sr = detect_onsets(f"data/{name}.wav")
true_counts, gt_times = load_ground_truth(f"data/{name}_log.csv")

t = np.arange(len(y)) / sr

plt.figure(figsize=(16, 4))
plt.plot(t, y, linewidth=0.5, alpha=0.6, label="waveform")
plt.vlines(gt_times, -1, 1, colors='green', linewidth=2, label='ground truth (real key)')
plt.vlines(onsets, -1, 1, colors='red', linestyle='--', linewidth=1, label='detected onset')
plt.legend()
plt.xlabel("time (s)")
plt.title(name)
plt.tight_layout()
plt.savefig(f"{name}_waveform.png", dpi=150)
print(f"Saved to {name}_waveform.png")