"""
Diagnostic script to check onset detection performance details (misses and extras).
"""

import argparse
import os
import numpy as np
from onset_detector import detect_onsets
from tune_and_evaluate import load_ground_truth

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def diagnose_session(name, tolerance=0.08):
    wav_path = os.path.join(DATA_DIR, f"{name}.wav")
    log_path = os.path.join(DATA_DIR, f"{name}_log.csv")

    if not os.path.exists(wav_path) or not os.path.exists(log_path):
        print(f"Error: Missing files for session {name}")
        return

    # 1. Run detect_onsets with the default parameters (using the new default min_gap_seconds=0.06)
    onsets, _, _ = detect_onsets(wav_path)
    # 2. Load ground truth
    _, gt_times = load_ground_truth(log_path)
    gt_times = np.array(gt_times)

    print(f"\n========================================================")
    print(f"  DIAGNOSING SESSION: {name}")
    print(f"  Detected onsets: {len(onsets)} | Ground truth: {len(gt_times)}")
    print(f"========================================================")

    # Find matched detections and ground truth
    matched_gt = set()
    matched_det = set()

    for i, det_t in enumerate(onsets):
        for j, gt_t in enumerate(gt_times):
            if j in matched_gt:
                continue
            if abs(det_t - gt_t) <= tolerance:
                matched_gt.add(j)
                matched_det.add(i)
                break

    # Identify Misses (ground-truth with no detection within tolerance)
    misses = []
    missed_gaps = []
    print("\n--- MISSES (Ground Truth without close detected onset) ---")
    for j, gt_t in enumerate(gt_times):
        if j not in matched_gt:
            # Calculate gap to nearest neighboring ground-truth timestamps
            prev_gt = gt_times[j - 1] if j > 0 else None
            next_gt = gt_times[j + 1] if j < len(gt_times) - 1 else None
            
            gap_prev = f"{gt_t - prev_gt:.3f}s" if prev_gt is not None else "N/A"
            gap_next = f"{next_gt - gt_t:.3f}s" if next_gt is not None else "N/A"
            
            print(f"  GT Miss at {gt_t:7.3f}s (Gap to Prev GT: {gap_prev}, Next GT: {gap_next})")
            misses.append(gt_t)

    # Calculate statistics for missed ground-truth timestamps gaps
    if len(misses) > 1:
        m_gaps = np.diff(misses)
        min_m_gap = np.min(m_gaps)
        mean_m_gap = np.mean(m_gaps)
        median_m_gap = np.median(m_gaps)
    else:
        min_m_gap = mean_m_gap = median_m_gap = 0.0

    # Identify Extras (detections with no ground-truth within tolerance)
    extras = []
    print("\n--- EXTRAS (Detected onsets without close ground truth) ---")
    for i, det_t in enumerate(onsets):
        if i not in matched_det:
            # Find nearest ground-truth
            dists = np.abs(gt_times - det_t)
            nearest_idx = np.argmin(dists)
            nearest_gt = gt_times[nearest_idx]
            dist_to_gt = det_t - nearest_gt
            print(f"  Extra detection at {det_t:7.3f}s (Nearest GT: {nearest_gt:7.3f}s, Diff: {dist_to_gt:+.3f}s)")
            extras.append((det_t, abs(dist_to_gt)))

    # Analyze if extras are near true keystrokes or random noise
    near_count = 0
    for det_t, dist in extras:
        if dist <= 0.30:
            near_count += 1
    
    total_extras = len(extras)
    clustering_msg = "N/A"
    if total_extras > 0:
        pct_near = (near_count / total_extras) * 100
        clustering_msg = f"{pct_near:.1f}% ({near_count}/{total_extras}) are within 300ms of a true keystroke"
        if pct_near >= 60.0:
            clustering_msg += " (Suggests double-detections / splits)"
        else:
            clustering_msg += " (Suggests random background noise)"

    print("\n--- SUMMARY STATISTICS ---")
    print(f"  Total Misses: {len(misses)}")
    print(f"  Total Extras: {len(extras)}")
    if len(misses) > 1:
        print(f"  Gaps between consecutive missed GT keys: min={min_m_gap:.3f}s, mean={mean_m_gap:.3f}s, median={median_m_gap:.3f}s")
    else:
        print("  Gaps between consecutive missed GT keys: N/A (fewer than 2 misses)")
    print(f"  Extra detections clustering analysis: {clustering_msg}")
    print(f"========================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Diagnose onset detection misses and extras.")
    parser.add_argument("--names", nargs="+", default=["session1", "session2", "session3"],
                        help="Session names to diagnose.")
    args = parser.parse_args()

    for name in args.names:
        diagnose_session(name)

if __name__ == "__main__":
    main()
