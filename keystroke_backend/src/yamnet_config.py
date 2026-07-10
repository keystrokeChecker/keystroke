TARGET_SAMPLE_RATE = 16000
WINDOW_SECONDS = 0.20
HOP_SECONDS = 0.05
CLASSIFIER_THRESHOLD = 0.40
SENSITIVITY_DELTA = 0.05
SMOOTHING_WINDOW = 2
MERGE_GAP_SECONDS = 0.12
GAP_THRESHOLD = 0.85

# Keys that produce no distinct keystroke sound and should be excluded from
# both training labels and ground-truth counts.
MODIFIER_KEYS = {
    "Key.shift", "Key.shift_r",
    "Key.ctrl", "Key.ctrl_r", "Key.ctrl_l",
    "Key.alt", "Key.alt_r", "Key.alt_l", "Key.alt_gr",
    "Key.cmd", "Key.cmd_r", "Key.cmd_l",
    "Key.caps_lock",
}
