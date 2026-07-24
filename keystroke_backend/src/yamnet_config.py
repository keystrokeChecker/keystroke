"""
Default configuration constants for the YAMNet keystroke prediction pipeline.

These are used as default form-parameter values in the FastAPI /analyze endpoint.
"""

# YAMNet classifier confidence threshold (0.0 to 1.0).
# onsets with predicted probability below this value are filtered out.
CLASSIFIER_THRESHOLD = 0.5

# Onset detection sensitivity (lower → more sensitive, more detections).
SENSITIVITY_DELTA = 0.07

# Word boundary gap threshold in seconds.
# ML models pool features per word segment and benefit from larger gap thresholds (0.75s).
# Rule-based DSP counts raw onsets per segment and relies on smaller gap thresholds (0.40s).
GAP_THRESHOLD_ML = 0.75
GAP_THRESHOLD_RULE = 0.40
GAP_THRESHOLD = GAP_THRESHOLD_ML  # Alias for backward compatibility

# Minimum gap between consecutive onsets in seconds.
# Detections closer than this are merged into a single keystroke event.
MERGE_GAP_SECONDS = 0.06
