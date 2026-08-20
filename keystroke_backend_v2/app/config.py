import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_DIR = os.path.join(BASE_DIR, "production")

MODEL_PATH = os.path.join(PROD_DIR, "model.joblib")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")
AUDIO_CONFIG_PATH = os.path.join(PROD_DIR, "audio_config.json")
FEATURE_CONFIG_PATH = os.path.join(PROD_DIR, "feature_config.json")
PREPROCESSING_CONFIG_PATH = os.path.join(PROD_DIR, "preprocessing_config.json")
THRESHOLDS_PATH = os.path.join(PROD_DIR, "thresholds.json")
METADATA_PATH = os.path.join(PROD_DIR, "model_metadata.json")
