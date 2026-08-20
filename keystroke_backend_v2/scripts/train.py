import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training.train import train_production_model


def main():
    print("=== EXECUTING PRODUCTION MODEL TRAINING ===")
    data_dir = os.path.join(BASE_DIR, "data", "raw")
    prod_dir = os.path.join(BASE_DIR, "production")

    metadata = train_production_model(data_dir=data_dir, prod_dir=prod_dir, model_name="ExtraTrees", feature_set="D", normalization="P2")
    print(f"Production model training finished successfully. Metadata: {metadata}")

if __name__ == "__main__":
    main()
