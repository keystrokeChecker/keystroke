import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training.benchmark import ExperimentRunner


def main():
    print("=== STARTING KEYSTROKE MODEL BENCHMARK SUITE ===")
    data_dir = os.path.join(BASE_DIR, "data", "raw")
    output_dir = BASE_DIR

    runner = ExperimentRunner(data_dir=data_dir, output_dir=output_dir)
    df_results = runner.run_all_required_experiments()

    print("\nBenchmark Suite Completed!")
    print(df_results[["experiment_id", "model", "features", "val_macro_f1", "test_macro_f1", "test_accuracy"]])

if __name__ == "__main__":
    main()
