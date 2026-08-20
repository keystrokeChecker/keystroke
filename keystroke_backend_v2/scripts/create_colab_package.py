import os
import zipfile
import sys

def make_colab_package():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_zip = os.path.join(os.path.dirname(base_dir), "keystroke_colab_package.zip")
    
    include_dirs = [
        "audio",
        "datasets",
        "evaluation",
        "features",
        "models",
        "training",
        "scripts",
        os.path.join("data", "raw")
    ]
    
    include_files = [
        "classes.json",
        "requirements.txt",
        "train_on_colab.ipynb",
        "README.md"
    ]
    
    print(f"Creating Colab archive: {output_zip}")
    file_count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add directories
        for d in include_dirs:
            full_dir = os.path.join(base_dir, d)
            if not os.path.exists(full_dir):
                print(f"Warning: directory not found: {full_dir}")
                continue
            for root, _, files in os.walk(full_dir):
                if "__pycache__" in root or ".pytest_cache" in root:
                    continue
                for f in files:
                    full_path = os.path.join(root, f)
                    arcname = os.path.join("keystroke_backend_v2", os.path.relpath(full_path, base_dir))
                    zf.write(full_path, arcname)
                    file_count += 1
                    
        # Add root files
        for f in include_files:
            full_path = os.path.join(base_dir, f)
            if os.path.exists(full_path):
                arcname = os.path.join("keystroke_backend_v2", f)
                zf.write(full_path, arcname)
                file_count += 1

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"Successfully created: {output_zip}")
    print(f"Total files packed: {file_count}")
    print(f"Total archive size: {size_mb:.2f} MB")

if __name__ == "__main__":
    make_colab_package()
