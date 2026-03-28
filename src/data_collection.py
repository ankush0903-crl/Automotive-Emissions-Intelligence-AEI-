import os
import zipfile
import subprocess
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Datasets
DATASETS = [
    {
        "name": "co2-emission-by-vehicles",
        "kaggle_path": "debajyotipodder/co2-emission-by-vehicles",
    },
    {
        "name": "fuel-consumption",
        "kaggle_path": "ahmettyilmazz/fuel-consumption",
    }
]

def download_dataset(kaggle_path, output_dir):
    """
    Downloads a dataset from Kaggle using the Kaggle CLI via subprocess.
    This ensures that it uses the kaggle API key configured on the system.
    """
    try:
        import sys
        kaggle_cmd = str(Path(sys.executable).parent / "kaggle")
        print(f"Downloading {kaggle_path} to {output_dir}...")
        # kaggle datasets download -d <dataset> -p <path> --unzip
        result = subprocess.run([
            kaggle_cmd, "datasets", "download", "-d", kaggle_path, "-p", str(output_dir), "--unzip"
        ], check=True, capture_output=True, text=True)
        print(f"Successfully downloaded {kaggle_path}.")
        # Optional: Print output if needed
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {kaggle_path}. Error:\n{e.stderr}")
        print("Please ensure your Kaggle API key is configured correctly in ~/.kaggle/kaggle.json")
    except FileNotFoundError:
        print("Kaggle CLI not found. Please ensure it is installed and in your PATH.")

def main():
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for ds in DATASETS:
        download_dataset(ds["kaggle_path"], DATA_DIR)
        
    print(f"Data collection complete. Datasets saved to {DATA_DIR}")
    
    # List files in raw directory
    print("\nFiles in raw directory:")
    for file in DATA_DIR.iterdir():
        print(f"- {file.name}")

if __name__ == "__main__":
    main()
