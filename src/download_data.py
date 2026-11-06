from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import io

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# Official UCI dataset ZIP
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/179/secom.zip"
)

print("Downloading semiconductor dataset...")

try:
    import urllib.request

    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        zip_content = response.read()

    with ZipFile(io.BytesIO(zip_content)) as zip_file:
        for file_name in zip_file.namelist():

            # Extract only the files we need
            base_name = Path(file_name).name

            if base_name in [
                "secom.data",
                "secom_labels.data",
                "secom.names"
            ]:
                output_path = RAW_DIR / base_name

                with zip_file.open(file_name) as source:
                    output_path.write_bytes(source.read())

                print(f"Downloaded: {base_name}")

    required_files = [
        "secom.data",
        "secom_labels.data"
    ]

    missing_files = [
        name for name in required_files
        if not (RAW_DIR / name).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing dataset files: {missing_files}"
        )

    print("\nDataset download completed successfully!")

except Exception as error:
    print(f"\nDownload failed: {error}")
    print("Check your internet connection and try again.")