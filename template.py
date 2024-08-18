import os
from pathlib import Path

list_of_files = [
    "requirements.txt",
    "setup.py",
    "README.md",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/predict_pipeline.py",
    "src/pipeline/__init__.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py"
]

for file_path in list_of_files:

    file_path = Path(file_path)

    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, 'w') as f:
            pass
    else:
        print(f"Skipping {file_path} as it already exists and is not empty")