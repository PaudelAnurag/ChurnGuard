from pathlib import Path

directories = [
    
    "reports/logs",
    "reports/images",
    "reports/models",
    
    "data/raw",
    "data/processed",
    
    "notebooks"
]

files = [
    "README.md",
    "requirements.txt",
    "app.py",
    ".gitignore"
]

for dir_path in directories:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

for file_path in files:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not path.exists():
        path.touch()
        print(f" Created file: {file_path}")
    
    else:
        print(f"Already exists: {file_path}")
    