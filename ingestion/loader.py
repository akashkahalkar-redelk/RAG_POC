import glob
from langchain_community.document_loaders import TextLoader
import os

def load_codebase():
    path = os.environ.get("PROJECT_PATH")
    if path:
        path = os.path.expanduser(path)
    
    if not path or path == "/":
        raise ValueError("PROJECT_PATH environment variable not set or is set to root. Please set it to the absolute path of your codebase.")

    print(f"Scanning {path} for swift files...", flush=True)
    # Use glob to find all swift files
    all_files = glob.glob(os.path.join(path, "**/*.swift"), recursive=True)
    print(f"Found {len(all_files)} total .swift files.", flush=True)

    documents = []
    excluded_count = 0
    
    for file_path in all_files:
        # Exclusion logic
        if "Externals/northstar" in file_path or "Pods" in file_path:
            excluded_count += 1
            continue
            
        try:
            # Load individual file
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    print(f"Ingestion summary: Loaded {len(documents)} files. Excluded {excluded_count} files.", flush=True)
    return documents