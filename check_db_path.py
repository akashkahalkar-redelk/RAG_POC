import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

def get_persist_dir():
    path = os.environ.get("PROJECT_PATH")
    if not path:
        return "db"
    if path.startswith("~"):
        path = os.path.expanduser(path)
    
    # Create a unique hash for the project path
    path_hash = hashlib.md5(path.encode()).hexdigest()
    return f"db/chroma_{path_hash}"

print(f"Project Path: {os.environ.get('PROJECT_PATH')}")
print(f"Expected DB Path: {get_persist_dir()}")
