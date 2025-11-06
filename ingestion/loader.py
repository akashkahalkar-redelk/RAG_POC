from langchain_community.document_loaders import DirectoryLoader
import os

def load_codebase():
    path = os.environ.get("PROJECT_PATH")
    if not path or path == "/":
        raise ValueError("PROJECT_PATH environment variable not set or is set to root. Please set it to the absolute path of your codebase.")

    loader = DirectoryLoader(
        path,
        glob="**/*.swift",
        show_progress=True,
    )
    return loader.load()