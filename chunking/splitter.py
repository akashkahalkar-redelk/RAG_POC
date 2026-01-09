from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import os

def chunk_documents(documents):
    # Smaller chunk size for more focused embeddings
    chunk_size = 1024
    chunk_overlap = 200

    # Language mapping for extensions
    extension_to_language = {
        ".swift": Language.SWIFT,
        ".h": Language.CPP,
        ".m": Language.CPP,
        ".mm": Language.CPP,
        ".cpp": Language.CPP,
        ".hpp": Language.CPP,
        ".c": Language.C,
    }

    # Cache splitters
    splitters = {}

    def get_splitter(extension):
        if extension not in extension_to_language:
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        lang = extension_to_language[extension]
        if lang not in splitters:
            splitters[lang] = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        return splitters[lang]

    all_chunks = []
    for doc in documents:
        file_path = doc.metadata.get("source", "")
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        splitter = get_splitter(ext)
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(documents)} source files.")
    return all_chunks