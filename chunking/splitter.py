from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def chunk_documents(documents):
    # Smaller chunk size for more focused embeddings
    chunk_size = 1024
    chunk_overlap = 200

    swift_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.SWIFT, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    all_chunks = []
    for doc in documents:
        file_path = doc.metadata.get("source", "")
        if file_path.endswith(".swift"):
            chunks = swift_splitter.split_documents([doc])
            all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from Swift files")
    return all_chunks