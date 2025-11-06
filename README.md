# RAG POC - Codebase Q&A Bot

This project is a Proof-of-Concept (POC) for a Retrieval-Augmented Generation (RAG) bot. It's designed to answer questions about a codebase by using a local Large Language Model (LLM) to understand and respond to queries based on the content of the source code.

## How it Works

The application uses a RAG pipeline to provide answers:

1.  **Ingestion:** It loads all `.swift` files from a specified project directory.
2.  **Chunking:** The code is split into smaller, manageable chunks.
3.  **Embedding:** The chunks are converted into vector embeddings using a local embedding model.
4.  **Vector Store:** The embeddings are stored in a local vector database (ChromaDB).
5.  **Retrieval:** When a question is asked, the application retrieves the most relevant code chunks from the vector store.
6.  **Generation:** The retrieved code and the original question are passed to a local LLM, which generates an answer.

This entire process runs locally, ensuring that your codebase remains private.

## Prerequisites

-   [Ollama](https://ollama.com/): For running the local LLM and embedding models.
-   Python 3.8+

## Setup

### 1. Install Ollama

Follow the instructions on the [Ollama website](https://ollama.com/) to download and install it for your operating system.

After installation, you need to pull the required models. Open your terminal and run:

```bash
ollama pull codellama:13b
ollama pull mxbai-embed-large
```

### 2. Project Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd RAG_POC
    ```

2.  **Create and activate a virtual environment:**
    -   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    -   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install dependencies:**
    Make sure you have a `requirements.txt` file with all the necessary packages. Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set the `PROJECT_PATH` environment variable:**
    This variable should point to the absolute path of the codebase you want to analyze.

    -   **macOS/Linux:**
        ```bash
        export PROJECT_PATH=/path/to/your/codebase
        ```
    -   **Windows (Command Prompt):**
        ```bash
        set PROJECT_PATH=C:\path\to\your\codebase
        ```
    -   **Windows (PowerShell):**
        ```bash
        $env:PROJECT_PATH="C:\path\to\your\codebase"
        ```
    **Note:** Do not set this to your root directory (`/`) as it could cause the application to scan your entire filesystem.

### 3. Running the Application

Once the setup is complete, you can start the FastAPI server:

```bash
uvicorn api.server:app --reload
```

The first time you run the application, it will ingest the codebase, create embeddings, and build the vector store. This might take some time depending on the size of your project. Subsequent runs will be much faster as they will load the existing vector store.

## API Usage

The application exposes a simple API to ask questions.

-   **Endpoint:** `/ask`
-   **Method:** `POST`
-   **Request Body:**
    ```json
    {
      "question": "Your question about the codebase"
    }
    ```
-   **Example using `curl`:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '''{"question": "How is the database initialized?"}'''
    ```