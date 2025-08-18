# Medicine Details Assistant

A Retrieval-Augmented Generation (RAG) system designed to provide detailed information about medicines by querying a local knowledge base. It leverages HuggingFace embeddings for vectorization, ChromaDB for efficient storage and retrieval, and Ollama for local Large Language Model (LLM) inference.

## Features
-   **Local Knowledge Base**: Creates a vector database from a CSV file containing medicine details, allowing for offline queries.
-   **Semantic Search**: Utilizes HuggingFace `all-MiniLM-L6-v2` embeddings to perform semantic similarity searches on medicine information.
-   **Local LLM Integration**: Interfaces with Ollama to use local Large Language Models (e.g., Llama 2) for generating human-like answers based on retrieved context.
-   **Contextual Answers**: Formulates responses by augmenting LLM queries with relevant information retrieved from the vector database, ensuring answers are grounded in the provided medicine data.
-   **Data Preprocessing**: Splits and chunks document content for optimal vector database indexing.
-   **Embedding Comparison Utility**: Includes a script to compare the cosine similarity between two medicine embeddings, useful for understanding semantic relationships.

## Installation
To set up and run the Medicine Details Assistant, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/galang006/medicine_details_assistant.git
    cd medicine_details_assistant
    ```

2.  **Set up Python Environment**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install Python Dependencies**
    Install the necessary libraries using pip:
    ```bash
    pip install langchain-huggingface langchain-chroma ollama pandas python-dotenv numpy
    ```

4.  **Prepare the Dataset**
    The project expects a CSV file named `Medicine_Details.csv` inside a `dataset/` directory at the root of the project. This file should contain columns like "Medicine Name", "Composition", "Uses", "Side\_effects", "Manufacturer", "Excellent Review %", "Average Review %", and "Poor Review %".
    Create the `dataset` directory if it doesn't exist:
    ```bash
    mkdir dataset
    ```
    Place your `Medicine_Details.csv` inside the `dataset` directory.

5.  **Download HuggingFace Embedding Model**
    The project uses the `sentence-transformers/all-MiniLM-L6-v2` model locally. You need to download this model and place it in the `models/` directory.
    Create the `models` directory:
    ```bash
    mkdir models
    ```
    You can download the model using the `sentence-transformers` library in a Python script or manually from the Hugging Face Hub:
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save_pretrained('./models/sentence-transformers_all-MiniLM-L6-v2')
    ```
    This will create the `models/sentence-transformers_all-MiniLM-L6-v2` directory containing the model files, which is where the application expects to find it.

6.  **Install and Configure Ollama**
    This project relies on Ollama for running the Large Language Model locally.
    *   **Download and Install Ollama**: Visit the [Ollama website](https://ollama.com/download) and follow the instructions to install Ollama for your operating system.
    *   **Pull the Llama 2 Model**: Once Ollama is installed and running (ensure the Ollama server is active), open your terminal and pull the `llama2` model:
        ```bash
        ollama pull llama2
        ```

## Usage
Once all prerequisites are met and installed, follow these steps to use the Medicine Details Assistant:

1.  **Create the Vector Database**
    The `create_db.py` script will process your `Medicine_Details.csv`, generate embeddings, and store them in a ChromaDB vector store located in the `chroma/` directory.
    ```bash
    python create_db.py
    ```
    This process might take some time depending on the size of your dataset and your system's performance.

2.  **Query the Medicine Details Assistant**
    Use the `main.py` script to ask questions about medicines. Provide your query text as a command-line argument.
    ```bash
    python main.py "What are the uses of Paracetamol?"
    ```
    Replace `"What are the uses of Paracetamol?"` with your desired question. The system will search the database for relevant information and use the local LLM to generate an answer.

3.  **Compare Embeddings (Utility)**
    The `compare_embeddings.py` script is a simple utility to demonstrate how to compare the cosine similarity between two medicine names' embeddings.
    ```bash
    python compare_embeddings.py
    ```
    This script will output the cosine similarity between "Avastin" and "Bevacizumab".

## Code Structure

The project is organized into the following main files and directories:

-   `.gitignore`: Specifies intentionally untracked files and directories to be ignored by Git (e.g., `dataset/`, `.venv/`, `models/`, `chroma/`, `.env`, `__pycache__/`).
-   `compare_embeddings.py`: A standalone script to demonstrate embedding comparison using cosine similarity. It helps to understand how similar two terms are in the embedding space.
-   `create_db.py`: The core script for preparing the knowledge base.
    -   Loads data from `dataset/Medicine_Details.csv`.
    -   Uses `RecursiveCharacterTextSplitter` to chunk the text data into smaller, manageable pieces.
    -   Generates embeddings for these chunks using `HuggingFaceEmbeddings` (specifically `all-MiniLM-L6-v2`).
    -   Stores the embeddings and original text in a ChromaDB vector store, persisted in the `chroma/` directory.
-   `main.py`: The primary application script for interacting with the RAG system.
    -   Parses command-line arguments for user queries.
    -   Loads the ChromaDB vector store.
    -   Performs a similarity search (`similarity_search_with_relevance_scores`) to find the most relevant medicine details based on the user's query.
    -   Constructs a prompt with the retrieved context and the user's question.
    -   Sends the prompt to the Ollama client (using the `llama2` model) to generate an answer.
    -   Prints the generated response.
-   `chroma/`: (Generated) Directory where the ChromaDB vector store is persisted after running `create_db.py`.
-   `dataset/`: Directory expected to contain `Medicine_Details.csv`, the raw data source for medicine information.
-   `models/`: Directory expected to contain the locally downloaded `sentence-transformers_all-MiniLM-L6-v2` embedding model.
-   `.env`: Optional file for environment variables (though no specific environment variables are explicitly configured or required by the provided code other than the `load_dotenv()` call).