from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
import pandas as pd

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "dataset\Medicine_Details.csv"


def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    df = pd.read_csv(DATA_PATH)
    documents = []
    for _, row in df.iterrows():
        text = f"""
                Medicine: {row['Medicine Name']}
                Composition: {row['Composition']}
                Uses: {row['Uses']}
                Side Effects: {row['Side_effects']}
                Manufacturer: {row['Manufacturer']}
                Excellent Review %: {row['Excellent Review %']}
                Average Review %: {row['Average Review %']}
                Poor Review %: {row['Poor Review %']}
                """
        doc = Document(page_content=text, metadata={"medicine": row["Medicine Name"]})
        documents.append(doc)
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(model_name="models\sentence-transformers_all-MiniLM-L6-v2"), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()