from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

def main():
    embedding_function = HuggingFaceEmbeddings(
        model_name="models/sentence-transformers_all-MiniLM-L6-v2"
    )

    vec1 = embedding_function.embed_query("Avastin")
    vec2 = embedding_function.embed_query("Bevacizumab")

    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"Cosine similarity: {cos_sim}")


if __name__ == "__main__":
    main()