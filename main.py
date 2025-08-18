import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import ollama
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings(model_name="models\sentence-transformers_all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    #print(results)
    print(f"Found {len(results)} results.")
    if len(results) == 0 or results[0][1] < 0.4:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    #print(f"Context: {context_text}")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    client = ollama.Client()
    model = "llama2"
    
    response_text = client.generate(prompt = prompt, model=model)

    formatted_response = f"Response: {response_text.response}"
    print(formatted_response)


if __name__ == "__main__":
    main()
