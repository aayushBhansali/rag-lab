import ollama
from src.rag.data_loader.AbstractDataLoader import AbstractDataLoader
from src.rag.data_loader.TextFileLoader import TextFileLoader
from pathlib import Path

dataset = []

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []


def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))



def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)



def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    # Temporary list to store chunks
    similarities = []

    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    # Sort by similarity
    similarities.sort(key=lambda x:x[1], reverse=True)

    return similarities[:top_n]


# Print the response from the chatbot in realtime



def main():
    data_file = Path("dataset.txt")
    loader: AbstractDataLoader = TextFileLoader()
    chunks: list[str] = loader.load(data_file)

    for i, chunk in enumerate(chunks):
        add_chunk_to_database(chunk)
        print(f"Added chunk {i+1} / {len(chunks)} to the database")

        
        
    # Continue chat loop to avoid re-embedding everything
    while True:
        input_query = input("\n\nAsk me a question: ")


        if input_query == "exit":
            break

        retreived_knowledge = retrieve(input_query)

        retrieved_chunks = [
            {'id': i, "text": chunk.strip()}
            for i, (chunk, _) in enumerate(retreived_knowledge)
        ]

        retrieved_chunk_str_builder = []

        for idx, chunk in enumerate(retrieved_chunks):
            retrieved_chunk_str_builder.append(f"{idx+1}: {chunk}")

        print(f"Retrieved {len(retrieved_chunks)} chunks")


        instruction_prompt = f"""
        You are a helpful chatbot. Use only the following pieces of information to answer the question, do not make up any new information.
        {'\n'.join(f"- {chunk}" for chunk, similarity in retreived_knowledge)}
        """


        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream=True
        )

        print('Chatbot response: ')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)



if __name__ == "__main__":
    main()