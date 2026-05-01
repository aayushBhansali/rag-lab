import ollama
import pickle
import pathlib

dataset = []

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []

pickled_file = "embeddings.bin"


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


if pathlib.Path(pickled_file).exists():
    print("Retreiving vector database from storage ...")
    with open(pickled_file, 'rb') as fs:
        VECTOR_DB = pickle.load(fs)
else:
    print("No stored vector database found, creating a new one ...")    
    with open("dataset.txt", "r", encoding="utf-8") as fs:
        dataset = fs.readlines()
        print(f"Loaded {len(dataset)} entries")


    for i, chunk in enumerate(dataset):
        add_chunk_to_database(chunk)
        print(f"Added chunk {i+1} / {len(dataset)} to the database")


    print("Saving vector state ...")
    with open(pickled_file, 'wb') as fs:
        pickle.dump(VECTOR_DB, fs)


# Print the response from the chatbot in realtime

# Continue chat loop to avoid re-embedding everything
while True:
    input_query = input("\nAsk me a question: ")


    if input_query == "exit":
        break

    retreived_knowledge = retrieve(input_query)

    print("Retrieved knowledge")
    for chunk, similarity in retreived_knowledge:
        print(f"- similarity: {similarity:.2f} {chunk}")


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
