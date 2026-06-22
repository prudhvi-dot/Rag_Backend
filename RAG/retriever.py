from config_vectorStore import get_vectorStore

vector_store = get_vectorStore()

retriever = vector_store.as_retriever()

result = retriever.invoke("summarize the story")

print(result)
