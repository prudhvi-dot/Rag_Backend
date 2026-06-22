from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from functools import lru_cache
import os

load_dotenv()


@lru_cache
def get_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("ragrevise")
    return index


@lru_cache
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@lru_cache
def get_vectorStore():

    vector_store = PineconeVectorStore(
        index=get_pinecone_index(), embedding=get_embedding_model()
    )

    return vector_store
