from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config_vectorStore import get_vectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("./documents/story1.pdf")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs = loader.load()

chunks = text_splitter.split_documents(docs)

chunks = [
    Document(
        page_content=chunk.page_content, metadata={**chunk.metadata, "doc_id": "story1"}
    )
    for chunk in chunks
]


vectorStore = get_vectorStore()

vectorStore.add_documents(chunks)
