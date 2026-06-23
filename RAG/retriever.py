from config_vectorStore import get_vectorStore
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    CrossEncoderReranker,
)

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config_vectorStore import get_embedding_model

embedding_model = get_embedding_model()

redundancy_filter = EmbeddingsRedundantFilter(embeddings=embedding_model)
reorderer = LongContextReorder()
cross_encoder = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    top_n=10,
)

retriever = get_vectorStore().as_retriever(
    search_type="mmr",
    search_kwargs={"k": 40, "lambda_mult": 0.25, "filter": {"doc_id": "story1"}},
)

pipeline = DocumentCompressorPipeline(
    transformers=[redundancy_filter, cross_encoder, reorderer]
)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever, base_compressor=pipeline
)

result = compression_retriever.invoke("What is the moral of the story")

print(result[0])
