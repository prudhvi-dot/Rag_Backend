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
from functools import lru_cache


@lru_cache
def get_encoder_model():
    return HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_encoder():
    return CrossEncoderReranker(
        model=get_encoder_model(),
        top_n=10,
    )


@lru_cache
def get_pipeline():

    redundancy_filter = EmbeddingsRedundantFilter(embeddings=get_embedding_model())
    reorderer = LongContextReorder()

    pipeline = DocumentCompressorPipeline(
        transformers=[redundancy_filter, get_encoder(), reorderer]
    )

    return pipeline


def get_retriever(doc_id: str):

    retriever = get_vectorStore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 40, "lambda_mult": 0.25, "filter": {"doc_id": doc_id}},
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever, base_compressor=get_pipeline()
    )

    return compression_retriever
