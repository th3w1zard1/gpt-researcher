from __future__ import annotations
import os
import asyncio
from typing import Callable, Any
from .retriever import SearchAPIRetriever, SectionRetriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..vector_store import VectorStoreWrapper
from ..utils.costs import estimate_embedding_cost
from ..memory.embeddings import OPENAI_EMBEDDING_MODEL
from ..prompts import PromptFamily


class VectorstoreCompressor:
    def __init__(
        self,
        vector_store: VectorStoreWrapper,
        max_results: int = 7,
        filter: dict[str, Any] | None = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        self.vector_store: VectorStoreWrapper = vector_store
        self.max_results: int = max_results
        self.filter: dict[str, Any] | None = filter
        self.kwargs: dict[str, Any] = kwargs
        self.prompt_family: type[PromptFamily] | PromptFamily = prompt_family

    async def async_get_context(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[str]:
        """Get relevant context from vector store"""
        results = await self.vector_store.async_similarity_search(
            query=query,
            k=max_results,
            filter=self.filter,
        )
        return self.prompt_family.pretty_print_docs(results)


class ContextCompressor:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        embeddings: Any,
        max_results: int = 5,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        self.max_results: int = max_results
        self.documents: list[dict[str, Any]] = documents
        self.kwargs: dict[str, Any] = kwargs
        self.embeddings: Any = embeddings
        self.similarity_threshold: float = os.environ.get("SIMILARITY_THRESHOLD", 0.35)
        self.prompt_family: type[PromptFamily] | PromptFamily = prompt_family

    def __get_contextual_retriever(self) -> ContextualCompressionRetriever:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold,
        )
        pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter, relevance_filter])
        base_retriever = SearchAPIRetriever(pages=self.documents)
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=base_retriever,
        )
        return contextual_retriever

    async def async_get_context(
        self,
        query: str,
        max_results: int = 5,
        cost_callback: Callable | None = None,
    ) -> list[str]:
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query)
        return self.prompt_family.pretty_print_docs(relevant_docs, max_results)


class WrittenContentCompressor:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        embeddings: Any,
        similarity_threshold: float,
        **kwargs,
    ):
        self.documents: list[dict[str, Any]] = documents
        self.kwargs: dict[str, Any] = kwargs
        self.embeddings: Any = embeddings
        self.similarity_threshold: float = similarity_threshold

    def __get_contextual_retriever(self) -> ContextualCompressionRetriever:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        relevance_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold,
        )
        pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter, relevance_filter])
        base_retriever = SectionRetriever(sections=self.documents)
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=base_retriever,
        )
        return contextual_retriever

    def __pretty_docs_list(
        self,
        docs: list[dict[str, Any]],
        top_n: int,
    ) -> list[str]:
        return [
            f"Title: {d.metadata.get('section_title')}\nContent: {d.page_content}\n"
            for i, d in enumerate(docs)
            if i < top_n
        ]

    async def async_get_context(
        self,
        query: str,
        max_results: int = 5,
        cost_callback: Callable | None = None,
    ) -> list[str]:
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query)
        return self.__pretty_docs_list(relevant_docs, max_results)
