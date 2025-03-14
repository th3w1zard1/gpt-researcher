from __future__ import annotations

import logging
import uuid

from typing import TYPE_CHECKING, Any

from gpt_researcher.config import Config
from gpt_researcher.memory import Memory
from gpt_researcher.prompts import PROMPT_CHAT_WITH_REPORT
from gpt_researcher.utils.llm import get_llm
from gpt_researcher.utils.logger import get_formatted_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from litellm.utils import get_max_tokens


logger: logging.Logger = get_formatted_logger(__name__)

if TYPE_CHECKING:
    import os

    from fastapi import WebSocket
    from langchain.tools import BaseTool
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.vectorstores.base import VectorStoreRetriever
    from langgraph.graph.graph import CompiledGraph


class ChatAgentWithMemory:
    def __init__(
        self,
        report: str,
        config_path: os.PathLike | str | None = None,
        headers: dict[str, Any] | None = None,
        vector_store: InMemoryVectorStore | None = None,
        config: Config | None = None,
    ):
        self.report: str = report
        self.headers: dict[str, Any] | None = headers
        self.config: Config = (
            Config.from_path(config_path)
            if config_path is not None
            else config
            if config is not None
            else Config()
        )
        self.vector_store: InMemoryVectorStore | None = vector_store
        self.graph: CompiledGraph = self.create_agent()

    def create_agent(self) -> CompiledGraph:
        """Create React Agent Graph."""
        assert self.config.SMART_LLM_PROVIDER is not None, "smart_llm_provider is not set"
        assert self.config.SMART_LLM_MODEL is not None, "smart_llm_model is not set"
        assert isinstance(self.config.SMART_LLM_PROVIDER, str), (
            f"smart_llm_provider is not a str, was instead {self.config.SMART_LLM_PROVIDER.__class__.__name__}"
        )
        assert isinstance(self.config.SMART_LLM_MODEL, str), (
            f"smart_llm_model is not a str, was instead {self.config.SMART_LLM_MODEL.__class__.__name__}"
        )
        assert isinstance(self.config.llm_kwargs, dict), (
            f"llm_kwargs is not a dict, was instead {self.config.llm_kwargs.__class__.__name__}"
        )
        assert isinstance(self.config.EMBEDDING_KWARGS, dict), (
            f"embedding_kwargs is not a dict, was instead {self.config.EMBEDDING_KWARGS.__class__.__name__}"
        )
        assert isinstance(self.config.EMBEDDING_PROVIDER, str), (
            f"embedding_provider is not a str, was instead {self.config.EMBEDDING_PROVIDER.__class__.__name__}"
        )
        assert isinstance(self.config.EMBEDDING_MODEL, str), (
            f"embedding_model is not a str, was instead {self.config.EMBEDDING_MODEL.__class__.__name__}"
        )

        params: dict[str, Any] = {"model": self.config.SMART_LLM_MODEL, "temperature": 0.35}
        try:
            max_tokens: int | None = get_max_tokens(self.config.SMART_LLM_MODEL)
            params["max_tokens"] = max_tokens
        except Exception as e:
            logger.warning(f"Error in get_max_tokens: {e.__class__.__name__}: {e}")

        params["model"] = self.config.SMART_LLM
        provider: BaseChatModel = get_llm(
            **params,
            **self.config.llm_kwargs,
        ).current_model

        # If vector_store is not initialized, process documents and add to vector_store
        if self.vector_store is None:
            self.chat_config: dict[str, Any] = {"configurable": {"thread_id": uuid.uuid4().hex}}
            self.embedding: Embeddings = Memory(
                self.config.EMBEDDING_PROVIDER,
                self.config.EMBEDDING_MODEL,
                **self.config.EMBEDDING_KWARGS,
            ).get_embeddings()
            self.vector_store = InMemoryVectorStore(self.embedding)
        chunks: list[str] = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        ).split_text(self.report)
        self.vector_store.add_texts(chunks)

        self.retriever: VectorStoreRetriever = self.vector_store.as_retriever(k=4)
        return create_react_agent(
            model=provider,
            tools=[self.vector_store_tool(self.vector_store)],
            checkpointer=MemorySaver(),
        )

    def vector_store_tool(
        self,
        vector_store: InMemoryVectorStore,
    ) -> BaseTool:
        """Create Vector Store Tool."""
        @tool
        def retrieve_info(query: str) -> list[str]:
            """Consult the report for relevant contexts whenever you don't know something."""
            return [page.page_content for page in self.retriever.invoke(query)]
        return retrieve_info

    async def chat(
        self,
        message: str,
        websocket: WebSocket | None = None,
    ) -> str:
        """Chat with React Agent."""
        message = PROMPT_CHAT_WITH_REPORT.format(report=self.report, message=message)
        inputs: dict[str, Any] = {"messages": [("user", message)]}
        cfg: RunnableConfig | Any = self.chat_config
        response: dict[str, Any] = await self.graph.ainvoke(inputs, config=cfg)
        last_message: BaseMessage = response["messages"][-1]
        ai_message: str | list[str | dict[str, Any]] = last_message.content
        if websocket is not None:  # fastapi
            await websocket.send_json({"type": "chat", "content": ai_message})
        if isinstance(ai_message, list):
            return "\n".join([item["text"] if isinstance(item, dict) else item for item in ai_message])
        elif isinstance(ai_message, str):
            return ai_message
        else:
            raise ValueError(f"Unexpected message type: {ai_message.__class__.__name__}")

    def get_context(self) -> str:
        """return the current context of the chat."""
        return self.report
