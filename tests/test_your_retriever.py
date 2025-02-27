from __future__ import annotations

import asyncio
import pprint

from dotenv import load_dotenv
from fastapi import WebSocket
from gpt_researcher.actions.retriever import get_retrievers
from gpt_researcher.config.config import Config
from gpt_researcher.skills.researcher import ResearchConductor
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

# Load environment variables from .env file
load_dotenv()


async def test_scrape_data_by_query():
    # Initialize the Config object
    config = Config()

    # Retrieve the retrievers based on the current configuration
    retrievers = get_retrievers({}, config)
    print("Retrievers:", retrievers)

    # Create a mock researcher object with necessary attributes
    class MockResearcher:
        def __init__(self):
            self.retrievers: list[type[BaseRetriever]] = retrievers
            self.cfg: Config = config
            self.verbose: bool = True
            self.websocket: WebSocket | None = None
            self.scraper_manager: None = None  # Mock or implement scraper manager
            self.vector_store: VectorStore | None = None  # Mock or implement vector store

    researcher = MockResearcher()
    research_conductor = ResearchConductor(researcher)  # type: ignore[arg-type]
    print("research_conductor", dir(research_conductor))
    print("MockResearcher", dir(researcher))

    # Define a sub-query to test
    sub_query = "design patterns for autonomous ai agents"

    # Iterate through all retrievers
    for retriever_class in retrievers:
        # Instantiate the retriever with the sub-query
        retriever = retriever_class()

        # Perform the search using the current retriever
        search_results = await retriever.ainvoke(sub_query)

        print("\033[35mSearch results:\033[0m")
        pprint.pprint(search_results, indent=4, width=80)


if __name__ == "__main__":
    asyncio.run(test_scrape_data_by_query())
