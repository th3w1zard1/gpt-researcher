# Tavily API Retriever
from __future__ import annotations

import json
import os

from typing import TYPE_CHECKING, Any, Literal, Sequence

import requests

from gpt_researcher.utils.logger import get_formatted_logger
from gpt_researcher.retrievers.retriever_abc import RetrieverABC

if TYPE_CHECKING:
    import logging

    from typing import Any, Literal, Sequence


class TavilySearch(RetrieverABC):
    """Tavily API Retriever."""

    logger: logging.Logger = get_formatted_logger(__name__)

    def __init__(
        self,
        query: str,
        headers: dict[str, Any] | None = None,
        topic: str = "general",
        query_domains: list[str] | None = None,
        *args: Any,  # provided for compatibility with other scrapers
        **kwargs: Any,  # provided for compatibility with other scrapers
    ):
        """
        Initializes the TavilySearch object.

        Args:
            query (str): The search query string.
            headers (dict, optional): Additional headers to include in the request. Defaults to None.
            topic (str, optional): The topic for the search. Defaults to "general".
            query_domains (list, optional): List of domains to include in the search. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.query: str = query
        self.headers: dict[str, str] = headers or {}
        self.topic: str = topic
        self.base_url: str = "https://api.tavily.com/search"
        self.api_key: str = self.get_api_key()
        self.headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        self.query_domains: list[str] | None = query_domains or None
        self.args: tuple[Any, ...] = args
        self.kwargs: dict[str, Any] = kwargs

    def get_api_key(self) -> str:
        """Gets the Tavily API key.

        Returns:
            str: The Tavily API Key
        """
        api_key: str | None = self.headers.get("tavily_api_key")
        if not api_key:
            try:
                api_key = os.environ["TAVILY_API_KEY"]
            except KeyError:
                print(
                    "Tavily API key not found, set to blank. If you need a retriver, please set the TAVILY_API_KEY environment variable."
                )
                return ""
        return api_key

    def _search(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: str = "general",
        days: int = 2,
        max_results: int | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Internal search method to send the request to the API."""
        if max_results is None:
            max_results = int(os.environ.get("MAX_SOURCES", 10))

        data: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "days": days,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "max_results": max_results,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_images": include_images,
            "api_key": self.api_key,
            "use_cache": use_cache,
        }

        response: requests.Response = requests.post(
            self.base_url,
            data=json.dumps(data),
            headers=self.headers,
            timeout=100,
        )

        if response.status_code == 200:
            return response.json()
        else:
            # Raises a HTTPError if the HTTP request returned an unsuccessful status code
            response.raise_for_status()
        return {}

    def search(
        self,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Searches the query.

        Args:
            max_results (int, optional): The maximum number of results to return. Defaults to 10.

        Returns:
            list[dict[str, Any]]: The search results.
        """
        if max_results is None:
            max_results = int(os.environ.get("MAX_SOURCES", 10))
        try:
            # Search the query
            results: dict[str, Any] = self._search(
                self.query,
                search_depth="basic",
                max_results=max_results,
                topic=self.topic,
                include_domains=self.query_domains,
            )
            sources: list[dict[str, Any]] = results.get("results", [])
            if not sources:
                raise Exception("No results found with Tavily API search.")
            # Return the results
            search_response: list[dict[str, Any]] = [
                {
                    "href": obj["url"],
                    "body": obj["content"],
                    "title": obj["title"],
                    "source": obj["source_name"],
                }
                for obj in sources
            ]
        except Exception:
            self.logger.exception(
                "Failed fetching sources. Resulting in empty response."
            )
            search_response = []
        return search_response
