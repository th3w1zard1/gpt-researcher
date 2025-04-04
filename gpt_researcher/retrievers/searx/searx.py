from __future__ import annotations

import json
import os

from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import requests

from gpt_researcher.utils.logger import get_formatted_logger
from gpt_researcher.retrievers.retriever_abc import RetrieverABC

if TYPE_CHECKING:
    import logging

logger: logging.Logger = get_formatted_logger(__name__)


class SearxSearch(RetrieverABC):
    """SearxNG API Retriever."""

    def __init__(
        self,
        query: str,
        query_domains: list[str] | None = None,
        *args: Any,  # provided for compatibility with other scrapers
        **kwargs: Any,  # provided for compatibility with other scrapers
    ):
        """Initializes the SearxSearch object.

        Args:
            query (str): Search query string.
            query_domains (list[str] | None): List of domains to search.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.query: str = query
        self.base_url: str = self.get_searxng_url()
        self.query_domains: list[str] = [] if query_domains is None else query_domains
        self.args: tuple[Any, ...] = args
        self.kwargs: dict[str, Any] = kwargs

    def get_searxng_url(self) -> str:
        """Gets the SearxNG instance URL from environment variables.

        Returns:
            str: Base URL of SearxNG instance.
        """
        try:
            base_url: str = os.environ["SEARX_URL"]
            if not base_url.endswith("/"):
                base_url += "/"
            return base_url
        except KeyError:
            raise Exception("SearxNG URL not found. Please set the SEARX_URL environment variable. You can find public instances at https://searx.space/")

    def search(
        self,
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        """Searches the query using SearxNG API.

        Args:
            max_results (int): Maximum number of results to return.

        Returns:
            list[dict[str, str]]: List of dictionaries containing search results.
        """
        if max_results is None:
            max_results = int(os.environ.get("MAX_SOURCES", 10))

        logger.info(f"SearxSearch: Searching with query:{os.linesep*2}```{self.query}{os.linesep}```")

        search_url: str = urljoin(self.base_url, "search")

        params: dict[str, str] = {
            # The search query.
            "q": self.query,
            # Output format of results. Format needs to be activated in searxng config.
            "format": "json",
        }

        try:
            response: requests.Response = requests.get(
                search_url,
                params=params,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            results: dict[str, Any] = response.json()

            # Normalize results to match the expected format
            search_response: list[dict[str, str]] = []
            for result in results.get("results", [])[:max_results]:
                search_response.append(
                    {
                        "href": result.get("url", ""),
                        "body": result.get("content", ""),
                    }
                )

            return search_response

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error querying SearxNG: {str(e)}")
        except json.JSONDecodeError:
            raise Exception("Error parsing SearxNG response")
