from __future__ import annotations

from fastapi import WebSocket
from typing import Any

from gpt_researcher import GPTResearcher


class BasicReport:
    def __init__(
        self,
        query: str,
        report_type: str,
        report_source: str,
        source_urls: list[str],
        document_urls: list[str],
        tone: Any,
        config_path: str,
        websocket: WebSocket | None = None,
        headers: dict[str, Any] | None = None,
    ):
        self.query: str = query
        self.report_type: str = report_type
        self.report_source: str = report_source
        self.source_urls: list[str] = source_urls
        self.document_urls: list[str] = document_urls
        self.tone: Any = tone
        self.config_path: str = config_path
        self.websocket: WebSocket | None = websocket
        self.headers: dict[str, Any] = {} if headers is None else headers
        self.researcher: GPTResearcher | None = None

    async def run(self):
        # Initialize researcher
        self.researcher = GPTResearcher(
            query=self.query,
            report_type=self.report_type,
            report_source=self.report_source,
            source_urls=self.source_urls,
            document_urls=self.document_urls,
            tone=self.tone,
            config_path=self.config_path,
            websocket=self.websocket,
            headers=self.headers,
            visited_urls=set(),
        )

        await self.researcher.conduct_research()
        report = await self.researcher.write_report()
        return report
