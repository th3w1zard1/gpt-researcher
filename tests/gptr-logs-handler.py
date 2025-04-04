from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

try:
    from gpt_researcher import GPTResearcher
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # Adjust the path to import GPTResearcher from the parent directory
    from gpt_researcher import GPTResearcher
from backend.server.server_utils import CustomLogsHandler
from gpt_researcher.utils.enum import Tone
from gpt_researcher.utils.logger import get_formatted_logger

if TYPE_CHECKING:
    import logging

logger: logging.Logger = get_formatted_logger(__name__)


async def run() -> str:
    """Run the research process and generate a report."""
    query: str = "What happened in the latest burning man floods?"
    report_type: str = "research_report"
    report_source: str = "online"
    tone: Tone = Tone.Informative
    # config_path = ...  # Path to the configuration file

    custom_logs_handler = CustomLogsHandler(
        task=query,
        websocket=None,  # pyright: ignore[reportArgumentType]
    )  # Pass query parameter  # pyright: ignore[reportArgumentType]

    researcher = GPTResearcher(
        query=query,
        report_type=report_type,
        report_source=report_source,
        tone=tone,
        # config=config_path,
        websocket=custom_logs_handler,
    )

    await researcher.conduct_research()  # Conduct the research
    report: str = await researcher.write_report()  # Write the research report
    logger.info("Report generated successfully.")  # Log report generation

    return report


# Run the asynchronous function using asyncio
if __name__ == "__main__":
    asyncio.run(run())
