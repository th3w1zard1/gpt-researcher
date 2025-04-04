from __future__ import annotations

import os

from typing import TYPE_CHECKING, Any

import pytest

from dotenv import load_dotenv
try:
    from gpt_researcher import GPTResearcher
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # Adjust the path to import GPTResearcher from the parent directory
    from gpt_researcher import GPTResearcher

load_dotenv()

from gpt_researcher.utils.logger import get_formatted_logger

if TYPE_CHECKING:
    import logging

logger: logging.Logger = get_formatted_logger(__name__)

# Define the report types to test
report_types = [
    "research_report",
    "custom_report",
    "subtopic_report",
    "summary_report",
    "detailed_report",
    "quick_report",
]

# Define a common query and sources for testing
query = "What can you tell me about myself based on my documents?"

# Define the output directory
output_dir = "./outputs"


@pytest.mark.asyncio
@pytest.mark.parametrize("report_type", report_types)
async def test_gpt_researcher(
    report_type: str,
    config: dict[str, Any] | None = None,
):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create an instance of GPTResearcher with report_source set to "documents"
    researcher = GPTResearcher(
        query=query,
        report_type=report_type,
        report_source="documents",
        config=config,
    )

    # Conduct research and write the report
    await researcher.conduct_research()
    report = await researcher.write_report()
    logger.debug("report:\n", report)

    # Define the expected output filenames
    pdf_filename = os.path.join(output_dir, f"{report_type}.pdf")
    docx_filename = os.path.join(output_dir, f"{report_type}.docx")

    # Check if the PDF and DOCX files are created
    assert os.path.exists(pdf_filename), f"PDF file not found for report type: {report_type}"
    assert os.path.exists(docx_filename), f"DOCX file not found for report type: {report_type}"

    # Clean up the generated files (optional)
    # os.remove(pdf_filename)
    # os.remove(docx_filename)


if __name__ == "__main__":
    pytest.main()
