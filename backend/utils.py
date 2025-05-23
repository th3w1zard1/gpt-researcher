from __future__ import annotations

import aiofiles
import urllib.parse
import mistune

async def write_to_file(
    filename: str,
    text: str,
) -> None:
    """Asynchronously write text to a file in UTF-8 encoding.

    Args:
        filename (str): The filename to write to.
        text (str): The text to write.
    """
    text_utf8: str = str(text).encode("utf-8", errors="replace").decode("utf-8")

    async with aiofiles.open(filename.replace(" ", "_"), "w", encoding="utf-8") as file:
        await file.write(text_utf8)

async def write_text_to_md(
    text: str,
    filename: str = "",
) -> str:
    """Write text to a Markdown file and returns the file path.

    Args:
        text (str): Text to write to the Markdown file.

    Returns:
        str: The file path of the generated Markdown file.
    """
    file_path: str = f"outputs/{filename[:60]}.md"
    await write_to_file(file_path, text)
    return urllib.parse.quote(file_path)

async def write_md_to_pdf(
    text: str,
    filename: str = "",
) -> str:
    """Convert Markdown text to a PDF file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated PDF.
    """
    file_path: str = f"outputs/{filename[:60]}.pdf"

    try:
        from md2pdf.core import md2pdf
        md2pdf(
            file_path,
            md_content=text,
            css_file_path="./frontend/pdf_styles.css",
            base_url=None,
        )
        print(f"Report written to '{file_path}'")
    except Exception as e:
        print(f"Error in converting Markdown to PDF: {e.__class__.__name__}: {e}")
        return ""

    encoded_file_path: str = urllib.parse.quote(file_path)
    return encoded_file_path

async def write_md_to_word(
    text: str,
    filename: str = "",
) -> str:
    """Convert Markdown text to a DOCX file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated DOCX.
    """
    file_path: str = f"outputs/{filename[:60]}.docx"

    try:
        from docx import Document
        from htmldocx import HtmlToDocx
        html: str = mistune.html(text)
        doc = Document()
        HtmlToDocx().add_html_to_document(html, doc)
        doc.save(file_path)

        print(f"Report written to '{file_path}'")

        encoded_file_path: str = urllib.parse.quote(file_path)
        return encoded_file_path

    except Exception as e:
        print(f"Error in converting Markdown to DOCX: {e.__class__.__name__}: {e}")
        return ""
