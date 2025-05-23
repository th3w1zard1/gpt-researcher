from __future__ import annotations

import os
import tempfile

from typing import Any

import aiohttp

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


class OnlineDocumentLoader:
    def __init__(self, urls: list[str]):
        self.urls: list[str] = urls

    async def load(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for url in self.urls:
            pages: list[Document] = await self._download_and_process(url)
            for page in pages:
                if page.page_content and str(page.page_content).strip():
                    docs.append(
                        {
                            "raw_content": page.page_content,
                            "url": page.metadata.get("source"),
                        }
                    )

        if not docs:
            raise ValueError("🤷 Failed to load any documents!")

        return docs

    async def _download_and_process(self, url: str) -> list[Document]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=6) as response:
                    if response.status != 200:
                        print(f"Failed to download {url}: HTTP {response.status}")
                        return []

                    content = await response.read()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_extension(url)) as tmp_file:
                        tmp_file.write(content)
                        tmp_file_path = tmp_file.name

                    return await self._load_document(tmp_file_path, self._get_extension(url).strip("."))
        except aiohttp.ClientError as e:
            print(f"Failed to process {url}")
            print(f"{e.__class__.__name__}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error processing {url}")
            print(f"{e.__class__.__name__}: {e}")
            return []

    async def _load_document(
        self,
        file_path: str,
        file_extension: str,
    ) -> list[Document]:
        ret_data: list[Document] = []
        try:
            loader_dict: dict[str, Any] = {
                "pdf": PyMuPDFLoader(file_path),
                "txt": TextLoader(file_path),
                "doc": UnstructuredWordDocumentLoader(file_path),
                "docx": UnstructuredWordDocumentLoader(file_path),
                "pptx": UnstructuredPowerPointLoader(file_path),
                "csv": UnstructuredCSVLoader(file_path, mode="elements"),
                "xls": UnstructuredExcelLoader(file_path, mode="elements"),
                "xlsx": UnstructuredExcelLoader(file_path, mode="elements"),
                "md": UnstructuredMarkdownLoader(file_path),
            }

            loader: Any = loader_dict.get(file_extension, None)
            if loader:
                ret_data: list[Document] = loader.load()

        except Exception as e:
            print(f"Failed to load document : {file_path}")
            print(f"{e.__class__.__name__}: {e}")
        finally:
            os.remove(file_path)  # 删除临时文件

        return ret_data

    @staticmethod
    def _get_extension(url: str) -> str:
        return os.path.splitext(url.split("?")[0])[1]
