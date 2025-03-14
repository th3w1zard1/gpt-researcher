from __future__ import annotations

import logging
import os

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gpt_researcher.utils.logger import get_formatted_logger
from gpt_researcher.utils.logging_config import setup_research_logging
from pydantic import BaseModel

from backend.server.server_utils import (
    execute_multi_agents,
    handle_file_deletion,
    handle_file_upload,
    handle_websocket_communication,
    handle_get_config,
    handle_save_config,
    handle_get_default_config,
    handle_get_settings,
    ConfigRequest,
)
from backend.server.websocket_manager import WebSocketManager

# Get logger instance
logger: logging.Logger = get_formatted_logger(__name__)

# Don't override parent logger settings
logger.propagate = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Only log to console
    ],
)


# Models


class ResearchRequest(BaseModel):
    agent: str
    query_domains: str = ""
    report_type: str
    source_urls: str = ""
    task: str
    tone: str = "objective"
    # Add any additional fields that might be needed for configuration


# App initialization
app = FastAPI()


# Static files and templates
app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")
templates = Jinja2Templates(directory="./frontend")

# WebSocket manager
manager = WebSocketManager()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    """Startup Event."""
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    os.makedirs(DOC_PATH, exist_ok=True)

    # Setup research logging
    log_file, json_file, research_logger, json_handler = setup_research_logging()  # Unpack all 4 values
    setattr(research_logger, "json_handler", json_handler)  # Use setattr to avoid linter error
    research_logger.info(f"Research log file: {log_file}")
    research_logger.info(f"Research JSON file: {json_file}")

    yield


# Routes


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "report": None})


@app.get("/files/")
async def list_files() -> dict[str, list[str]]:
    files: list[str] = os.listdir(DOC_PATH)
    logger.debug(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents() -> JSONResponse:
    return await execute_multi_agents(manager)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)) -> dict[str, str]:
    return await handle_file_upload(file, DOC_PATH)


@app.delete("/files/{filename}")
async def delete_file(filename: str) -> JSONResponse:
    return await handle_file_deletion(filename, DOC_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@app.get("/api/config")
async def get_config() -> JSONResponse:
    """Get the current configuration."""
    return await handle_get_config()


@app.post("/api/config/export")
async def save_config(config: ConfigRequest) -> JSONResponse:
    """Save a configuration."""
    return await handle_save_config(config)


@app.get("/api/config/default")
async def get_default_config() -> JSONResponse:
    """Get the default configuration."""
    return await handle_get_default_config()


@app.post("/api/config/import")
async def import_config(config: ConfigRequest) -> JSONResponse:
    """Import a configuration from a file."""
    return await handle_save_config(config)


@app.get("/api/settings")
async def get_settings() -> JSONResponse:
    """Get the current settings."""
    return await handle_get_settings()
