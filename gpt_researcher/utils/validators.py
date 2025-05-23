from __future__ import annotations

from pydantic import BaseModel, Field


class Subtopic(BaseModel):
    task: str = Field(description="Task name", min_length=1)


class Subtopics(BaseModel):
    subtopics: list[Subtopic] = []
