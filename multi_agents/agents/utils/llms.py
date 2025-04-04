from __future__ import annotations

from typing import TYPE_CHECKING, Any

import json5 as json
import json_repair
from gpt_researcher.config.config import Config
from gpt_researcher.llm_provider.generic.base import GenericLLMProvider
from langchain_community.adapters.openai import convert_openai_messages

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable

    from langchain_core.messages.base import BaseMessage

from gpt_researcher.utils.logger import get_formatted_logger

logger: logging.Logger = get_formatted_logger(__name__)


async def call_model(
    prompt: list,
    model: str,
    response_format: str | None = None,
    cost_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Call an LLM model with the given prompt.

    Args:
        prompt: The prompt to send
        model: The model to use
        response_format: Optional response format
        cost_callback: Optional callback for cost tracking

    Returns:
        The model's response
    """
    cfg = Config()
    lc_messages: list[BaseMessage] | list[dict[str, str]] = convert_openai_messages(
        prompt
    )  # pyright: ignore[reportAssignmentType]

    try:
        provider = GenericLLMProvider(
            cfg.SMART_LLM_PROVIDER
            if model is None
            else f"{cfg.SMART_LLM_PROVIDER}:{model}",
            fallback_models=cfg.FALLBACK_MODELS,
            temperature=0,
            **cfg.llm_kwargs,
        )
        response: str = await provider.get_chat_response(
            messages=lc_messages,
            stream=False,
            cost_callback=cost_callback,
        )

        if response_format == "json":
            try:
                result = json.loads(response.strip("```json\n"))
                if isinstance(result, dict):
                    return result
                else:
                    raise ValueError(
                        f"Unexpected response format: {result.__class__.__name__} with value: {result!r}"
                    )
            except Exception as e:
                logger.warning("⚠️ Error in reading JSON, attempting to repair JSON ⚠️")
                logger.exception(
                    f"Error in reading JSON: {e.__class__.__name__}: {e}. Attempting to repair reponse: {response}"
                )
                result = json_repair.loads(response)
                assert isinstance(result, dict)
                return result
        else:
            assert isinstance(response, dict)
            return {"response": response}

    except Exception as e:
        logger.exception(f"Error in calling model: {e.__class__.__name__}: {e}")
        return {"response": None}
