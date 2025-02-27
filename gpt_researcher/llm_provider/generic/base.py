from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys

from typing import Any, Callable, Dict, List, Sequence, cast

from colorama import Fore, Style, init
from fastapi import WebSocket
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage  # keep out of type checking block because of pydantic's nonsense.
from litellm.exceptions import BadRequestError, ContextWindowExceededError
from llm_fallbacks.config import LiteLLMBaseModelSpec
from pydantic import SecretStr

logger: logging.Logger = logging.getLogger(__name__)
os.environ["LITELLM_LOG"] = "DEBUG"  # pyright: ignore[reportPrivateImportUsage]


_SUPPORTED_PROVIDERS: set[str] = {
    "anthropic",
    "azure_openai",
    "bedrock",
    "cohere",
    "dashscope",
    "deepseek",
    "fireworks",
    "google_genai",
    "google_vertexai",
    "groq",
    "huggingface",
    "litellm",
    "mistralai",
    "ollama",
    "openai",
    "together",
    "xai",
}


class GenericLLMProvider:
    def __init__(
        self,
        llm: BaseChatModel | str,
        use_fallbacks: bool = False,
        fallback_models: Sequence[tuple[str, LiteLLMBaseModelSpec] | BaseChatModel] | None = None,
        **kwargs: Any,
    ):
        self.current_model: BaseChatModel = self._get_base_model(llm, **kwargs)
        self.use_fallbacks: bool = use_fallbacks
        from llm_fallbacks.config import FREE_MODELS

        current_model_name: str = str(self.current_model.name).casefold() if str(self.current_model.name or "").strip() else str(self.current_model).casefold()
        self.fallback_models: list[tuple[str, LiteLLMBaseModelSpec] | BaseChatModel] = []
        if not fallback_models:
            for model_name, model_spec in FREE_MODELS:
                if model_name.casefold() == current_model_name:
                    continue
                # Do not suppress/wrap in try/except, we want to fail loudly if a fallback model fails to initialize.
                provider: str = model_spec["litellm_provider"]  # pyright: ignore[reportTypedDictNotRequiredAccess]  # Incorrect type warning.
                # Create the model directly without recursive fallbacks
                fallback_model = self._get_base_model(f"litellm:{provider}/{model_name}", **model_spec)
                self.fallback_models.append(fallback_model)
        elif fallback_models is not None:
            self.fallback_models = list(fallback_models)

        self.fallback_models.insert(0, self.current_model)
        print(
            Fore.GREEN + Style.BRIGHT + "Using LLM provider: " + Style.RESET_ALL + f"{self.current_model}",
            file=sys.__stdout__,
        )
        print(
            Fore.GREEN + Style.BRIGHT + "Using fallbacks: " + Style.RESET_ALL + f"{self.fallback_models}",
            file=sys.__stdout__,
        )
        print(
            f"Using model '{self.current_model}' with fallbacks '{self.fallback_models!r}'",
            file=sys.__stdout__,
        )

    @classmethod
    async def create_chat_completion(
        cls,
        messages: list,  # type: ignore
        model: str | None = None,
        temperature: float | None = 0.4,
        max_tokens: int | None = 16000,
        llm_provider: str | None = None,
        stream: bool | None = False,
        websocket: Any | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        cost_callback: Callable[[Any], None] | None = None,
    ) -> str:
        """Create a chat completion using the specified LLM provider.

        Args:
        ----
            messages (list[dict[str | str]]): The messages to send to the chat completion
            model (str | None): The model to use. Defaults to None.
            temperature (float | None): The temperature to use. Defaults to 0.4.
            max_tokens (int | None): The max tokens to use. Defaults to 16000.
            stream (bool | None): Whether to stream the response. Defaults to False.
            llm_provider (str | None): The LLM Provider to use.
            websocket (WebSocket | None): The websocket used in the current request,
            cost_callback (Callable[[Any], None] | None): Callback function for updating cost

        Returns:
        -------
            str: The response from the chat completion.

        Raises:
        ------
            ValueError: If model is None or max_tokens > 16000
            RuntimeError: If provider fails to get response
        """
        # validate input
        if model is None:
            raise ValueError("Model cannot be None")
        if max_tokens is not None and max_tokens >= 16000:
            raise ValueError(f"Max tokens cannot be more than 16000, but got {max_tokens}")

        # Get the provider from supported providers
        assert llm_provider is not None
        provider: GenericLLMProvider = cls(
            llm_provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **(llm_kwargs or {}),
        )

        response: str = ""
        # create response
        response = await provider.get_chat_response(
            messages,
            bool(stream),
            websocket,
        )

        if cost_callback is not None:
            from gpt_researcher.utils.costs import estimate_llm_cost

            llm_costs: float = estimate_llm_cost(
                str(messages),
                response,
            )
            cost_callback(llm_costs)

        if not response.strip():
            logger.error(f"Failed to get response from {llm_provider} API")
            raise RuntimeError(f"Failed to get response from {llm_provider} API")

        return response

    @classmethod
    def _get_base_model(
        cls,
        provider: str | BaseChatModel,
        **kwargs: Any,
    ) -> BaseChatModel:
        if isinstance(provider, BaseChatModel):
            return provider
        provider, model = provider.split(":", 1) if ":" in provider else (provider, "")
        model_name: str = str(kwargs.get("model", "") or model).strip().casefold()
        model_name = model_name.split(":", 1)[1] if ":" in model_name else model_name
        if "model" not in kwargs:
            kwargs["model"] = model_name
        else:
            kwargs["model"] = model_name
        if provider == "openai":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(**kwargs)
        elif provider == "anthropic":
            _check_pkg("langchain_anthropic")
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(**kwargs)
        elif provider == "azure_openai":
            _check_pkg("langchain_openai")
            from langchain_openai import AzureChatOpenAI

            if "model" in kwargs:
                model_name = str(kwargs.get("model", "") or "").strip().casefold() or model_name
                kwargs = {"azure_deployment": model_name, **kwargs}

            llm = AzureChatOpenAI(**kwargs)
        elif provider == "cohere":
            _check_pkg("langchain_cohere")
            from langchain_cohere import ChatCohere

            llm = ChatCohere(**kwargs)
        elif provider == "google_vertexai":
            _check_pkg("langchain_google_vertexai")
            from langchain_google_vertexai import ChatVertexAI

            llm = ChatVertexAI(**kwargs)
        elif provider == "google_genai":
            _check_pkg("langchain_google_genai")
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(**kwargs)
        elif provider == "fireworks":
            _check_pkg("langchain_fireworks")
            from langchain_fireworks import ChatFireworks

            llm = ChatFireworks(**kwargs)
        elif provider == "ollama":
            _check_pkg("langchain_community")
            from langchain_ollama import ChatOllama

            llm = ChatOllama(base_url=os.environ["OLLAMA_BASE_URL"], **kwargs)
        elif provider == "together":
            _check_pkg("langchain_together")
            from langchain_together import ChatTogether

            llm = ChatTogether(**kwargs)
        elif provider == "mistralai":
            _check_pkg("langchain_mistralai")
            from langchain_mistralai import ChatMistralAI

            llm = ChatMistralAI(**kwargs)
        elif provider == "huggingface":
            _check_pkg("langchain_huggingface")
            from langchain_huggingface import ChatHuggingFace

            if "model" in kwargs or "model_name" in kwargs:
                model_id = kwargs.pop("model", kwargs.pop("model_name", None))
                kwargs = {"model_id": model_id, **kwargs}
            llm = ChatHuggingFace(**kwargs)
        elif provider == "groq":
            _check_pkg("langchain_groq")
            from langchain_groq import ChatGroq

            llm = ChatGroq(**kwargs)
        elif provider == "bedrock":
            _check_pkg("langchain_aws")
            from langchain_aws import ChatBedrock

            if "model" in kwargs or "model_name" in kwargs:
                model_id = kwargs.pop("model", None) or kwargs.pop("model_name", None)
                kwargs = {
                    "model_id": model_id,
                    "model_kwargs": kwargs,
                }
            llm = ChatBedrock(**kwargs)
        elif provider == "dashscope":
            _check_pkg("langchain_dashscope")
            from langchain_dashscope import ChatDashScope

            llm = ChatDashScope(**kwargs)
        elif provider == "xai":
            _check_pkg("langchain_xai")
            from langchain_xai import ChatXAI

            llm = ChatXAI(**kwargs)
        elif provider == "deepseek":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                base_url="https://api.deepseek.com",
                api_key=SecretStr(os.environ["DEEPSEEK_API_KEY"]),
                **kwargs,
            )
        elif provider == "litellm":
            _check_pkg("langchain_community")
            from langchain_community.chat_models.litellm import ChatLiteLLM

            llm = ChatLiteLLM(**kwargs)
        else:
            # Check if provider was provided with the model name. e.g. openai:gpt-4o or anthropic:claude-sonnet-3.5
            if ":" in provider:
                return cls._get_base_model(provider.split(":")[0], **kwargs)
            # checks litellm's providers.
            if "/" in provider:
                return cls._get_base_model(provider.split("/")[0], **kwargs)
            supported = ", ".join(_SUPPORTED_PROVIDERS)
            raise ValueError(f"Unsupported provider: '{provider}'.\nSupported model providers are:\n[{supported}]\n")

        return llm

    def _convert_to_base_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[BaseMessage]:
        """Convert dict messages to BaseMessage objects."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        converted: list[BaseMessage] = []
        for msg in messages:
            role: str = msg["role"]
            content: str = msg["content"]
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:  # user or default
                converted.append(HumanMessage(content=content))
        return converted

    @classmethod
    def _convert_entry_to_str(
        cls,
        entry: BaseMessage | str | str | dict,
    ) -> str:
        result = ""
        if isinstance(entry, str):
            # Return string directly without any processing
            return entry
        elif isinstance(entry, dict):
            assert not isinstance(entry, tuple)  # model.ainvoke/BaseMessage are statically typed incorrectly, this line helps static type checkers.
            # Get content directly from dict
            return str(entry.get("contents", entry.get("content", entry)))
        elif isinstance(entry, tuple):
            # Join tuple elements with space
            return " ".join(str(x) for x in entry)
        elif isinstance(entry, list):
            return cls._convert_to_str(entry)
        elif isinstance(entry, BaseMessage):
            # Handle BaseMessage content
            content: str | list[str | dict] = entry.content
            if isinstance(content, list):
                return cls._convert_to_str(content)
            return cls._convert_entry_to_str(content)
        else:
            raise ValueError(f"Unexpected response type: {entry.__class__.__name__}: repr: {entry!r}")

    @classmethod
    def _convert_to_str(
        cls,
        msg: BaseMessage | str | dict | list,
    ) -> str:
        """Convert various message types to string without unnecessary splitting."""
        if not isinstance(msg, list):
            return cls._convert_entry_to_str(msg)
        
        # Handle list of messages
        parts: list[str] = []
        for entry in msg:
            part: str = cls._convert_entry_to_str(entry)
            if part.strip():
                parts.append(part)
        
        # Join with newlines only between actual content
        return "\n".join(parts)

    async def get_chat_response(
        self,
        messages: list[dict[str, str]] | list[BaseMessage],
        stream: bool,
        websocket: WebSocket | None = None,
        max_retries: int = 1,
        cost_callback: Callable[[float], None] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Get chat response with fallback support.

        Args:
        ----
            messages (list[dict[str, str]] | list[BaseMessage]): List of message dictionaries (with 'role' and 'content' keys), or BaseMessage objects
            stream (bool): Whether to stream the response
            websocket (WebSocket | None): Optional websocket for streaming
            max_retries (int): Maximum number of retries per model

        Returns:
        -------
            (str): The response text.
        """
        headers = {} if headers is None else headers

        for i, model in enumerate(self.fallback_models):
            retries = 0
            while retries < max_retries:
                try:
                    if stream:
                        msgs: list[dict[str, Any]] = cast(List[Dict[str, Any]], messages)  # TODO: cleanup the static types here later.
                        return await self.stream_response(msgs, websocket)
                    response: BaseMessage = await self.current_model.ainvoke(messages, headers=headers)
                    result: str = self._convert_to_str(response)
                    logger.debug(f"Response: {result}")
                    if cost_callback is not None:
                        from gpt_researcher.utils.costs import estimate_llm_cost

                        cost_callback(estimate_llm_cost(self._convert_to_str(messages), result))
                    return result
                except ContextWindowExceededError:
                    logger.warning("Context window exceeded, trying fallback models...", exc_info=True)
                    break  # Move to the next model
                except BadRequestError as e:
                    logger.exception(f"Bad request error with model '{model}'! {e.__class__.__name__}: {e}, retrying {retries}/{max_retries}")
                except Exception as e:
                    err_str: str = str(e).casefold()
                    if "rate limit" in err_str or "too many requests" in err_str:
                        logger.exception(f"Rate limit error with model '{model}'! {e.__class__.__name__}: {e}, retrying {retries}/{max_retries}")
                    else:
                        logger.exception(f"Error with model '{model}'! {e.__class__.__name__}: {e}, retrying {retries}/{max_retries}")
                    retries += 1
                    if retries == max_retries:
                        # If there are no more fallback models, raise an error
                        if self.fallback_models and i == len(self.fallback_models) - 1:
                            raise ValueError(f"All models failed. Last error: {e.__class__.__name__}: {e}")
                        self.next_fallback_model()
                        break
        raise ValueError("All models failed to generate a response")

    def next_fallback_model(
        self,
        i: int | None = None,
    ) -> None:
        if i is None:
            i = self.fallback_models.index(self.current_model)
        next_model: tuple[str, LiteLLMBaseModelSpec] | BaseChatModel = self.fallback_models[i + 1]
        if isinstance(next_model, tuple):
            model_name, model_spec = next_model
            fallback_provider = GenericLLMProvider(model_name, **model_spec)
        else:
            fallback_provider = GenericLLMProvider(next_model)
        self.current_model = fallback_provider.current_model

    async def stream_response(
        self,
        messages: list[dict[str, str]],
        websocket: WebSocket | None = None,
    ) -> str:
        from langchain_core.messages import BaseMessage
        
        paragraph: str = ""
        response: str = ""

        # Streaming the response using the chain astream method from langchain
        async for chunk in self.current_model.astream(messages):
            # Convert chunk to string content
            if isinstance(chunk, dict):
                chunk_content = chunk.get("contents") or chunk.get("content", "")
            elif isinstance(chunk, (list, tuple)):
                chunk_content = " ".join(str(x) for x in chunk)
            elif isinstance(chunk, BaseMessage):
                chunk_content = chunk.content
            else:
                chunk_content = str(chunk)
                
            content = str(chunk_content)
            
            # Append to response and paragraph
            response += content
            paragraph += content
            
            # Only split on actual newlines in the content
            if "\n" in content:
                await self._send_output(paragraph.strip(), websocket)
                paragraph = ""

        # Send any remaining content
        if paragraph.strip():
            await self._send_output(paragraph.strip(), websocket)

        return response

    async def _send_output(
        self,
        content: str,
        websocket: WebSocket | None = None,
    ):
        """Helper to send output to websocket or console."""
        if websocket is not None:
            await websocket.send_json(
                {
                    "type": "report",
                    "output": content.strip(),
                }
            )
        logger.debug(f"{Fore.GREEN}{content.strip()}{Style.RESET_ALL}")


def _check_pkg(pkg: str) -> None:
    if not importlib.util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        # Import colorama and initialize it
        init(autoreset=True)
        # Use Fore.RED to color the error message
        raise ImportError(Fore.RED + f"Unable to import {pkg_kebab}. Please install with `pip install -U {pkg_kebab}`")
