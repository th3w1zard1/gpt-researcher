from __future__ import annotations

import json
import os
import warnings

from typing import Any, ClassVar, List, Union, get_args, get_origin

from llm_fallbacks.config import LiteLLMBaseModelSpec

from gpt_researcher.config.variables.base import BaseConfig
from gpt_researcher.config.variables.default import DEFAULT_CONFIG
from gpt_researcher.retrievers.utils import get_all_retriever_names


class Config:
    """Config class for GPT Researcher."""

    CONFIG_DIR: ClassVar[str] = os.path.join(os.path.dirname(__file__), "variables")

    def __init__(
        self,
        config_path: str | None = None,
    ):
        """Initialize the config class."""
        self.config_path: str | None = config_path
        self.llm_kwargs: dict[str, Any] = {}
        self.embedding_kwargs: dict[str, Any] = {}

        config_to_use: BaseConfig = self.load_config(config_path)
        self._set_attributes(config_to_use)
        self._set_embedding_attributes()
        self._set_llm_attributes()
        self._handle_deprecated_attributes()
        if config_to_use["REPORT_SOURCE"] != "web":
            self._set_doc_path(config_to_use)

        # MCP support configuration
        self.mcp_servers: list[str] = config_to_use.get("MCP_SERVERS", [])

        # Allowed root paths for MCP servers
        self.mcp_allowed_root_paths: list[str] = config_to_use.get("MCP_ALLOWED_ROOT_PATHS", [])

    def _set_attributes(
        self,
        config: dict[str, Any],
    ) -> None:
        for key, value in config.items():
            env_value: str | None = os.getenv(key)
            if env_value is not None:
                value: Any = self.convert_env_value(key, env_value, BaseConfig.__annotations__[key])
            setattr(self, key.casefold(), value)
            setattr(self, key.upper(), value)

        # Handle RETRIEVER with default value
        retriever_env: str = os.environ.get("RETRIEVER", config.get("RETRIEVER", "tavily"))
        try:
            self.retrievers: list[str] = self.parse_retrievers(retriever_env)
        except ValueError as e:
            print(f"Warning: {str(e)}. Defaulting to 'tavily' retriever.")
            self.retrievers = ["tavily"]

    def _set_embedding_attributes(self) -> None:
        self.embedding_provider, self.embedding_model = self.parse_embedding(self.embedding)  # pyright: ignore[reportAttributeAccessIssue]

        # Parse fallbacks for embedding model
        self.embedding_fallback_list: list[str] = self.parse_model_fallbacks(
            self.embedding_fallbacks,  # pyright: ignore[reportAttributeAccessIssue]
            "embedding",
            self.embedding,  # pyright: ignore[reportAttributeAccessIssue]
        )

    def _set_llm_attributes(self) -> None:
        self.fast_llm_provider, self.fast_llm_model = self.parse_llm(self.fast_llm)  # pyright: ignore[reportAttributeAccessIssue]
        self.smart_llm_provider, self.smart_llm_model = self.parse_llm(self.smart_llm)  # pyright: ignore[reportAttributeAccessIssue]
        self.strategic_llm_provider, self.strategic_llm_model = self.parse_llm(self.strategic_llm)  # pyright: ignore[reportAttributeAccessIssue]

        # Parse fallbacks for each LLM type
        self.fast_llm_fallback_list: list[str] = self.parse_model_fallbacks(self.fast_llm_fallbacks, "chat", self.fast_llm)  # pyright: ignore[reportAttributeAccessIssue]
        self.smart_llm_fallback_list: list[str] = self.parse_model_fallbacks(self.smart_llm_fallbacks, "chat", self.smart_llm)  # pyright: ignore[reportAttributeAccessIssue]
        self.strategic_llm_fallback_list: list[str] = self.parse_model_fallbacks(self.strategic_llm_fallbacks, "chat", self.strategic_llm)  # pyright: ignore[reportAttributeAccessIssue]

    def _handle_deprecated_attributes(self) -> None:
        if os.getenv("EMBEDDING_PROVIDER") is not None:
            warnings.warn(
                "EMBEDDING_PROVIDER is deprecated and will be removed soon. Use EMBEDDING instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.embedding_provider: str | None = (os.environ["EMBEDDING_PROVIDER"] or self.embedding_provider or "").strip() or None

            match os.environ["EMBEDDING_PROVIDER"]:
                case "ollama":
                    self.embedding_model: str | None = (os.environ["OLLAMA_EMBEDDING_MODEL"] or "").strip() or None
                case "custom":
                    self.embedding_model = (os.environ["OPENAI_EMBEDDING_MODEL"] or "").strip() or None
                case "openai":
                    self.embedding_model = "text-embedding-3-large"
                case "azure_openai":
                    self.embedding_model = "text-embedding-3-large"
                case "huggingface":
                    self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                case "gigachat":
                    self.embedding_model = "Embeddings"
                case "google_genai":
                    self.embedding_model = "text-embedding-004"
                case _:
                    raise Exception("Embedding provider not found.")

        _deprecation_warning = "LLM_PROVIDER, FAST_LLM_MODEL and SMART_LLM_MODEL are deprecated and " "will be removed soon. Use FAST_LLM and SMART_LLM instead."
        if os.getenv("LLM_PROVIDER") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_provider: str | None = (os.environ["LLM_PROVIDER"] or self.fast_llm_provider or "").strip() or None
            self.smart_llm_provider: str | None = (os.environ["LLM_PROVIDER"] or self.smart_llm_provider or "").strip() or None
        if os.getenv("FAST_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_model: str | None = (os.environ["FAST_LLM_MODEL"] or self.fast_llm_model or "").strip() or None
        if os.getenv("SMART_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.smart_llm_model: str | None = (os.environ["SMART_LLM_MODEL"] or self.smart_llm_model or "").strip() or None

    def _set_doc_path(self, config: dict[str, Any]) -> None:
        self.doc_path: str = config["DOC_PATH"]
        if self.doc_path and self.doc_path.strip():
            try:
                self.validate_doc_path()
            except Exception as e:
                print(f"Warning: Error validating doc_path: {str(e)}. Using default doc_path.")
                self.doc_path = DEFAULT_CONFIG["DOC_PATH"]

    @classmethod
    def load_config(cls, config_path: str | None) -> BaseConfig:
        """Load a configuration by name."""
        # Merge with default config to ensure all keys are present
        copied_default_cfg: BaseConfig | dict[str, Any] = DEFAULT_CONFIG.copy()

        if config_path is None or not config_path.strip():
            print("[WARN] config json not provided, loading default!")
            return copied_default_cfg

        # config_path = os.path.join(cls.CONFIG_DIR, config_path)
        if not os.path.exists(config_path):
            if config_path.strip().casefold() != "default":
                print(f"Warning: Configuration not found at '{config_path}'. Using default configuration.")
                if not config_path.casefold().endswith(".json"):
                    print(f"Did you mean: '{config_path}.json'?")
            return copied_default_cfg

        with open(config_path, "r") as f:
            print(f"[INFO] Loading config json from '{os.path.abspath(config_path)}'...")
            custom_config = json.load(f)

        copied_default_cfg.update(custom_config)
        return copied_default_cfg

    @classmethod
    def list_available_configs(cls) -> list[str]:
        """List all available configuration names."""
        configs: list[str] = ["default"]
        for file in os.listdir(cls.CONFIG_DIR):
            if file.casefold().endswith(".json"):
                configs.append(file[:-5])  # Remove .json extension
        return configs

    def parse_retrievers(self, retriever_str: str) -> list[str]:
        """Parse the retriever string into a list of retrievers and validate them."""
        retrievers: list[str] = [retriever.strip() for retriever in retriever_str.strip().split(",")]
        valid_retrievers: list[Any] = get_all_retriever_names() or []
        invalid_retrievers: list[str] = [r for r in retrievers if r not in valid_retrievers]
        if invalid_retrievers:
            raise ValueError(f"Invalid retriever(s) found: {', '.join(invalid_retrievers)}. " f"Valid options are: {', '.join(valid_retrievers)}.")
        return retrievers

    @staticmethod
    def parse_llm(llm_str: str | None) -> tuple[str | None, str | None]:
        """Parse llm string into (llm_provider, llm_model)."""
        from gpt_researcher.llm_provider.generic.base import _SUPPORTED_PROVIDERS

        if llm_str is None:
            return None, None
        try:
            llm_provider, llm_model = llm_str.split(":", 1)
            assert llm_provider in _SUPPORTED_PROVIDERS, f"Unsupported {llm_provider}.\nSupported llm providers are: " + ", ".join(_SUPPORTED_PROVIDERS)
            return llm_provider, llm_model
        except ValueError:
            raise ValueError("Set SMART_LLM or FAST_LLM = '<llm_provider>:<llm_model>' " "e.g. 'openai:gpt-4o-mini'")

    @staticmethod
    def parse_embedding(embedding_str: str | None) -> tuple[str | None, str | None]:
        """Parse embedding string into (embedding_provider, embedding_model)."""
        from gpt_researcher.memory.embeddings import _SUPPORTED_PROVIDERS

        if embedding_str is None:
            return None, None
        try:
            embedding_provider, embedding_model = embedding_str.split(":", 1)
            assert embedding_provider in _SUPPORTED_PROVIDERS, f"Unsupported {embedding_provider}.\nSupported embedding providers are: " + ", ".join(_SUPPORTED_PROVIDERS)
            return embedding_provider, embedding_model
        except ValueError:
            raise ValueError("Set EMBEDDING = '<embedding_provider>:<embedding_model>' " "Eg 'openai:text-embedding-3-large'")

    def validate_doc_path(self):
        """Ensure that the folder exists at the doc path"""
        os.makedirs(self.doc_path, exist_ok=True)

    @staticmethod
    def convert_env_value(
        key: str,
        env_value: str,
        type_hint: type,
    ) -> Any:
        """Convert environment variable to the appropriate type based on the type hint."""
        origin: Any | None = get_origin(type_hint)
        args: tuple[Any, ...] = get_args(type_hint)

        if origin is Union:
            # Handle Union types (e.g., Union[str, None])
            for arg in args:
                if arg is type(None):
                    if env_value.casefold().strip() in ("none", "null", ""):
                        return None
                else:
                    try:
                        return Config.convert_env_value(key, env_value, arg)
                    except ValueError:
                        continue
            raise ValueError(f"Cannot convert {env_value} to any of {args}")

        if type_hint is bool:
            return env_value.casefold().strip() in ("true", "1", "yes", "on")
        elif type_hint is int:
            return int(env_value)
        elif type_hint is float:
            return float(env_value)
        elif type_hint in (str, Any):
            return env_value
        elif origin is list or origin is List:
            return json.loads(env_value)
        elif type_hint is dict:
            return json.loads(env_value)
        else:
            raise ValueError(f"Unsupported type {type_hint} for key {key}")

    def set_verbose(self, verbose: bool) -> None:
        """Set the verbosity level."""
        self.llm_kwargs["verbose"] = verbose

    def get_mcp_server_config(self, server_name: str) -> dict:
        """Get the configuration for an MCP server.

        Args:
            server_name (str): The name of the MCP server to get the config for.

        Returns:
            dict: The server configuration, or an empty dict if the server is not found.
        """
        if not server_name or not self.mcp_servers:
            return {}

        for server in self.mcp_servers:
            if isinstance(server, dict) and server.get("name") == server_name:
                return server

        return {}

    @staticmethod
    def parse_model_fallbacks(
        fallbacks_str: str,
        model_type: str,
        primary_model: str,
    ) -> list[str]:
        """Parse fallbacks string into a list of model names.

        Args:
            fallbacks_str: Comma-separated list of model names or "auto"
            model_type: Type of model (chat, completion, embedding, etc.)
            primary_model: The primary model to exclude from fallbacks

        Returns:
            List of model names to use as fallbacks
        """
        if not fallbacks_str or fallbacks_str.strip() == "":
            return []

        # Check for auto configuration - build fallback list from free models
        if fallbacks_str.strip().casefold() == "auto":
            try:
                # Import all models and filter for free ones
                from llm_fallbacks.core import get_litellm_models, sort_models_by_cost_and_limits

                # Get all models and sort by cost (free first)
                all_models: dict[str, LiteLLMBaseModelSpec] = get_litellm_models()

                # For embeddings, we need to get embedding models specifically
                if model_type == "embedding":
                    from llm_fallbacks.core import get_embedding_models

                    all_models = get_embedding_models()

                free_models_list: list[tuple[str, LiteLLMBaseModelSpec]] = sort_models_by_cost_and_limits(all_models, free_only=True)

                # Extract model names from free_models_list
                free_models: list[str] = [f"{spec.get('litellm_provider', 'openrouter')}:{model_name}" for model_name, spec in free_models_list]

                return free_models
            except ImportError:
                print("Warning: llm_fallbacks module not available. Auto fallbacks disabled.")
                return []

        # Parse comma-separated list
        fallbacks: list[str] = [model.strip() for model in fallbacks_str.split(",") if model.strip()]

        # Remove the primary model from fallbacks if present
        if primary_model in fallbacks:
            fallbacks.remove(primary_model)

        return fallbacks
