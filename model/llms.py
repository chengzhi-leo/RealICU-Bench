# basic calling for llms, for example with openai api
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import openai
from anthropic import Anthropic, AnthropicVertex
from config.config import get_config

try:
    from google import genai as google_genai_sdk
except ImportError:
    google_genai_sdk = None

try:
    import google.generativeai as google_generativeai_sdk
except ImportError:
    google_generativeai_sdk = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Look for .env file in project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass


class LLMClient:
    """
    Unified LLM client supporting multiple providers (OpenAI, Anthropic, Gemini).
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        retry_max_delay_seconds: float = 30.0,
        request_timeout_seconds: float = 300.0,
    ):
        """
        Initialize LLM client.

        Args:
            provider: "openai", "anthropic", "google", or "gemini"
            model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-flash")
            api_key: API key (if None, will use environment variable)
            temperature: Optional sampling temperature override. If None, read from config.llm.temperature.
            max_tokens: Maximum tokens in response. If None, read from config.llm.max_tokens.
            max_input_tokens: Maximum tokens in request input prompt. If None, read from config.llm.max_input_tokens.
            max_retries: Maximum number of retries for transient API errors
            retry_base_delay_seconds: Base backoff delay in seconds
            retry_max_delay_seconds: Maximum backoff delay in seconds
            request_timeout_seconds: Per-request timeout for provider API calls
        """
        self.provider = provider.lower()
        config = None
        if temperature is None or max_tokens is None or max_input_tokens is None:
            config = get_config()

        resolved_temperature = temperature
        if resolved_temperature is None:
            resolved_temperature = config.llm_temperature

        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        resolved_max_input_tokens = max_input_tokens
        if resolved_max_input_tokens is None:
            resolved_max_input_tokens = config.llm_max_input_tokens

        try:
            self.temperature = float(resolved_temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid temperature={resolved_temperature!r}; expected float") from exc

        try:
            parsed_max_tokens = int(resolved_max_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid max_tokens={resolved_max_tokens!r}; expected positive integer") from exc
        if parsed_max_tokens <= 0:
            raise ValueError(f"Invalid max_tokens={parsed_max_tokens}; must be > 0")

        try:
            parsed_max_input_tokens = int(resolved_max_input_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid max_input_tokens={resolved_max_input_tokens!r}; expected positive integer"
            ) from exc
        if parsed_max_input_tokens <= 0:
            raise ValueError(f"Invalid max_input_tokens={parsed_max_input_tokens}; must be > 0")
        if tiktoken is None:
            raise ImportError("tiktoken is required for enforcing llm.max_input_tokens. Install with: pip install tiktoken")

        self.max_tokens = parsed_max_tokens
        self.max_input_tokens = parsed_max_input_tokens
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._token_encoding = None

        if self.provider == "anthropic":
            self.model = model or "claude-3-5-sonnet-20241022"
            use_vertex_ai = self._resolve_anthropic_vertex_mode()
            if use_vertex_ai:
                project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT_ID")
                location = (
                    os.getenv("ANTHROPIC_VERTEX_REGION")
                    or os.getenv("GOOGLE_CLOUD_LOCATION")
                    or os.getenv("GOOGLE_LOCATION")
                )
                if not project:
                    raise ValueError(
                        "GOOGLE_CLOUD_PROJECT (or GOOGLE_PROJECT_ID) not found in environment for Anthropic Vertex."
                    )
                if not location:
                    raise ValueError(
                        "ANTHROPIC_VERTEX_REGION (or GOOGLE_CLOUD_LOCATION / GOOGLE_LOCATION) "
                        "not found in environment for Anthropic Vertex."
                    )
                timeout_seconds = self._resolve_timeout_seconds()
                vertex_kwargs: Dict[str, Any] = {
                    "project_id": project,
                    "region": location,
                }
                if timeout_seconds is not None:
                    vertex_kwargs["timeout"] = timeout_seconds
                self.client = AnthropicVertex(**vertex_kwargs)
                self.anthropic_sdk = "anthropic_vertex"
            else:
                api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY not found in environment. "
                        "For Vertex AI Claude, set ANTHROPIC_USE_VERTEXAI=true and configure GOOGLE_CLOUD_PROJECT."
                    )
                timeout_seconds = self._resolve_timeout_seconds()
                anthropic_kwargs: Dict[str, Any] = {"api_key": api_key}
                if timeout_seconds is not None:
                    anthropic_kwargs["timeout"] = timeout_seconds
                self.client = Anthropic(**anthropic_kwargs)
                self.anthropic_sdk = "anthropic"

        elif self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            openai.api_key = api_key
            self.client = openai

        elif self.provider in {"google", "gemini"}:
            self.model = model or "gemini-1.5-flash"
            api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            use_vertex_ai = self._resolve_google_vertexai_mode()

            if use_vertex_ai:
                if google_genai_sdk is None:
                    raise ImportError("Vertex AI mode requires google-genai. Install with: pip install google-genai")
                project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT_ID")
                location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_LOCATION") or "us-central1"
                if not project:
                    raise ValueError("GOOGLE_CLOUD_PROJECT (or GOOGLE_PROJECT_ID) not found in environment")

                self.google_sdk = "google_genai"
                client_kwargs = {
                    "vertexai": True,
                    "project": project,
                    "location": location,
                }
                timeout_seconds = self._resolve_timeout_seconds()
                if timeout_seconds is not None and hasattr(google_genai_sdk, "types"):
                    # google-genai HttpOptions.timeout is in milliseconds.
                    timeout_millis = max(int(timeout_seconds * 1000.0), 10_000)
                    client_kwargs["http_options"] = google_genai_sdk.types.HttpOptions(timeout=timeout_millis)
                self.client = google_genai_sdk.Client(**client_kwargs)
            else:
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment. "
                        "For Vertex AI, set GOOGLE_GENAI_USE_VERTEXAI=true and configure GOOGLE_CLOUD_PROJECT."
                    )
                if google_genai_sdk is not None:
                    self.google_sdk = "google_genai"
                    client_kwargs = {"api_key": api_key}
                    timeout_seconds = self._resolve_timeout_seconds()
                    if timeout_seconds is not None and hasattr(google_genai_sdk, "types"):
                        # google-genai HttpOptions.timeout is in milliseconds.
                        timeout_millis = max(int(timeout_seconds * 1000.0), 10_000)
                        client_kwargs["http_options"] = google_genai_sdk.types.HttpOptions(timeout=timeout_millis)
                    self.client = google_genai_sdk.Client(**client_kwargs)
                elif google_generativeai_sdk is not None:
                    self.google_sdk = "google_generativeai"
                    google_generativeai_sdk.configure(api_key=api_key)
                    self.client = google_generativeai_sdk
                else:
                    raise ImportError(
                        "No Gemini SDK found. Install one of: pip install google-genai OR pip install google-generativeai"
                    )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat(
        self, prompt: str, system_prompt: Optional[str] = None, response_format: str = "text", **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            response_format: "text" or "json" for structured output
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary with 'content' and optionally 'parsed' (for JSON responses)
        """
        input_tokens = self._estimate_input_tokens(prompt=prompt, system_prompt=system_prompt)
        if input_tokens > self.max_input_tokens:
            raise ValueError(
                "Input prompt token limit exceeded: "
                f"{input_tokens} > {self.max_input_tokens} "
                f"(provider={self.provider}, model={self.model})"
            )

        def _single_attempt() -> Dict[str, Any]:
            if self.provider == "anthropic":
                return self._chat_anthropic(prompt, system_prompt, response_format, **kwargs)
            if self.provider == "openai":
                return self._chat_openai(prompt, system_prompt, response_format, **kwargs)
            if self.provider in {"google", "gemini"}:
                return self._chat_gemini(prompt, system_prompt, response_format, **kwargs)
            raise ValueError(f"Unsupported provider: {self.provider}")

        return self._chat_with_retries(_single_attempt)

    def estimate_input_tokens(self, prompt: str, system_prompt: Optional[str] = None) -> int:
        """Estimate input tokens without sending a provider request."""
        return self._estimate_input_tokens(prompt=prompt, system_prompt=system_prompt)

    def _chat_with_retries(self, chat_call) -> Dict[str, Any]:
        """Retry transient provider failures with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return chat_call()
            except Exception as e:
                if attempt >= self.max_retries or not self._is_retryable_error(e):
                    raise
                delay_seconds = self._compute_retry_delay_seconds(e, attempt)
                print(
                    f"[LLMClient] transient API error from {self.provider} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay_seconds:.1f}s: {e}"
                )
                time.sleep(delay_seconds)

        # Unreachable because loop either returns or raises.
        raise RuntimeError("LLM retry loop terminated unexpectedly.")

    def _get_token_encoding(self):
        if self._token_encoding is not None:
            return self._token_encoding
        try:
            encoding = tiktoken.encoding_for_model(str(self.model))
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        self._token_encoding = encoding
        return encoding

    def _estimate_input_tokens(self, prompt: str, system_prompt: Optional[str]) -> int:
        encoding = self._get_token_encoding()
        messages = [{"role": "user", "content": str(prompt or "")}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": str(system_prompt)})

        total = 0
        for message in messages:
            total += 4
            total += len(encoding.encode(message["role"]))
            total += len(encoding.encode(message["content"]))
        total += 2
        return int(total)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Detect provider errors that should be retried."""
        status_code = self._extract_status_code(error)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True

        message = str(error).lower()
        retry_markers = (
            "rate limit",
            "too many requests",
            "resource_exhausted",
            "high demand",
            "deadline exceeded",
            "timed out",
            "timeout",
            "please try again later",
            "temporarily unavailable",
            "service unavailable",
            "status': 'unavailable'",
            '"status": "unavailable"',
        )
        return any(marker in message for marker in retry_markers)

    def _resolve_timeout_seconds(self, **kwargs) -> Optional[float]:
        timeout_value = kwargs.get("timeout_seconds", kwargs.get("timeout", self.request_timeout_seconds))
        if timeout_value is None:
            return None
        try:
            timeout_seconds = float(timeout_value)
        except (TypeError, ValueError):
            return None
        if timeout_seconds <= 0:
            return None
        return timeout_seconds

    def _resolve_temperature(self, **kwargs) -> Optional[float]:
        if "temperature" in kwargs:
            temperature_value = kwargs.get("temperature")
        else:
            temperature_value = self.temperature

        if temperature_value is None:
            return None
        try:
            parsed = float(temperature_value)
        except (TypeError, ValueError):
            return None
        return parsed

    def _resolve_google_vertexai_mode(self) -> bool:
        """Whether Gemini client should use Vertex AI credentials flow."""
        env_value = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
        if env_value is not None:
            return env_value.strip().lower() in {"1", "true", "yes", "on"}

        # Auto-enable Vertex mode when a project is configured and no API key is present.
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT_ID")
        has_api_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        return bool(project) and not has_api_key

    def _resolve_anthropic_vertex_mode(self) -> bool:
        env_value = os.getenv("ANTHROPIC_USE_VERTEXAI")
        if env_value is None:
            return False
        return env_value.strip().lower() in {"1", "true", "yes", "on"}

    def _extract_status_code(self, error: Exception) -> Optional[int]:
        """Extract HTTP status code from SDK exceptions when available."""
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(error, "response", None)
        if response is not None:
            response_status = getattr(response, "status_code", None)
            if isinstance(response_status, int):
                return response_status

        return None

    def _extract_retry_after_seconds(self, error: Exception) -> Optional[float]:
        """Extract Retry-After hints from exception response headers when available."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None

        retry_after_ms = headers.get("retry-after-ms")
        if retry_after_ms is not None:
            try:
                parsed = float(retry_after_ms) / 1000.0
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass

        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                parsed = float(retry_after)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass

        return None

    def _compute_retry_delay_seconds(self, error: Exception, attempt: int) -> float:
        """Use Retry-After when present, otherwise exponential backoff."""
        retry_after_seconds = self._extract_retry_after_seconds(error)
        if retry_after_seconds is not None:
            return min(retry_after_seconds, self.retry_max_delay_seconds)

        delay = self.retry_base_delay_seconds * (2**attempt)
        return min(delay, self.retry_max_delay_seconds)

    def _chat_anthropic(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """Anthropic-specific chat implementation."""
        request_params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            request_params["system"] = system_prompt

        temperature = self._resolve_temperature(**kwargs)
        if temperature is not None:
            request_params["temperature"] = temperature

        timeout_seconds = self._resolve_timeout_seconds(**kwargs)
        if timeout_seconds is not None:
            request_params["timeout"] = timeout_seconds

        response = self.client.messages.create(**request_params)

        parts = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text_value = getattr(block, "text", None)
                if text_value:
                    parts.append(text_value)
        content = "".join(parts)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0

        result = {
            "content": content,
            "model": getattr(response, "model", None) or self.model,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

        if response_format == "json":
            try:
                result["parsed"] = json.loads(result["content"])
            except json.JSONDecodeError as e:
                result["parse_error"] = str(e)
                result["parsed"] = None

        return result

    def _chat_openai(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """OpenAI-specific chat implementation."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
        }

        # Some models (gpt-5*, o1*) only support default temperature.
        # For other models, only set temperature when explicitly provided.
        temperature = self._resolve_temperature(**kwargs)
        if temperature is not None and not (self.model.startswith("gpt-5") or self.model.startswith("o1")):
            request_params["temperature"] = temperature

        # Determine which token parameter to use based on model
        # Newer models (gpt-5*, o1*) use max_completion_tokens
        # Older models use max_tokens
        max_tokens_value = kwargs.get("max_tokens", self.max_tokens)
        if self.model.startswith("gpt-5") or self.model.startswith("o1"):
            request_params["max_completion_tokens"] = max_tokens_value
        else:
            request_params["max_tokens"] = max_tokens_value

        timeout_seconds = self._resolve_timeout_seconds(**kwargs)
        if timeout_seconds is not None:
            request_params["timeout"] = timeout_seconds

        # Add JSON mode if requested
        if response_format == "json":
            request_params["response_format"] = {"type": "json_object"}

        # Send request
        response = self.client.chat.completions.create(**request_params)

        result = {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens},
        }

        # Parse JSON if requested
        if response_format == "json":
            try:
                result["parsed"] = json.loads(result["content"])
            except json.JSONDecodeError as e:
                result["parse_error"] = str(e)
                result["parsed"] = None

        return result

    def _chat_gemini(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """Google Gemini-specific chat implementation."""
        try:
            if getattr(self, "google_sdk", None) == "google_genai":
                return self._chat_gemini_google_genai(prompt, system_prompt, response_format, **kwargs)
            return self._chat_gemini_google_generativeai(prompt, system_prompt, response_format, **kwargs)
        except Exception as e:
            err = str(e)
            if "NOT_FOUND" in err and "models/" in err:
                raise ValueError(
                    f"Model '{self.model}' is not available for Google generateContent on current project/location. "
                    "For Model Garden models, use a publisher-prefixed ID like "
                    "'qwen/qwen3-next-80b-a3b-instruct-maas' or 'anthropic/claude-sonnet-4-5@20250929', "
                    "and ensure access is granted in Vertex AI."
                ) from e
            raise

    def _chat_gemini_google_genai(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """Gemini implementation using the google-genai SDK."""
        config_kwargs = {
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        temperature = self._resolve_temperature(**kwargs)
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if response_format == "json":
            config_kwargs["response_mime_type"] = "application/json"
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        config = config_kwargs
        if google_genai_sdk is not None and hasattr(google_genai_sdk, "types"):
            config = google_genai_sdk.types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        content = getattr(response, "text", "") or ""
        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        if usage_metadata and output_tokens == 0:
            total_tokens = getattr(usage_metadata, "total_token_count", 0) or 0
            if total_tokens >= input_tokens:
                output_tokens = total_tokens - input_tokens

        result = {
            "content": content,
            "model": getattr(response, "model_version", None) or self.model,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

        if response_format == "json":
            try:
                result["parsed"] = json.loads(result["content"])
            except json.JSONDecodeError as e:
                result["parse_error"] = str(e)
                result["parsed"] = None

        return result

    def _chat_gemini_google_generativeai(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """Gemini implementation using the google-generativeai SDK."""
        generation_config = {
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        temperature = self._resolve_temperature(**kwargs)
        if temperature is not None:
            generation_config["temperature"] = temperature
        if response_format == "json":
            generation_config["response_mime_type"] = "application/json"

        if system_prompt:
            model = self.client.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
            )
        else:
            model = self.client.GenerativeModel(model_name=self.model)

        generate_content_kwargs: Dict[str, Any] = {
            "generation_config": generation_config,
        }
        timeout_seconds = self._resolve_timeout_seconds(**kwargs)
        if timeout_seconds is not None:
            generate_content_kwargs["request_options"] = {"timeout": timeout_seconds}

        response = model.generate_content(
            prompt,
            **generate_content_kwargs,
        )

        content = ""
        if getattr(response, "text", None):
            content = response.text
        else:
            candidates = getattr(response, "candidates", []) or []
            for candidate in candidates:
                candidate_content = getattr(candidate, "content", None)
                parts = getattr(candidate_content, "parts", []) if candidate_content is not None else []
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        content += part_text

        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        if usage_metadata and output_tokens == 0:
            total_tokens = getattr(usage_metadata, "total_token_count", 0) or 0
            if total_tokens >= input_tokens:
                output_tokens = total_tokens - input_tokens

        result = {
            "content": content,
            "model": getattr(response, "model_version", None) or self.model,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

        if response_format == "json":
            try:
                result["parsed"] = json.loads(result["content"])
            except json.JSONDecodeError as e:
                result["parse_error"] = str(e)
                result["parsed"] = None

        return result
