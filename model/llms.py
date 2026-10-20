"""Explicit model configuration with provider SDKs loaded only when needed."""

import math
import os
import time


class LLMClient:
    def __init__(
        self, *, provider: str, model: str, connection: dict,
        temperature: float | None, max_tokens: int, max_input_tokens: int,
        token_encoding: str, max_retries: int, retry_base_delay_seconds: float,
        retry_max_delay_seconds: float, request_timeout_seconds: float,
    ):
        if provider not in {"openai", "anthropic", "google"} or not model.strip():
            raise ValueError("An explicit supported provider and model are required")
        if any(type(value) is not int or value < 1 for value in (max_tokens, max_input_tokens)):
            raise ValueError("Token limits must be positive integers")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        for value in (retry_base_delay_seconds, retry_max_delay_seconds, request_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Retry delays and request timeout must be finite and positive")
        if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
            raise ValueError("temperature must be finite and nonnegative, or null to omit it")
        if not token_encoding.strip():
            raise ValueError("token_encoding is required")
        backend = connection["backend"]
        if backend not in {"api", "vertex"} or (provider == "openai" and backend != "api"):
            raise ValueError(f"Unsupported backend {backend} for {provider}")
        if backend == "vertex" and any(not connection[key].strip() for key in ("project", "location")):
            raise ValueError("Vertex requires explicit project and location")
        self.provider = provider
        self.model = model
        self.connection = connection
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.token_encoding = token_encoding
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._client = None
        self._encoding = None
        if provider != "openai" and connection.get("base_url") is not None:
            raise ValueError("connection.base_url is supported only for the OpenAI-compatible API")

    def validate_credentials(self) -> None:
        if self.connection["backend"] == "api":
            name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}[self.provider]
            self._api_key(name)

    @property
    def client(self):
        if self._client is not None:
            return self._client
        backend = self.connection["backend"]
        if self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key("OPENAI_API_KEY"), base_url=self.connection["base_url"],
                timeout=self.request_timeout_seconds, max_retries=0,
            )
        elif self.provider == "anthropic":
            from anthropic import Anthropic, AnthropicVertex

            if backend == "vertex":
                self._client = AnthropicVertex(
                    project_id=self.connection["project"], region=self.connection["location"],
                    timeout=self.request_timeout_seconds, max_retries=0,
                )
            else:
                self._client = Anthropic(
                    api_key=self._api_key("ANTHROPIC_API_KEY"),
                    timeout=self.request_timeout_seconds, max_retries=0,
                )
        else:
            from google import genai
            from google.genai import types

            options = {"http_options": types.HttpOptions(
                timeout=int(self.request_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            )}
            if backend == "vertex":
                options.update(vertexai=True, project=self.connection["project"], location=self.connection["location"])
            else:
                options["api_key"] = self._api_key("GOOGLE_API_KEY")
            self._client = genai.Client(**options)
        return self._client

    @staticmethod
    def _api_key(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"Missing API credential: {name}")
        return value

    def estimate_input_tokens(self, prompt: str, system_prompt: str | None = None) -> int:
        if self._encoding is None:
            import tiktoken

            self._encoding = tiktoken.get_encoding(self.token_encoding)
        return len(self._encoding.encode((system_prompt or "") + prompt, disallowed_special=()))

    def chat(self, prompt: str, system_prompt: str | None = None, response_format: str = "text") -> dict:
        tokens = self.estimate_input_tokens(prompt, system_prompt)
        if tokens > self.max_input_tokens:
            raise ValueError(f"Input context length {tokens} exceeds configured limit {self.max_input_tokens}")
        for attempt in range(self.max_retries + 1):
            try:
                return self._generate(prompt, system_prompt, response_format)
            except Exception as error:
                status = getattr(error, "status_code", getattr(error, "code", None))
                transient = isinstance(error, (TimeoutError, ConnectionError)) or (
                    isinstance(status, int) and (status == 429 or status >= 500)
                ) or type(error).__name__ in {"APITimeoutError", "APIConnectionError"}
                if not transient or attempt == self.max_retries:
                    raise
                time.sleep(min(self.retry_max_delay_seconds, self.retry_base_delay_seconds * 2 ** attempt))
        raise RuntimeError("Unreachable retry state")

    def _generate(self, prompt: str, system_prompt: str | None, response_format: str) -> dict:
        options = {} if self.temperature is None else {"temperature": self.temperature}
        if self.provider == "openai":
            if self.model.startswith(("gpt-5", "o1")):
                options.pop("temperature", None)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            token_key = "max_completion_tokens" if self.model.startswith(("gpt-5", "o1")) else "max_tokens"
            if response_format == "json":
                options["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **{token_key: self.max_tokens}, **options,
            )
            return {"content": response.choices[0].message.content, "usage": {
                "input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens,
            }}
        if self.provider == "anthropic":
            if system_prompt:
                options["system"] = system_prompt
            response = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}], **options,
            )
            return {"content": "".join(block.text for block in response.content if block.type == "text"), "usage": {
                "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
            }}
        from google.genai import types

        if system_prompt:
            options["system_instruction"] = system_prompt
        if response_format == "json":
            options["response_mime_type"] = "application/json"
        response = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=self.max_tokens, **options),
        )
        usage = response.usage_metadata
        return {"content": response.text, "usage": {
            "input_tokens": usage.prompt_token_count, "output_tokens": usage.candidates_token_count,
        }}
