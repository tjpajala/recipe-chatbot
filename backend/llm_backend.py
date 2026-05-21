"""Centralized LLM backend using LiteLLM for multi-provider support.

This module provides a unified interface for all LLM calls in the application,
making it easy to switch between providers (OpenAI, Anthropic, etc.) and
centralizing configuration, error handling, and logging.
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict, Any, Optional, Literal

import litellm  # type: ignore
from litellm import completion  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)

# Configure LiteLLM settings
litellm.drop_params = True  # Drop unsupported params instead of erroring
litellm.set_verbose = False  # Disable verbose logging by default


class LLMBackend:
    """Centralized backend for all LLM interactions using LiteLLM.

    This class provides a unified interface for different LLM use cases:
    - Chat completions (multi-turn conversations)
    - Single completions (one-off generations)
    - Judge evaluations (structured outputs)

    LiteLLM automatically handles provider-specific formatting and routing.
    """

    def __init__(
        self,
        default_model: str = "gpt-4o-mini",
        default_temperature: float = 0.7,
        default_max_tokens: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """Initialize the LLM backend.

        Parameters
        ----------
        default_model : str
            Default model to use (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022")
        default_temperature : float
            Default temperature for completions (0.0 to 1.0)
        default_max_tokens : Optional[int]
            Default max tokens for completions
        max_retries : int
            Maximum number of retries for failed requests
        retry_delay : float
            Base delay between retries (uses exponential backoff)
        """
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dicts with "role" and "content" keys
        model : Optional[str]
            Model to use (overrides default)
        temperature : Optional[float]
            Temperature for sampling (overrides default)
        max_tokens : Optional[int]
            Max tokens to generate (overrides default)
        **kwargs : Any
            Additional arguments to pass to litellm.completion()

        Returns
        -------
        str
            The assistant's response content

        Raises
        ------
        Exception
            If all retry attempts fail
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature

        # Build completion parameters
        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        if max_tokens is not None or self.default_max_tokens is not None:
            completion_params["max_tokens"] = max_tokens or self.default_max_tokens

        # Execute with retry logic
        response = self._execute_with_retry(completion_params)

        # Extract content from response
        content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        return content.strip()

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Send a single completion request (convenience wrapper around chat).

        Parameters
        ----------
        prompt : str
            The user prompt
        model : Optional[str]
            Model to use (overrides default)
        temperature : Optional[float]
            Temperature for sampling (overrides default)
        max_tokens : Optional[int]
            Max tokens to generate (overrides default)
        system_prompt : Optional[str]
            Optional system prompt to prepend
        **kwargs : Any
            Additional arguments to pass to litellm.completion()

        Returns
        -------
        str
            The assistant's response content
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _execute_with_retry(self, completion_params: Dict[str, Any]) -> Any:
        """Execute a completion request with exponential backoff retry logic.

        Parameters
        ----------
        completion_params : Dict[str, Any]
            Parameters to pass to litellm.completion()

        Returns
        -------
        Any
            The completion response object

        Raises
        ------
        Exception
            If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = completion(**completion_params)
                return response

            except Exception as e:
                last_exception = e

                # Log the error
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                # Don't retry on last attempt
                if attempt == self.max_retries - 1:
                    break

                # Exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)

        # All retries failed
        logger.error(f"All {self.max_retries} retry attempts failed for LLM request")
        raise last_exception  # type: ignore[misc]

    def create_judge_backend(
        self,
        judge_model: Optional[str] = None,
    ) -> "JudgeBackend":
        """Create a specialized backend for judge evaluations.

        Parameters
        ----------
        judge_model : Optional[str]
            Model to use for judge evaluations (defaults to main model)

        Returns
        -------
        JudgeBackend
            A backend configured for judge evaluations
        """
        return JudgeBackend(
            model=judge_model or self.default_model,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )


class JudgeBackend:
    """Specialized backend for LLM-as-judge evaluations.

    Provides deterministic evaluations with structured output handling.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """Initialize the judge backend.

        Parameters
        ----------
        model : str
            Model to use for judge evaluations
        max_retries : int
            Maximum number of retries for failed requests
        retry_delay : float
            Base delay between retries
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Create underlying LLM backend with deterministic settings
        self.llm = LLMBackend(
            default_model=model,
            default_temperature=0.0,  # Deterministic for judges
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def evaluate(
        self,
        query: str,
        output: str,
        judge_prompt: str,
    ) -> str:
        """Run a judge evaluation.

        Parameters
        ----------
        query : str
            The user query/test case
        output : str
            The system's output to evaluate
        judge_prompt : str
            The judge prompt with evaluation criteria

        Returns
        -------
        str
            The judge's evaluation response
        """
        evaluation_messages = [
            {"role": "system", "content": judge_prompt},
            {
                "role": "user",
                "content": f"User Query: {query}\n\nAssistant Response: {output}\n\nEvaluation:",
            },
        ]

        return self.llm.chat(
            messages=evaluation_messages,
            temperature=0.0,  # Always deterministic for judges
        )


# Global backend instance (can be configured via environment variables)
_default_backend: Optional[LLMBackend] = None


def get_default_backend() -> LLMBackend:
    """Get or create the default LLM backend instance.

    Returns
    -------
    LLMBackend
        The default backend instance
    """
    global _default_backend

    if _default_backend is None:
        import os

        # Read configuration from environment
        model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        temperature = float(os.environ.get("MODEL_TEMPERATURE", "0.7"))

        _default_backend = LLMBackend(
            default_model=model,
            default_temperature=temperature,
        )

    return _default_backend


def set_default_backend(backend: LLMBackend) -> None:
    """Set the default LLM backend instance.

    Parameters
    ----------
    backend : LLMBackend
        The backend to use as default
    """
    global _default_backend
    _default_backend = backend
