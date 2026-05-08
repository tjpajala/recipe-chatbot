from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import datetime
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Final, List, Dict, Any, Optional

import litellm  # type: ignore
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

# Load system prompt from markdown file
_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT: Final[str] = _PROMPT_PATH.read_text().strip()

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")
MODEL_NAME_JUDGE: Final[str] = os.environ.get("MODEL_NAME_JUDGE", "gpt-4o-mini")

# File paths
_BASE_DIR = Path(__file__).parent.parent
EVALS_CACHE_PATH: Final[Path] = _BASE_DIR / "data" / "evals_cache.json"
TEST_CASES_PATH: Final[Path] = _BASE_DIR / "data" / "evals_test_cases.json"
JUDGE_PROMPT_PATH: Final[Path] = _BASE_DIR / "data" / "evals_default_judge_prompt.md"
TRACES_DIR: Final[Path] = _BASE_DIR / "annotation" / "traces"


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )

    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages


# --- Trace utilities ------------------------------------------------------------

def parse_trace_timestamp(timestamp_str: str) -> datetime.datetime:
    """Parse trace timestamp from multiple possible formats.

    Parameters
    ----------
    timestamp_str : str
        Timestamp string extracted from trace filename.

    Returns
    -------
    datetime.datetime
        Parsed datetime object.

    Raises
    ------
    ValueError
        If timestamp cannot be parsed in any known format.
    """
    formats = [
        "%Y-%m-%d_%H%M%S",
        "%Y%m%d_%H%M%S_%f",
        "%Y%m%d_%H%M%S"
    ]

    for fmt in formats:
        try:
            return datetime.datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse timestamp: {timestamp_str}")


# --- Evals utilities ------------------------------------------------------------

def get_cache_key() -> str:
    """Generate cache key from system prompt and model name.

    Cache should be invalidated when either the system prompt or model changes.

    Returns
    -------
    str
        Cache key combining prompt hash and model name.
    """
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode('utf-8')).hexdigest()
    return f"{prompt_hash}:{MODEL_NAME}"


def load_evals_cache() -> Dict[str, Any]:
    """Load the evals cache from disk.

    Returns
    -------
    Dict[str, Any]
        Cache data with cache_key and cached_outputs.
        Returns empty structure if cache doesn't exist.
    """
    if not EVALS_CACHE_PATH.exists():
        return {
            "cache_key": "",
            "cached_outputs": []
        }

    try:
        with open(EVALS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load evals cache: {e}. Returning empty cache.")
        return {
            "cache_key": "",
            "cached_outputs": []
        }


def save_evals_cache(cache_data: Dict[str, Any]) -> None:
    """Save the evals cache to disk.

    Parameters
    ----------
    cache_data : Dict[str, Any]
        Cache data to save.

    Raises
    ------
    IOError
        If cache cannot be written to disk (disk full, permission denied, etc.)
    """
    EVALS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Write to temporary file first for atomicity
        temp_path = EVALS_CACHE_PATH.with_suffix('.json.tmp')
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        # Atomic rename (prevents partial writes)
        temp_path.replace(EVALS_CACHE_PATH)

    except (IOError, OSError) as e:
        # In production, log the traceback here
        logger.error(f"Failed to save evals cache: {e}")
        raise IOError(f"Failed to save cache to disk: {e}") from e


def _extract_json_from_response(response: str) -> str:
    """Extract JSON from markdown code blocks or raw text.

    Parameters
    ----------
    response : str
        LLM response potentially containing JSON in markdown code blocks.

    Returns
    -------
    str
        Extracted JSON string.
    """
    # Try to find JSON in markdown code blocks using regex
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def run_judge_evaluation(query: str, output: str, judge_prompt: str) -> Dict[str, Any]:
    """Run LLM-as-judge evaluation on a single query-output pair.

    Parameters
    ----------
    query : str
        The user query/test case.
    output : str
        The chatbot's generated output.
    judge_prompt : str
        The judge prompt with evaluation criteria.

    Returns
    -------
    Dict[str, Any]
        Evaluation result with 'reasoning' and 'result' (PASS/FAIL).
    """
    evaluation_messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": f"User Query: {query}\n\nAssistant Response: {output}\n\nEvaluation:"}
    ]

    completion = litellm.completion(
        model=MODEL_NAME_JUDGE,
        messages=evaluation_messages,
        temperature=0.0  # Use deterministic evaluation
    )

    judge_response = completion["choices"][0]["message"]["content"].strip()  # type: ignore[index]

    # Extract and parse JSON from response
    try:
        json_str = _extract_json_from_response(judge_response)
        result = json.loads(json_str)

        # Ensure result has required fields
        if "reasoning" not in result or "result" not in result:
            raise ValueError("Missing required fields: 'reasoning' and 'result'")

        # Normalize result to uppercase
        result["result"] = result["result"].upper()

        return result

    except (json.JSONDecodeError, ValueError) as e:
        # In production, log the traceback here
        logger.warning(f"Failed to parse judge response: {e}. Response: {judge_response[:100]}")
        return {
            "reasoning": f"Failed to parse judge response: {str(e)}",
            "result": "FAIL",
            "raw_response": judge_response
        }


def calculate_judgy_metrics(
    results: List[Dict[str, Any]],
    tpr: float = 0.95,
    tnr: float = 0.95
) -> Dict[str, Any]:
    """Calculate bias-corrected evaluation metrics using judgy methodology.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of evaluation results with 'result' field (PASS/FAIL).
    tpr : float
        True Positive Rate (sensitivity) of the judge. Default: 0.95
    tnr : float
        True Negative Rate (specificity) of the judge. Default: 0.95

    Returns
    -------
    Dict[str, Any]
        Metrics including raw_pass_rate, corrected_pass_rate, confidence_interval.
    """
    if not results:
        return {
            "sample_size": 0,
            "raw_pass_count": 0,
            "raw_pass_rate": 0.0,
            "corrected_pass_rate": 0.0,
            "confidence_interval": [0.0, 0.0],
            "tpr": tpr,
            "tnr": tnr,
            "using_defaults": True
        }

    # Count passes
    pass_count = sum(1 for r in results if r.get("result") == "PASS")
    n = len(results)
    p_obs = pass_count / n  # Observed pass rate

    # Bias correction formula from judgy
    # θ̂ = (p_obs + TNR - 1) / (TPR + TNR - 1)
    # Where θ̂ is the corrected success rate

    denominator = tpr + tnr - 1
    if abs(denominator) < 0.001:  # Avoid division by zero
        corrected_rate = p_obs
    else:
        corrected_rate = (p_obs + tnr - 1) / denominator

    # Clamp to [0, 1]
    corrected_rate = max(0.0, min(1.0, corrected_rate))

    # Calculate 95% confidence interval using Wilson score interval
    # Simplified approximation for the corrected estimate
    import math

    z = 1.96  # 95% confidence

    # Variance estimate for corrected rate
    # This is a simplified approximation
    p = corrected_rate
    variance = p * (1 - p) / n
    se = math.sqrt(variance)

    # Confidence interval
    ci_lower = max(0.0, p - z * se)
    ci_upper = min(1.0, p + z * se)

    return {
        "sample_size": n,
        "raw_pass_count": pass_count,
        "raw_pass_rate": p_obs,
        "corrected_pass_rate": corrected_rate,
        "confidence_interval": [ci_lower, ci_upper],
        "correction_applied": corrected_rate - p_obs,
        "tpr": tpr,
        "tnr": tnr,
        "using_defaults": True  # Mark that we're using default TPR/TNR
    } 