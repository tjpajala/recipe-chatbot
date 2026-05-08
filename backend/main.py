from __future__ import annotations

"""FastAPI application entry-point for the recipe chatbot."""

from pathlib import Path
from typing import Final, List, Dict, Any, Optional
import datetime
import json
import os
import re

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.utils import (
    get_agent_response,
    get_cache_key,
    load_evals_cache,
    save_evals_cache,
    run_judge_evaluation,
    calculate_judgy_metrics,
    parse_trace_timestamp,
    SYSTEM_PROMPT,
    TRACES_DIR,
    TEST_CASES_PATH,
    JUDGE_PROMPT_PATH,
)  # noqa: WPS433 import from parent

# Configure logging
import logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Application setup
# -----------------------------------------------------------------------------

APP_TITLE: Final[str] = "Recipe Chatbot"
app = FastAPI(title=APP_TITLE)

# Serve static assets (currently just the HTML) under `/static/*`.
STATIC_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -----------------------------------------------------------------------------
# Request / response models
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Schema for a single message in the chat history."""
    role: str = Field(..., description="Role of the message sender (system, user, or assistant).")
    content: str = Field(..., description="Content of the message.")

class ChatRequest(BaseModel):
    """Schema for incoming chat messages."""

    messages: List[ChatMessage] = Field(..., description="The entire conversation history.")


class ChatResponse(BaseModel):
    """Schema for the assistant's reply returned to the front-end."""

    messages: List[ChatMessage] = Field(..., description="The updated conversation history.")


class TraceItem(BaseModel):
    """Schema for a single trace item in the list view."""

    id: str = Field(..., description="Trace identifier (filename without extension).")
    timestamp: str = Field(..., description="ISO format timestamp.")
    preview: str = Field(..., description="Preview of the first user message.")
    message_count: int = Field(..., description="Total number of messages in the trace.")


class TraceListResponse(BaseModel):
    """Schema for paginated trace list response."""

    traces: List[TraceItem] = Field(..., description="List of trace items for current page.")
    total: int = Field(..., description="Total number of traces.")
    page: int = Field(..., description="Current page number.")
    total_pages: int = Field(..., description="Total number of pages.")


class TestCase(BaseModel):
    """Schema for a single test case."""

    id: int = Field(..., description="Test case ID.")
    query: str = Field(..., description="Test query.")


class EvalsConfigResponse(BaseModel):
    """Schema for evals configuration response."""

    test_cases: List[TestCase] = Field(..., description="Default test cases.")
    judge_prompt: str = Field(..., description="Default judge prompt.")
    system_prompt: str = Field(..., description="Current system prompt.")
    system_prompt_hash: str = Field(..., description="Hash of current system prompt.")


class EvalsRunRequest(BaseModel):
    """Schema for running evals."""

    test_cases: List[TestCase] = Field(..., description="Test cases to evaluate.")
    judge_prompt: str = Field(..., description="Judge prompt for evaluation.")
    tpr: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="True Positive Rate (judge accuracy, must be in [0,1]).")
    tnr: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="True Negative Rate (judge specificity, must be in [0,1]).")


class EvalResult(BaseModel):
    """Schema for a single evaluation result."""

    id: int = Field(..., description="Test case ID.")
    query: str = Field(..., description="Test query.")
    output: str = Field(..., description="Generated chatbot output.")
    reasoning: str = Field(..., description="Judge reasoning.")
    result: str = Field(..., description="PASS or FAIL.")
    from_cache: bool = Field(..., description="Whether output was from cache.")


class EvalsRunResponse(BaseModel):
    """Schema for evals run response."""

    results: List[EvalResult] = Field(..., description="Individual evaluation results.")
    metrics: Dict[str, Any] = Field(..., description="Aggregated metrics from judgy.")
    system_prompt_hash: str = Field(..., description="Hash of system prompt used.")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:  # noqa: WPS430
    """Main conversational endpoint.

    It proxies the user's message list to the underlying agent and returns the updated list.
    """
    # Convert Pydantic models to simple dicts for the agent
    request_messages: List[Dict[str, str]] = [msg.model_dump() for msg in payload.messages]

    try:
        updated_messages_dicts = get_agent_response(request_messages)
    except Exception as exc:  # noqa: BLE001 broad; surface as HTTP 500
        # In production you would log the traceback here.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    response = ChatResponse(messages=[ChatMessage(**msg) for msg in updated_messages_dicts])

    # Save trace (request and response) in one place
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_path = TRACES_DIR / f"trace_{ts}.json"
    with open(trace_path, "w") as f:
        json.dump({
            "request": payload.model_dump(),
            "response": response.model_dump()
        }, f)

    return response


@app.get("/api/traces", response_model=TraceListResponse)
async def list_traces(page: int = 1, per_page: int = 20) -> TraceListResponse:
    """List all traces with pagination.

    Returns paginated metadata about available traces for the traces list view.
    """
    if not TRACES_DIR.exists():
        return TraceListResponse(traces=[], total=0, page=1, total_pages=0)

    # Get all trace files sorted by modification time (newest first)
    trace_files = sorted(
        [f for f in TRACES_DIR.glob("trace_*.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    total = len(trace_files)
    total_pages = (total + per_page - 1) // per_page if total > 0 else 0

    # Validate page number
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages

    # Apply pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_files = trace_files[start_idx:end_idx]

    # Extract metadata for each trace
    items = []
    for trace_file in page_files:
        try:
            with open(trace_file, "r") as f:
                trace_data = json.load(f)

            # Extract and parse timestamp from filename
            filename_stem = trace_file.stem  # trace_YYYY-MM-DD_HHMMSS
            timestamp_str = filename_stem.replace("trace_", "")
            dt = parse_trace_timestamp(timestamp_str)
            timestamp_iso = dt.isoformat()

            # Extract first user message for preview
            request_messages = trace_data.get("request", {}).get("messages", [])
            preview = "No user message"
            for msg in request_messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    preview = content[:100] + "..." if len(content) > 100 else content
                    break

            # Count messages in response
            response_messages = trace_data.get("response", {}).get("messages", [])
            message_count = len(response_messages)

            items.append(TraceItem(
                id=trace_file.stem,
                timestamp=timestamp_iso,
                preview=preview,
                message_count=message_count
            ))
        except Exception:
            # Skip malformed traces
            continue

    return TraceListResponse(
        traces=items,
        total=total,
        page=page,
        total_pages=total_pages
    )


@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str) -> Dict:
    """Retrieve a single trace by ID.

    Returns the full trace data including request and response messages.
    """
    # Security: Validate trace_id to prevent path traversal
    # Format: trace_YYYY-MM-DD_HHMMSS or trace_YYYYMMDD_HHMMSS_ffffff
    if not re.match(r"^trace_[\d-]+_\d+(_\d+)?$", trace_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trace ID format"
        )

    trace_path = TRACES_DIR / f"{trace_id}.json"

    if not trace_path.exists() or not trace_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trace not found"
        )

    try:
        with open(trace_path, "r") as f:
            trace_data = json.load(f)

        # Add timestamp to response
        timestamp_str = trace_id.replace("trace_", "")
        dt = parse_trace_timestamp(timestamp_str)

        trace_data["id"] = trace_id
        trace_data["timestamp"] = dt.isoformat()

        return trace_data
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse trace file"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read trace: {str(exc)}"
        ) from exc


@app.get("/api/evals/config", response_model=EvalsConfigResponse)
async def get_evals_config() -> EvalsConfigResponse:
    """Get default configuration for evals dashboard.

    Returns default test cases, judge prompt, and current system prompt.
    """
    # Security: Verify paths are within expected directories
    if not TEST_CASES_PATH.is_relative_to(Path(__file__).parent.parent / "data"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid test cases path"
        )

    if not JUDGE_PROMPT_PATH.is_relative_to(Path(__file__).parent.parent / "data"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid judge prompt path"
        )

    # Load test cases
    if not TEST_CASES_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test cases file not found"
        )

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases_data = json.load(f)

    # Load default judge prompt
    if not JUDGE_PROMPT_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Default judge prompt file not found"
        )

    judge_prompt = JUDGE_PROMPT_PATH.read_text(encoding="utf-8").strip()

    return EvalsConfigResponse(
        test_cases=[TestCase(**tc) for tc in test_cases_data],
        judge_prompt=judge_prompt,
        system_prompt=SYSTEM_PROMPT,
        system_prompt_hash=get_cache_key()
    )


@app.post("/api/evals/run", response_model=EvalsRunResponse)
async def run_evals(payload: EvalsRunRequest) -> EvalsRunResponse:
    """Run evaluation on test cases with caching.

    Generates outputs for test cases (using cache when valid), then evaluates
    them using the judge prompt. Returns individual results and aggregated metrics.
    """
    # Validate input
    if not payload.test_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one test case is required"
        )

    current_cache_key = get_cache_key()
    cache = load_evals_cache()

    # Check if cache is valid (system prompt + model must match)
    cache_valid = cache.get("cache_key") == current_cache_key

    # Build a lookup for cached outputs
    cached_outputs_map = {}
    if cache_valid:
        for cached_item in cache.get("cached_outputs", []):
            cached_outputs_map[cached_item["query"]] = cached_item["output"]

    # Generate outputs for each test case (use cache when available)
    outputs = []
    for test_case in payload.test_cases:
        query = test_case.query
        from_cache = False

        if cache_valid and query in cached_outputs_map:
            # Use cached output
            output = cached_outputs_map[query]
            from_cache = True
        else:
            # Generate new output
            messages = [{"role": "user", "content": query}]
            try:
                updated_messages = get_agent_response(messages)
                # Extract assistant's response
                output = next(
                    (msg["content"] for msg in reversed(updated_messages) if msg["role"] == "assistant"),
                    "No response generated"
                )
            except Exception as exc:
                # In production, log the traceback here
                logger.error(f"Failed to generate output for query '{query}': {exc}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate output for query '{query}': {str(exc)}"
                ) from exc

        outputs.append({
            "id": test_case.id,
            "query": query,
            "output": output,
            "from_cache": from_cache
        })

    # Update cache with new outputs
    # Note: Reload cache to handle potential concurrent modifications
    current_cache = load_evals_cache()
    new_cache = {
        "cache_key": current_cache_key,
        "cached_outputs": [
            {"query": item["query"], "output": item["output"]}
            for item in outputs
        ],
        "last_updated": datetime.datetime.now().isoformat()
    }

    try:
        save_evals_cache(new_cache)
    except IOError as e:
        # Log error but continue - cache failure shouldn't break evaluation
        # In production, log the traceback here
        logger.error(f"Failed to save cache: {e}")
        # Don't raise - we can still return results even if cache save fails

    # Run judge evaluation on all outputs
    eval_results = []
    for item in outputs:
        try:
            judge_result = run_judge_evaluation(
                query=item["query"],
                output=item["output"],
                judge_prompt=payload.judge_prompt
            )

            eval_results.append(EvalResult(
                id=item["id"],
                query=item["query"],
                output=item["output"],
                reasoning=judge_result.get("reasoning", "No reasoning provided"),
                result=judge_result.get("result", "FAIL"),
                from_cache=item["from_cache"]
            ))
        except Exception as exc:
            # If evaluation fails, mark as FAIL with error reasoning
            # In production, log the traceback here
            logger.error(f"Judge evaluation failed for test case {item['id']}: {exc}")
            eval_results.append(EvalResult(
                id=item["id"],
                query=item["query"],
                output=item["output"],
                reasoning=f"Evaluation error: {str(exc)}",
                result="FAIL",
                from_cache=item["from_cache"]
            ))

    # Calculate aggregated metrics using judgy methodology
    metrics = calculate_judgy_metrics(
        results=[{"result": r.result} for r in eval_results],
        tpr=payload.tpr or 0.95,
        tnr=payload.tnr or 0.95
    )

    return EvalsRunResponse(
        results=eval_results,
        metrics=metrics,
        system_prompt_hash=current_cache_key
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:  # noqa: WPS430
    """Serve the chat UI."""

    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend not found. Did you forget to build it?",
        )

    return HTMLResponse(html_path.read_text(encoding="utf-8")) 