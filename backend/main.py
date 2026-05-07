from __future__ import annotations

"""FastAPI application entry-point for the recipe chatbot."""

from pathlib import Path
from typing import Final, List, Dict
import datetime
import json
import os
import re

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.utils import get_agent_response  # noqa: WPS433 import from parent

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
    traces_dir = Path(__file__).parent.parent / "annotation" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_path = traces_dir / f"trace_{ts}.json"
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
    traces_dir = Path(__file__).parent.parent / "annotation" / "traces"

    if not traces_dir.exists():
        return TraceListResponse(traces=[], total=0, page=1, total_pages=0)

    # Get all trace files sorted by modification time (newest first)
    trace_files = sorted(
        [f for f in traces_dir.glob("trace_*.json")],
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

            # Extract timestamp from filename: trace_YYYY-MM-DD_HHMMSS.json
            filename_stem = trace_file.stem  # trace_YYYY-MM-DD_HHMMSS
            timestamp_str = filename_stem.replace("trace_", "")

            # Parse timestamp - handle multiple formats
            try:
                # Format: YYYY-MM-DD_HHMMSS
                dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
            except ValueError:
                try:
                    # Format: YYYYMMDD_HHMMSS_ffffff
                    dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
                except ValueError:
                    # Format: YYYYMMDD_HHMMSS
                    dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

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

    traces_dir = Path(__file__).parent.parent / "annotation" / "traces"
    trace_path = traces_dir / f"{trace_id}.json"

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
        try:
            # Format: YYYY-MM-DD_HHMMSS
            dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
        except ValueError:
            try:
                # Format: YYYYMMDD_HHMMSS_ffffff
                dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
            except ValueError:
                # Format: YYYYMMDD_HHMMSS
                dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

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