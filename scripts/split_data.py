"""Split labeled traces into train/dev/test sets.

Reads all labeled traces from annotation/traces/, shuffles them with a fixed seed,
and splits them into train (15%), dev (40%), test (45%) sets.

Outputs:
- data/train.jsonl
- data/dev.jsonl
- data/test.jsonl
- data/splits_metadata.json (metadata about which trace IDs are in which split)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
TRACES_DIR = PROJECT_ROOT / "annotation" / "traces"
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_RATIO = 0.15
DEV_RATIO = 0.40
TEST_RATIO = 0.45
DEFAULT_SEED = 42


def load_labeled_traces() -> List[Dict[str, Any]]:
    """Load all labeled traces from the traces directory.

    Returns:
        List of trace dictionaries that have been labeled.
    """
    if not TRACES_DIR.exists():
        print(f"Error: Traces directory not found: {TRACES_DIR}")
        return []

    labeled_traces = []
    trace_files = list(TRACES_DIR.glob("trace_*.json"))

    print(f"Found {len(trace_files)} trace files")

    for trace_file in trace_files:
        try:
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)

            # Only include traces that have been labeled
            if not trace_data.get("labeled"):
                continue

            # Extract the fields we need
            request_messages = trace_data.get("request", {}).get("messages", [])
            response_messages = trace_data.get("response", {}).get("messages", [])

            # Get query (first user message)
            query = ""
            for msg in request_messages:
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break

            # Get response (last assistant message)
            response = ""
            for msg in reversed(response_messages):
                if msg.get("role") == "assistant":
                    response = msg.get("content", "")
                    break

            # Create simplified trace object
            simplified_trace = {
                "trace_id": trace_file.stem,
                "query": query,
                "response": response,
                "label": trace_data.get("label", ""),
                "reasoning": trace_data.get("reasoning", ""),
                "confidence": trace_data.get("confidence", "")
            }

            labeled_traces.append(simplified_trace)

        except Exception as e:
            print(f"Warning: Failed to load {trace_file}: {e}")
            continue

    print(f"Loaded {len(labeled_traces)} labeled traces")
    return labeled_traces


def split_traces(traces: List[Dict[str, Any]], seed: int = DEFAULT_SEED) -> Dict[str, List[Dict[str, Any]]]:
    """Split traces into train/dev/test sets.

    Args:
        traces: List of trace dictionaries
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'dev', and 'test' keys containing their respective traces
    """
    if not traces:
        return {"train": [], "dev": [], "test": []}

    # Shuffle with fixed seed
    random.seed(seed)
    shuffled = traces.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * TRAIN_RATIO)
    dev_end = int(n * (TRAIN_RATIO + DEV_RATIO))

    splits = {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:]
    }

    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])} ({len(splits['train'])/n*100:.1f}%)")
    print(f"  Dev:   {len(splits['dev'])} ({len(splits['dev'])/n*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} ({len(splits['test'])/n*100:.1f}%)")

    return splits


def save_splits(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Save splits to JSONL files and metadata.

    Args:
        splits: Dictionary with 'train', 'dev', and 'test' keys
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save each split to JSONL
    for split_name, split_traces in splits.items():
        output_path = DATA_DIR / f"{split_name}.jsonl"
        with open(output_path, 'w') as f:
            for trace in split_traces:
                f.write(json.dumps(trace) + '\n')
        print(f"Saved {len(split_traces)} traces to {output_path}")

    # Save metadata (trace IDs by split)
    metadata = {
        "train_ids": [t["trace_id"] for t in splits["train"]],
        "dev_ids": [t["trace_id"] for t in splits["dev"]],
        "test_ids": [t["trace_id"] for t in splits["test"]],
        "train_count": len(splits["train"]),
        "dev_count": len(splits["dev"]),
        "test_count": len(splits["test"]),
        "total_count": sum(len(s) for s in splits.values()),
        "seed": DEFAULT_SEED
    }

    metadata_path = DATA_DIR / "splits_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved split metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Split labeled traces into train/dev/test sets")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling (default: {DEFAULT_SEED})"
    )
    args = parser.parse_args()

    print("Loading labeled traces...")
    traces = load_labeled_traces()

    if not traces:
        print("\nError: No labeled traces found!")
        print("Please label some traces first using the Label tab in the app.")
        return

    print(f"\nSplitting {len(traces)} traces with seed={args.seed}...")
    splits = split_traces(traces, seed=args.seed)

    print("\nSaving splits to JSONL files...")
    save_splits(splits)

    print("\n✅ Data split complete!")
    print("\nNext steps:")
    print("1. View train set examples to select few-shot examples")
    print("2. Write your judge prompt based on train set")
    print("3. Validate judge on dev set")
    print("4. Report final metrics on test set")


if __name__ == "__main__":
    main()
