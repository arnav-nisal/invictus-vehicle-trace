"""Shared API key management and multi-key failover helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

# Ensure .env is loaded from project root once
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_nvidia_api_keys() -> List[Tuple[str, str]]:
    """Return ordered, deduplicated (key_label, secret_value) pairs for NVIDIA NIM.

    Order of priority:
      1. NVIDIA_API_KEY
      2. NVIDIA_API_KEY_1
      3. NVIDIA_API_KEY_2
      4. Any higher NVIDIA_API_KEY_<N> in ascending numerical order

    Deduplicates by secret_value (preserving the first encountered key_label).
    Logs and audit traces must ONLY store key_label, never raw secret_value.
    """
    candidate_labels = ["NVIDIA_API_KEY"]

    # Discover any numbered keys like NVIDIA_API_KEY_1, NVIDIA_API_KEY_2, etc.
    numbered: List[Tuple[int, str]] = []
    for k in os.environ.keys():
        m = re.fullmatch(r"NVIDIA_API_KEY_(\d+)", k)
        if m:
            numbered.append((int(m.group(1)), k))
    numbered.sort(key=lambda x: x[0])

    for _, label in numbered:
        if label not in candidate_labels:
            candidate_labels.append(label)

    results: List[Tuple[str, str]] = []
    seen_secrets = set()

    for label in candidate_labels:
        secret = os.getenv(label, "").strip()
        if secret and secret not in seen_secrets:
            seen_secrets.add(secret)
            results.append((label, secret))

    return results
