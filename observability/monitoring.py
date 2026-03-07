"""
monitoring.py — Prometheus metrics, structured JSON logging, and observability hooks.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Prometheus Metrics Definitions (mocked fallback if not available)
if PROMETHEUS_AVAILABLE:
    QUERY_LATENCY = Histogram(
        "rag_query_latency_seconds",
        "Latency of RAG queries in seconds",
        ["repo"]
    )
    TOKEN_USAGE = Counter(
        "rag_token_usage_total",
        "Total LLM tokens consumed",
        ["repo", "model"]
    )
    QUERY_ERRORS = Counter(
        "rag_query_errors_total",
        "Total failed RAG queries",
        ["repo"]
    )
else:
    # Dummy mock class to prevent import crashes
    class DummyMetric:
        def labels(self, *args, **kwargs):
            return self
        def observe(self, amount):
            pass
        def inc(self, amount=1):
            pass

    QUERY_LATENCY = DummyMetric()
    TOKEN_USAGE = DummyMetric()
    QUERY_ERRORS = DummyMetric()


class StructuredJSONFormatter(logging.Formatter):
    """Formats log records as JSON objects for structured parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "file": record.filename,
            "line": record.lineno,
        }
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        return json.dumps(log_data)


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Setup structured JSON formatting on the root logger handler."""
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJSONFormatter())
    
    # Remove existing handlers to avoid duplicates
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
