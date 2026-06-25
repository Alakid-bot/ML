"""Shared training schema constants."""

from __future__ import annotations

FEATURE_COLUMNS = [
    "traffic_input_mbps",
    "cpu_cores",
    "ram_gb",
    "link_capacity_mbps",
    "cpu_utilization_percent",
    "memory_utilization_percent",
    "latency_ms",
    "throughput_mbps",
    "packet_loss_percent",
]

TARGET_COLUMN = "max_supported_load_mbps"
DEFAULT_MODELS = ["dummy_mean", "ridge", "mlp", "adaptive_hybrid"]
MIN_ROWS = 20

__all__ = ["DEFAULT_MODELS", "FEATURE_COLUMNS", "MIN_ROWS", "TARGET_COLUMN"]
