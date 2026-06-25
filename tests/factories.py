"""Shared test factories for the ML project test suite.

These helpers exist so every test module builds profiling datasets the same
way. Keeping the row generator in one place avoids subtle drift between the
``make_dataset`` copies that previously lived in each test file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def make_dataset(row_count: int = 30, *, include_metadata: bool = False) -> pd.DataFrame:
    """Build a synthetic profiling dataset with deterministic rows.

    The DataFrame always contains the nine feature columns and the
    ``max_supported_load_mbps`` target column used by the training pipeline.

    Args:
        row_count: Number of rows to generate.
        include_metadata: When True, prepend ``sample_id`` and ``service_name``
            columns so the DataFrame mirrors the real profiling CSV layout
            that ships with the project template. The extra columns are not
            required by the schema and are ignored by validation and training.

    Returns:
        A DataFrame with feature and target columns, plus optional leading
        metadata columns when requested.
    """
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        traffic = 100 + index * 10
        cpu_cores = 1 + index % 4
        ram_gb = 2 + (index % 3) * 2
        link_capacity = 1000
        cpu_utilization = min(95.0, 35 + traffic / 10 - cpu_cores * 4)
        memory_utilization = min(90.0, 30 + ram_gb * 3 + index % 5)
        latency = 10 + traffic / 50 + max(0, cpu_utilization - 70) * 0.8
        throughput = traffic * (1 - min(0.1, index / 1000))
        packet_loss = max(0.0, (latency - 35) / 100)
        max_supported_load = traffic + cpu_cores * 45 + ram_gb * 8 - latency * 1.5

        row: dict[str, object] = {
            "traffic_input_mbps": traffic,
            "cpu_cores": cpu_cores,
            "ram_gb": ram_gb,
            "link_capacity_mbps": link_capacity,
            "cpu_utilization_percent": cpu_utilization,
            "memory_utilization_percent": memory_utilization,
            "latency_ms": latency,
            "throughput_mbps": throughput,
            "packet_loss_percent": packet_loss,
            "max_supported_load_mbps": max_supported_load,
        }
        if include_metadata:
            row = {
                "sample_id": index + 1,
                "service_name": "snort",
                **row,
            }
        rows.append(row)

    return pd.DataFrame(rows)


def write_dataset(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to CSV with minimal quoting, matching train CSVs."""
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


__all__ = ["make_dataset", "write_dataset"]
