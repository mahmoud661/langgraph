"""pytest configuration for the benchmark suite.

The benchmarks under `tests/benchmarks/` are only meaningful under CodSpeed
(or its local walltime fallback). Outside of `--codspeed` they would run as
ordinary tests and noticeably slow down `make test`, so we skip them. This
hook is scoped to items collected from this directory; it leaves the rest of
the suite alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).parent


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--codspeed", default=False):
        # CodSpeed is active: it selects the benchmark tests itself.
        return
    skip = pytest.mark.skip(
        reason="benchmark; run with `pytest tests/benchmarks --codspeed`"
    )
    for item in items:
        item_path = Path(str(getattr(item, "fspath", "")))
        if _BENCH_DIR == item_path.parent or _BENCH_DIR in item_path.parents:
            item.add_marker(skip)
