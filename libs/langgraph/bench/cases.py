"""Shared benchmark scenario definitions.

These mirror the scenarios in `bench/__main__.py` (the legacy pyperf runner)
but expose them as lazy factories so the pytest-codspeed suite
(`tests/benchmarks/`) can build each graph on demand instead of compiling
dozens of graphs at import time.

Graphs and inputs are returned by callables: importing this module is cheap,
and each benchmark constructs a fresh, isolated graph + payload when it runs.
Factories are built with `functools.partial` over small named helpers so the
checkpointer (and every other resource) is created fresh on each call rather
than shared across runs.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from bench.fanout_to_subgraph import fanout_to_subgraph, fanout_to_subgraph_sync
from bench.pydantic_state import pydantic_state
from bench.react_agent import react_agent
from bench.sequential import create_sequential
from bench.serde_allowlist import collect_allowlist_large, collect_allowlist_small
from bench.wide_dict import wide_dict
from bench.wide_state import wide_state
from langgraph.graph.state import StateGraph
from langgraph.pregel import Pregel


@dataclass(frozen=True)
class GraphBenchmark:
    """A graph-execution benchmark.

    `make_async` / `make_sync` each build a freshly compiled graph and
    `make_input` builds a fresh input payload. Keeping these lazy means
    importing this module does not compile any graphs.
    """

    id: str
    make_input: Callable[[], dict]
    make_async: Callable[[], Pregel]
    make_sync: Callable[[], Pregel] | None
    # Whether to also measure first-event (time-to-first-token) latency.
    first_event_latency: bool = False


@dataclass(frozen=True)
class CompilationBenchmark:
    """A graph-compilation benchmark: build the (uncompiled) graph, time
    `.compile()`."""

    id: str
    make_builder: Callable[[], StateGraph]


@dataclass(frozen=True)
class FunctionBenchmark:
    """A plain synchronous function benchmark (e.g. serde allowlist)."""

    id: str
    func: Callable[[], object]


# --- input payload builders -------------------------------------------------


def _subjects_input(n: int) -> dict:
    return {
        "subjects": [
            random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(n)
        ]
    }


def _wide_input(outer: int, inner: int) -> dict:
    return {
        "messages": [
            {
                str(i) * 10: {
                    str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                    for j in range(inner)
                }
                for i in range(outer)
            }
        ]
    }


def _react_input() -> dict:
    return {"messages": [HumanMessage("hi?")]}


def _empty_messages_input() -> dict:
    return {"messages": []}


# --- compiled-graph builders (fresh checkpointer per call) ------------------


def _checkpointer(checkpoint: bool) -> BaseCheckpointSaver | None:
    return InMemorySaver() if checkpoint else None


def _build_fanout(sync: bool, checkpoint: bool) -> Pregel:
    builder = fanout_to_subgraph_sync if sync else fanout_to_subgraph
    return builder().compile(checkpointer=_checkpointer(checkpoint))


def _build_react(n: int, checkpoint: bool) -> Pregel:
    return react_agent(n, checkpointer=_checkpointer(checkpoint))


def _build_sized(
    builder: Callable[[int], StateGraph], size: int, checkpoint: bool
) -> Pregel:
    return builder(size).compile(checkpointer=_checkpointer(checkpoint))


def _build_sequential(n: int) -> Pregel:
    return create_sequential(n).compile()


# (label, size arg passed to the builder, outer range, inner range)
_WIDE_SHAPES: tuple[tuple[str, int, int, int], ...] = (
    ("25x300", 300, 5, 5),
    ("15x600", 600, 3, 5),
    ("9x1200", 1200, 3, 3),
)

# Scenarios that additionally get a first-event-latency benchmark. Mirrors the
# intent of `GRAPHS_FOR_1st_EVENT_LATENCY` in `bench/__main__.py`.
_FIRST_EVENT_LATENCY = frozenset({"sequential_1000", "pydantic_state_25x300"})


def graph_benchmarks() -> list[GraphBenchmark]:
    """All full-graph-run benchmarks (async + sync), with and without a
    checkpointer."""
    out: list[GraphBenchmark] = []

    # fanout_to_subgraph: async and sync use distinct builders.
    for n in (10, 100):
        out.append(
            GraphBenchmark(
                id=f"fanout_to_subgraph_{n}x",
                make_input=partial(_subjects_input, n),
                make_async=partial(_build_fanout, False, False),
                make_sync=partial(_build_fanout, True, False),
            )
        )
        out.append(
            GraphBenchmark(
                id=f"fanout_to_subgraph_{n}x_checkpoint",
                make_input=partial(_subjects_input, n),
                make_async=partial(_build_fanout, False, True),
                make_sync=partial(_build_fanout, True, True),
            )
        )

    # react_agent
    for n in (10, 100):
        out.append(
            GraphBenchmark(
                id=f"react_agent_{n}x",
                make_input=_react_input,
                make_async=partial(_build_react, n, False),
                make_sync=partial(_build_react, n, False),
            )
        )
        out.append(
            GraphBenchmark(
                id=f"react_agent_{n}x_checkpoint",
                make_input=_react_input,
                make_async=partial(_build_react, n, True),
                make_sync=partial(_build_react, n, True),
            )
        )

    # wide_state / wide_dict / pydantic_state share the same input shapes.
    for builder, prefix in (
        (wide_state, "wide_state"),
        (wide_dict, "wide_dict"),
        (pydantic_state, "pydantic_state"),
    ):
        for label, size, outer, inner in _WIDE_SHAPES:
            base_id = f"{prefix}_{label}"
            out.append(
                GraphBenchmark(
                    id=base_id,
                    make_input=partial(_wide_input, outer, inner),
                    make_async=partial(_build_sized, builder, size, False),
                    make_sync=partial(_build_sized, builder, size, False),
                    first_event_latency=base_id in _FIRST_EVENT_LATENCY,
                )
            )
            out.append(
                GraphBenchmark(
                    id=f"{base_id}_checkpoint",
                    make_input=partial(_wide_input, outer, inner),
                    make_async=partial(_build_sized, builder, size, True),
                    make_sync=partial(_build_sized, builder, size, True),
                )
            )

    # sequential
    for n in (10, 1000):
        base_id = f"sequential_{n}"
        out.append(
            GraphBenchmark(
                id=base_id,
                make_input=_empty_messages_input,
                make_async=partial(_build_sequential, n),
                make_sync=partial(_build_sequential, n),
                first_event_latency=base_id in _FIRST_EVENT_LATENCY,
            )
        )

    return out


def compilation_benchmarks() -> list[CompilationBenchmark]:
    """Graph-compilation benchmarks."""
    return [
        CompilationBenchmark(
            id="sequential_1000", make_builder=partial(create_sequential, 1_000)
        ),
        CompilationBenchmark(
            id="pydantic_state_25x300", make_builder=partial(pydantic_state, 300)
        ),
        CompilationBenchmark(
            id="wide_state_15x600", make_builder=partial(wide_state, 600)
        ),
    ]


def function_benchmarks() -> list[FunctionBenchmark]:
    """Standalone function benchmarks."""
    return [
        FunctionBenchmark(id="serde_allowlist_small", func=collect_allowlist_small),
        FunctionBenchmark(id="serde_allowlist_large", func=collect_allowlist_large),
    ]
