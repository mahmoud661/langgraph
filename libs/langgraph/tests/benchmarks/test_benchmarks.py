"""pytest-codspeed benchmark suite.

Mirrors the scenarios in `bench/__main__.py` (the legacy pyperf runner) so the
two systems can run side by side. Run with::

    pytest tests/benchmarks --codspeed

In CI these are executed by the CodSpeed GitHub Action (walltime mode on a
dedicated runner); locally `--codspeed` falls back to walltime on the current
machine. Outside of `--codspeed` they are skipped (see `conftest.py`) so they
do not slow down the normal test suite.

Async graphs are driven through a single reused event loop (uvloop when
available, otherwise the stdlib loop) so loop construction is not counted in
the measurement -- matching the legacy runner's `loop_factory` behaviour.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from typing import Any, cast
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from pytest_codspeed import BenchmarkFixture

from bench.cases import (
    CompilationBenchmark,
    FunctionBenchmark,
    GraphBenchmark,
    compilation_benchmarks,
    function_benchmarks,
    graph_benchmarks,
)
from langgraph.pregel import Pregel


def _new_event_loop() -> asyncio.AbstractEventLoop:
    """A fresh event loop, preferring uvloop when it is installed."""
    try:
        from uvloop import new_event_loop
    except ImportError:
        return asyncio.new_event_loop()
    return new_event_loop()


def _config() -> RunnableConfig:
    """A fresh run config with a unique thread id (mirrors the legacy runner)."""
    return {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": 1000000000,
    }


async def _arun(graph: Pregel, input: dict) -> None:
    """Consume the full async stream."""
    len([c async for c in graph.astream(input, _config(), durability="exit")])


def _run(graph: Pregel, input: dict) -> None:
    """Consume the full sync stream."""
    len([c for c in graph.stream(input, _config(), durability="exit")])


async def _arun_first_event(graph: Pregel, input: dict) -> None:
    """Run until the first streamed event, then stop."""
    # astream returns an async generator at runtime; cast so aclose() type-checks.
    stream = cast(
        "AsyncGenerator[Any, None]",
        graph.astream(input, _config(), durability="exit"),
    )
    try:
        async for _ in stream:
            break
    finally:
        await stream.aclose()


def _run_first_event(graph: Pregel, input: dict) -> None:
    """Run until the first streamed event, then stop."""
    # stream returns a generator at runtime; cast so close() type-checks.
    stream = cast(
        "Generator[Any, None, None]",
        graph.stream(input, _config(), durability="exit"),
    )
    try:
        for _ in stream:
            break
    finally:
        stream.close()


def _bench_async(
    benchmark: BenchmarkFixture,
    coro_factory: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    """Benchmark an async callable on a reused event loop."""
    loop = _new_event_loop()
    try:
        benchmark(lambda: loop.run_until_complete(coro_factory()))
    finally:
        loop.close()


_GRAPH_BENCHMARKS = graph_benchmarks()


@pytest.mark.parametrize(
    "case", _GRAPH_BENCHMARKS, ids=[c.id for c in _GRAPH_BENCHMARKS]
)
def test_graph_async(benchmark: BenchmarkFixture, case: GraphBenchmark) -> None:
    graph = case.make_async()
    input = case.make_input()
    _bench_async(benchmark, lambda: _arun(graph, input))


@pytest.mark.parametrize(
    "case",
    [c for c in _GRAPH_BENCHMARKS if c.make_sync is not None],
    ids=[c.id for c in _GRAPH_BENCHMARKS if c.make_sync is not None],
)
def test_graph_sync(benchmark: BenchmarkFixture, case: GraphBenchmark) -> None:
    assert case.make_sync is not None
    graph = case.make_sync()
    input = case.make_input()
    benchmark(_run, graph, input)


_FIRST_EVENT_BENCHMARKS = [c for c in _GRAPH_BENCHMARKS if c.first_event_latency]


@pytest.mark.parametrize(
    "case", _FIRST_EVENT_BENCHMARKS, ids=[c.id for c in _FIRST_EVENT_BENCHMARKS]
)
def test_graph_first_event_async(
    benchmark: BenchmarkFixture, case: GraphBenchmark
) -> None:
    graph = case.make_async()
    input = case.make_input()
    _bench_async(benchmark, lambda: _arun_first_event(graph, input))


@pytest.mark.parametrize(
    "case",
    [c for c in _FIRST_EVENT_BENCHMARKS if c.make_sync is not None],
    ids=[c.id for c in _FIRST_EVENT_BENCHMARKS if c.make_sync is not None],
)
def test_graph_first_event_sync(
    benchmark: BenchmarkFixture, case: GraphBenchmark
) -> None:
    assert case.make_sync is not None
    graph = case.make_sync()
    input = case.make_input()
    benchmark(_run_first_event, graph, input)


_COMPILATION_BENCHMARKS = compilation_benchmarks()


@pytest.mark.parametrize(
    "case", _COMPILATION_BENCHMARKS, ids=[c.id for c in _COMPILATION_BENCHMARKS]
)
def test_graph_compilation(
    benchmark: BenchmarkFixture, case: CompilationBenchmark
) -> None:
    builder = case.make_builder()
    benchmark(builder.compile)


_FUNCTION_BENCHMARKS = function_benchmarks()


@pytest.mark.parametrize(
    "case", _FUNCTION_BENCHMARKS, ids=[c.id for c in _FUNCTION_BENCHMARKS]
)
def test_function(benchmark: BenchmarkFixture, case: FunctionBenchmark) -> None:
    benchmark(case.func)
