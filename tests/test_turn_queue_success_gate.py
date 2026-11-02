"""The queued-message success gate.

Queued messages run only when the previous turn is judged a success: a
completed turn releases the queue through the turn-success judge, while a
failed or interrupted turn pauses it directly.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from mistralai_vibe_local_harness.protocol import RustTextContentBlock
from mistralai_vibe_local_harness.session_protocol import (
    TextContentBlock,
    TurnEnqueueParams,
    TurnUserInputEntry,
)
from mistralai_vibe_local_harness.vibe._runtime_config import LocalRuntimeAdapterConfig
from mistralai_vibe_local_harness.vibe._session import UnifiedHarnessSessionBackend
from mistralai_vibe_local_harness.vibe._transcript import build_transcript
from mistralai_vibe_local_harness.vibe._turn_queue import PreparedTurnEntry
from mistralai_vibe_local_harness.vibe._turn_success import JevTurnSuccessJudge

_DEFAULT_JUDGE_ENTRIES = [
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
    }
]


def _session(
    judge: Any = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    *,
    gate: bool | str = "jev",
) -> UnifiedHarnessSessionBackend:
    session = UnifiedHarnessSessionBackend(
        session_id="s-1",
        created_at=1,
        adapter_config=LocalRuntimeAdapterConfig(queue_success_gate=gate),
        turn_success_judge=judge,
    )
    if monkeypatch is not None:
        # The drain promotes a Runtime when none exists; hand back a stand-in.
        async def _fake_runtime(**kwargs: Any) -> Any:
            return SimpleNamespace()

        monkeypatch.setattr(session, "_runtime_for_turn_locked", _fake_runtime)
    return session


def _enqueue(session: UnifiedHarnessSessionBackend, text: str = "next") -> None:
    params = TurnEnqueueParams(
        session_id="s-1",
        entries=[TurnUserInputEntry(content=[TextContentBlock(text=text)])],
    )
    session._turn_queue.enqueue(
        params,
        (PreparedTurnEntry(role="user", content=(RustTextContentBlock(text=text),)),),
        created_at=1,
    )


async def _settle(session: UnifiedHarnessSessionBackend) -> None:
    gate = session._queue_gate_task
    if gate is not None:
        await gate
    drain = session._queue_drain_task
    if drain is not None:
        await drain


@pytest.mark.asyncio
async def test_judged_success_runs_the_next_queued_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[str] = []
    session = _session(
        judge=lambda entries: _success(entries, True), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        session,
        "_start_queued_turn",
        lambda runtime, record: started.append(record.queued_turn.id),
    )
    _enqueue(session)

    session._handle_terminal_turn("turn-1", "completed")
    await _settle(session)

    assert started, "the queued message should have started"
    assert session._turn_queue.state.paused is False
    assert not session._turn_queue


@pytest.mark.asyncio
async def test_judged_failure_pauses_the_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []
    session = _session(
        judge=lambda entries: _success(entries, False), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        session,
        "_start_queued_turn",
        lambda runtime, record: started.append(record.queued_turn.id),
    )
    _enqueue(session)

    session._handle_terminal_turn("turn-1", "completed")
    await _settle(session)

    assert not started
    assert session._turn_queue.state.paused is True
    assert session._turn_queue  # the message stays queued


@pytest.mark.asyncio
async def test_failed_turn_pauses_without_calling_the_judge() -> None:
    calls: list[Any] = []

    async def judge(entries: Any) -> bool:
        calls.append(entries)
        return True

    session = _session(judge=judge)
    _enqueue(session)

    session._handle_terminal_turn("turn-1", "failed")
    await _settle(session)

    assert not calls
    assert session._turn_queue.state.paused is True
    assert session._queue_gate_task is None


@pytest.mark.asyncio
async def test_completed_turn_with_empty_queue_skips_the_judge() -> None:
    calls: list[Any] = []

    async def judge(entries: Any) -> bool:
        calls.append(entries)
        return True

    session = _session(judge=judge)

    session._handle_terminal_turn("turn-1", "completed")
    await _settle(session)

    assert not calls
    assert session._queue_gate_task is None


@pytest.mark.asyncio
async def test_drain_cannot_slip_past_pending_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()
    started: list[str] = []

    async def judge(entries: Any) -> bool:
        await release.wait()
        return True

    session = _session(judge=judge, monkeypatch=monkeypatch)
    monkeypatch.setattr(
        session,
        "_start_queued_turn",
        lambda runtime, record: started.append(record.queued_turn.id),
    )
    _enqueue(session)

    session._handle_terminal_turn("turn-1", "completed")
    assert session._queue_gate_task is not None
    session._schedule_turn_queue_drain()
    assert session._queue_drain_task is None

    release.set()
    await _settle(session)

    assert started
    assert session._turn_queue.state.paused is False


def test_the_gate_is_disabled_by_default() -> None:
    assert LocalRuntimeAdapterConfig().queue_success_gate is False


@pytest.mark.asyncio
async def test_the_gate_is_disabled_by_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    started: list[str] = []

    async def judge(entries: Any) -> bool:
        calls.append(entries)
        return False

    async def _fake_runtime(**kwargs: Any) -> Any:
        return SimpleNamespace()

    # The default config (gate off), as a session without an explicit opt-in.
    session = UnifiedHarnessSessionBackend(
        session_id="s-1",
        created_at=1,
        adapter_config=LocalRuntimeAdapterConfig(),
        turn_success_judge=judge,
    )
    monkeypatch.setattr(session, "_runtime_for_turn_locked", _fake_runtime)
    monkeypatch.setattr(
        session,
        "_start_queued_turn",
        lambda runtime, record: started.append(record.queued_turn.id),
    )
    _enqueue(session)

    session._handle_terminal_turn("turn-1", "completed")
    await _settle(session)

    assert not calls, "the judge must not run when the gate is disabled"
    assert session._queue_gate_task is None
    assert started, "a completed turn drains the queue directly when disabled"
    assert session._turn_queue.state.paused is False


async def _success(entries: Any, verdict: bool) -> bool:
    return verdict


def test_the_jev_judge_asks_one_typed_noul_question() -> None:
    request = JevTurnSuccessJudge._build_request("user: fix it\n\nassistant: done")
    assert request["model"] == "jev-latest"
    assert request["state"] == "user: fix it\n\nassistant: done"
    questions = request["questions"]
    assert isinstance(questions, dict)
    question = questions["turn_succeeded"]
    assert isinstance(question, dict)
    assert question["type"] == "noul"
    criteria = question["criteria"]
    assert isinstance(criteria, dict)
    assert set(criteria) == {"true", "false"}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"answers": {"turn_succeeded": {"type": "noul", "noul": 0.95}}}, True),
        ({"answers": {"turn_succeeded": {"type": "noul", "noul": 0.5}}}, True),
        ({"answers": {"turn_succeeded": {"type": "noul", "noul": 0.2}}}, False),
        ({"answers": {"turn_succeeded": {"type": "noul", "noul": "yes"}}}, False),
        ({"answers": {}}, False),
        ({}, False),
        ("not a dict", False),
    ],
)
def test_the_jev_judge_reads_the_noul_probability(
    response: Any, expected: bool
) -> None:
    assert JevTurnSuccessJudge._parse_success(response) is expected


@pytest.mark.asyncio
async def test_the_jev_judge_fails_closed_and_bypasses_a_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    seen: list[tuple[Any, str]] = []

    async def fetch(request: Any, api_key: str) -> Any:
        seen.append((request, api_key))
        return {"answers": {"turn_succeeded": {"type": "noul", "noul": 0.9}}}

    config = LocalRuntimeAdapterConfig()
    assert (
        await JevTurnSuccessJudge(config, fetch=fetch)(_DEFAULT_JUDGE_ENTRIES) is True
    )
    assert seen[0][1] == "test-key"
    assert seen[0][0]["model"] == "jev-latest"

    async def failing_fetch(request: Any, api_key: str) -> Any:
        raise RuntimeError("boom")

    assert (
        await JevTurnSuccessJudge(config, fetch=failing_fetch)(_DEFAULT_JUDGE_ENTRIES)
        is False
    )

    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert (
        await JevTurnSuccessJudge(config, fetch=fetch)(_DEFAULT_JUDGE_ENTRIES) is True
    )
    assert len(seen) == 1, "a missing key must not reach the network"


def test_build_transcript_covers_messages_and_tool_effects() -> None:
    entries = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "text", "text": "fix it"}],
        },
        {
            "type": "effect",
            "detail": {"toolName": "file_system.edit", "input": {"path": "a.py"}},
            "state": {"status": "failed", "error": {"message": "boom"}},
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
        },
    ]
    transcript = build_transcript(entries)
    assert "user: fix it" in transcript
    assert "tool(file_system.edit)" in transcript
    assert "tool result(file_system.edit) failed: boom" in transcript
    assert "assistant: done" in transcript
