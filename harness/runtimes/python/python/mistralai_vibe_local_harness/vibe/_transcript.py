"""Render a session's history entries as a compact transcript.

Shared by the utility calls that read the transcript: session-title
generation and the queued-turn success judge. Limits keep the transcript
small: a long transcript keeps its first and last stretch with an elision
in between.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json

from mistralai_vibe_local_harness.session_protocol import JsonObject

_ELISION = "\n\n[…]\n\n"


@dataclasses.dataclass(frozen=True, slots=True)
class TranscriptPolicy:
    max_transcript_chars: int = 6000
    head_transcript_chars: int = 1500
    max_message_chars: int = 2000
    max_tool_chars: int = 400

    @property
    def tail_transcript_chars(self) -> int:
        return self.max_transcript_chars - self.head_transcript_chars


DEFAULT_TRANSCRIPT_POLICY = TranscriptPolicy()


def build_transcript(
    entries: Sequence[JsonObject],
    *,
    policy: TranscriptPolicy = DEFAULT_TRANSCRIPT_POLICY,
) -> str:
    blocks: list[str] = []
    for entry in entries:
        entry_type = entry.get("type")
        if entry_type == "message":
            block = _message_block(entry, policy=policy)
        elif entry_type == "effect":
            block = _tool_block(entry, policy=policy)
        else:
            continue
        if block:
            blocks.append(block)
    transcript = "\n\n".join(blocks).strip()
    if len(transcript) <= policy.max_transcript_chars:
        return transcript
    head = transcript[: policy.head_transcript_chars].rstrip()
    tail = transcript[-policy.tail_transcript_chars :].lstrip()
    return f"{head}{_ELISION}{tail}"


def _message_block(entry: JsonObject, *, policy: TranscriptPolicy) -> str | None:
    role = entry.get("role")
    if role not in {"user", "assistant"}:
        return None
    text = _entry_text(entry).strip()
    if not text:
        return None
    return f"{role}: {text[: policy.max_message_chars]}"


def _tool_block(entry: JsonObject, *, policy: TranscriptPolicy) -> str | None:
    detail = entry.get("detail")
    if not isinstance(detail, dict):
        detail = {}
    name = detail.get("toolName")
    if not isinstance(name, str) or not name:
        title = entry.get("title")
        name = title if isinstance(title, str) and title else "tool"
    lines: list[str] = []
    call = detail.get("input")
    if call is not None:
        call_text = (
            call
            if isinstance(call, str)
            else json.dumps(call, ensure_ascii=False, default=str)
        ).strip()
        if call_text and call_text not in {"{}", "[]", '""'}:
            lines.append(f"tool({name}): {call_text[: policy.max_tool_chars]}")
    state = entry.get("state")
    if isinstance(state, dict):
        if state.get("status") == "failed":
            # The failure flag outranks any partial output: a reader judging
            # or naming the turn must not miss that the call failed.
            error = state.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            if isinstance(message, str) and message.strip():
                lines.append(
                    f"tool result({name}) failed: "
                    f"{message.strip()[: policy.max_tool_chars]}"
                )
            else:
                lines.append(f"tool failed({name})")
        else:
            output = state.get("outputText")
            if isinstance(output, str) and output.strip():
                lines.append(
                    f"tool result({name}): {output.strip()[: policy.max_tool_chars]}"
                )
    return "\n".join(lines) if lines else None


def _entry_text(entry: JsonObject) -> str:
    content = entry.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            value = part.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


__all__ = ["DEFAULT_TRANSCRIPT_POLICY", "TranscriptPolicy", "build_transcript"]
