"""Judge whether a finished turn succeeded, gating the queued turns behind it.

A queued message runs only when the previous turn is judged a success. The
``TurnSuccessJudge`` protocol is the seam the Session talks to: it supplies a
stub in tests, and production wires the Jev judge below. A judge that cannot
run its call fails closed, so the queue pauses instead of running -- with one
exception: a missing API key is unconfiguration, not a verdict, so it bypasses
the gate and drains the queue as if the gate were off.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
import dataclasses
import logging
import os
from typing import Any, ClassVar, Protocol

import httpx2

from mistralai_vibe_local_harness.session_protocol import JsonObject
from mistralai_vibe_local_harness.vibe._http_transport import (
    env_proxy_transport,
    keepalive_socket_options,
)
from mistralai_vibe_local_harness.vibe._runtime_config import LocalRuntimeAdapterConfig
from mistralai_vibe_local_harness.vibe._ssl import build_ssl_context
from mistralai_vibe_local_harness.vibe._transcript import (
    TranscriptPolicy,
    build_transcript,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class TurnSuccessPolicy(TranscriptPolicy):
    """The judge's budget: transcript limits inherited, plus its call window."""

    request_timeout_seconds: float = 10.0
    total_timeout_seconds: float = 20.0


DEFAULT_TURN_SUCCESS_POLICY = TurnSuccessPolicy()


# Rules whether the finished turn behind ``entries`` succeeded. The port is
# the replacement seam: the Session supplies a stub in tests, production
# uses the Jev judge below.
class TurnSuccessJudge(Protocol):
    async def __call__(self, entries: Sequence[JsonObject]) -> bool: ...


# One call to the Jev endpoint, injectable into ``JevTurnSuccessJudge`` for
# tests. Receives the request body and the Bearer key, returns the decoded
# response body.
type JevFetch = Callable[[JsonObject, str], Awaitable[JsonObject]]


class JevTurnSuccessJudge(TurnSuccessJudge):
    """Production judge: one TypeSafe AI Jev decision call on the transcript.

    The Jev decision model (System One) answers a single typed noul (yes/no)
    question about the turn's transcript in one pass, returning a calibrated
    probability instead of generated text. Fail closed: no transcript, an
    HTTP error, a timeout, or an unreadable answer all return ``False`` so
    the queue stays put. A missing API key is different -- it is static
    misconfiguration that no turn can fix, so it bypasses the gate (``True``,
    with a warning) rather than pausing the queue forever. ``fetch`` is a
    test seam; production posts to the TypeSafe AI endpoint.
    """

    _API_BASE = "https://api.typesafe.ai"
    _API_KEY_ENV_VAR = "TYPESAFE_API_KEY"
    _MODEL = "jev-latest"
    _SYSTEMONE_PATH = "/v1/systemone"
    _QUESTION_ID = "turn_succeeded"
    _SUCCESS_THRESHOLD = 0.5
    _QUESTION_INSTRUCTIONS = (
        "Did the assistant successfully complete what the user asked in this "
        "turn? The state is the transcript of the agent's last turn."
    )
    _QUESTION_CRITERIA: ClassVar[dict[str, str]] = {
        "true": (
            "The user's request was actually fulfilled: the requested change is "
            "done, the question was answered, or the task completed with no "
            "unresolved error."
        ),
        "false": (
            "The turn failed, was cut short, left the task unfinished, ended in "
            "an error, or the assistant only described what it would do."
        ),
    }

    def __init__(
        self,
        config: LocalRuntimeAdapterConfig,
        *,
        policy: TurnSuccessPolicy = DEFAULT_TURN_SUCCESS_POLICY,
        fetch: JevFetch | None = None,
    ) -> None:
        self._config = config
        self._policy = policy
        self._fetch = fetch

    async def __call__(self, entries: Sequence[JsonObject]) -> bool:
        transcript = build_transcript(entries, policy=self._policy)
        if not transcript:
            return False
        api_key = os.environ.get(self._API_KEY_ENV_VAR, "").strip()
        if not api_key:
            # Unconfigured, not judged: pausing on a missing key would hold
            # the queue forever, so the gate steps aside and the queue drains.
            logger.warning(
                "Turn success judge has no %s; leaving the queue ungated",
                self._API_KEY_ENV_VAR,
            )
            return True
        request = self._build_request(transcript)
        try:
            async with asyncio.timeout(self._policy.total_timeout_seconds):
                response = await self._post_jev(request, api_key)
        except TimeoutError:
            logger.warning("Turn success judge timed out")
            return False
        except Exception:
            logger.warning("Turn success judge call failed", exc_info=True)
            return False
        return self._parse_success(response)

    @classmethod
    def _build_request(cls, transcript: str) -> JsonObject:
        return {
            "model": cls._MODEL,
            "state": transcript,
            "questions": {
                cls._QUESTION_ID: {
                    "type": "noul",
                    "instructions": cls._QUESTION_INSTRUCTIONS,
                    "criteria": dict(cls._QUESTION_CRITERIA),
                }
            },
        }

    @classmethod
    def _parse_success(cls, response: Any) -> bool:
        """Read the typed noul answer; anything unrecognized fails closed."""
        if not isinstance(response, dict):
            return False
        answers = response.get("answers")
        if not isinstance(answers, dict):
            return False
        answer = answers.get(cls._QUESTION_ID)
        if not isinstance(answer, dict):
            return False
        probability = answer.get("noul")
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            return False
        return float(probability) >= cls._SUCCESS_THRESHOLD

    async def _post_jev(self, request: JsonObject, api_key: str) -> JsonObject:
        if self._fetch is not None:
            return await self._fetch(request, api_key)
        url = f"{self._API_BASE}{self._SYSTEMONE_PATH}"
        transport = env_proxy_transport(
            verify=build_ssl_context(
                use_system_trust_store=self._config.enable_system_trust_store
            ),
            limits=httpx2.Limits(
                max_keepalive_connections=2, max_connections=4, keepalive_expiry=30.0
            ),
            socket_options=keepalive_socket_options(),
        )
        async with httpx2.AsyncClient(
            timeout=httpx2.Timeout(self._policy.request_timeout_seconds),
            transport=transport,
            follow_redirects=False,
        ) as client:
            result = await client.post(
                url,
                json=request,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        result.raise_for_status()
        return result.json()


__all__ = [
    "DEFAULT_TURN_SUCCESS_POLICY",
    "JevFetch",
    "JevTurnSuccessJudge",
    "TurnSuccessJudge",
    "TurnSuccessPolicy",
]
