"""Edge/cloud per-step router for the Hermes agent loop (Mint hybrid mode).

Pure decision logic plus light stateful helpers; no model calls, no Rust deps.
Loop wiring lives in conversation_loop.py. See
docs/Mint-Hermes-in-loop端云逐步路由方案.md.
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass

from agent.tool_guardrails import IDEMPOTENT_TOOL_NAMES, MUTATING_TOOL_NAMES


class Reason:
    """Routing / escalation reason codes (stats contract, spec §5.7)."""

    SESSION_TOOL_INCAPABLE = "session_tool_incapable"
    MULTIMODAL = "multimodal"
    CONTEXT_OVERFLOW = "context_overflow"
    UNKNOWN_CONTEXT = "unknown_context"
    ENDPOINT_DOWN = "endpoint_down"
    LOCAL_API_ERROR = "local_api_error"
    DANGEROUS_TOOL = "dangerous_tool"
    PARSE_FAIL = "parse_fail"
    EMPTY_OR_REFUSAL = "empty_or_refusal"
    NO_PROGRESS = "no_progress"
    LOCAL_TIMEOUT = "local_timeout"
    PRIOR_TOOL_ERROR = "prior_tool_error"
    FINAL_ANSWER_KNOB = "final_answer_knob"
    FIRST_STEP_KNOB = "first_step_knob"
    REDO_CLOUD = "redo_cloud"


ACCEPT_LOCAL = "accept_local"
DEFAULT_LOCAL = "default_local"


@dataclass(frozen=True)
class RouteDecision:
    """backend in {'local', 'cloud'}; reason is a Reason code or sentinel."""

    backend: str
    reason: str


_PROCESS_READ_ACTIONS = frozenset({"list", "poll", "log", "wait"})
_KANBAN_LOCAL_TOOLS = frozenset({"kanban_show", "kanban_list"})


def _parse_args(raw) -> dict:
    """Normalized tool_calls carry .arguments as a JSON string or dict."""

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def classify_tool(name: str, args: dict | None) -> str:
    """Return 'local' if one tool call is safe for the edge model."""

    args = args or {}
    if name == "process":
        action = str(args.get("action", "")).strip().lower()
        return "local" if action in _PROCESS_READ_ACTIONS else "cloud"
    if name in _KANBAN_LOCAL_TOOLS:
        return "local"
    if name == "browser_console":
        has_expression = bool(str(args.get("expression", "")).strip())
        clears = bool(args.get("clear"))
        return "cloud" if has_expression or clears else "local"
    if name in MUTATING_TOOL_NAMES:
        return "cloud"
    if name in IDEMPOTENT_TOOL_NAMES:
        return "local"
    return "cloud"


def route_tool_calls(tool_calls) -> RouteDecision:
    """Accept local only if every proposed tool call is local-tier."""

    for tool_call in tool_calls:
        name = tool_call.function.name
        if classify_tool(name, _parse_args(tool_call.function.arguments)) == "cloud":
            return RouteDecision("cloud", Reason.DANGEROUS_TOOL)
    return RouteDecision("local", ACCEPT_LOCAL)


_IMAGE_PART_TYPES = frozenset({"image_url", "image", "input_image"})


def messages_have_image(api_messages) -> bool:
    """True if any message carries multimodal image content parts."""

    for message in api_messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES:
                    return True
    return False


def context_overflow(estimated_input, reserved_output, local_ctx, *, safety=0.85) -> bool:
    """True means cloud is required; unknown local context is treated as overflow."""

    if not local_ctx or local_ctx <= 0:
        return True
    return (estimated_input + reserved_output) > (local_ctx * safety)


def accept_local_text(
    user_text,
    output_text,
    *,
    tools_available,
    final_answer_knob="local",
) -> RouteDecision:
    """Accept local no-tool-call text answers unless explicitly disabled."""

    if final_answer_knob == "cloud":
        return RouteDecision("cloud", Reason.FINAL_ANSWER_KNOB)
    return RouteDecision("local", ACCEPT_LOCAL)


_TRUE_STRINGS = frozenset({"1", "true", "yes", "on", "enabled"})


@dataclass(frozen=True)
class LocalFingerprint:
    base_url: str
    model: str
    tool_capable: bool
    vision_capable: bool
    context: int

    @classmethod
    def from_env(cls, env) -> "LocalFingerprint":
        def _bool(key: str) -> bool:
            return str(env.get(key, "")).strip().lower() in _TRUE_STRINGS

        def _int(key: str) -> int:
            try:
                return int(str(env.get(key, "0")).strip() or "0")
            except (TypeError, ValueError):
                return 0

        return cls(
            base_url=env.get("MINT_LOCAL_LLM_URL", "") or "",
            model=env.get("MINT_LOCAL_LLM_MODEL", "") or "",
            tool_capable=_bool("MINT_LOCAL_LLM_TOOL_CAPABLE"),
            vision_capable=_bool("MINT_LOCAL_LLM_VISION_CAPABLE"),
            context=_int("MINT_LOCAL_LLM_CONTEXT"),
        )


class HealthGate:
    """TTL-cached health plus failure cooldown.

    State only: the actual endpoint probe is done by the loop, which calls
    mark_ok() or mark_failure().
    """

    def __init__(self, clock, *, ok_ttl: float = 20.0, cooldown: float = 30.0):
        self._clock = clock
        self._ok_ttl = ok_ttl
        self._cooldown = cooldown
        self._last_ok_at = None
        self._cooldown_until = None

    def in_cooldown(self) -> bool:
        return self._cooldown_until is not None and self._clock() < self._cooldown_until

    def fresh_ok(self) -> bool:
        return self._last_ok_at is not None and (self._clock() - self._last_ok_at) < self._ok_ttl

    def needs_probe(self) -> bool:
        """True means probe /health; False means cooldown or fresh cached ok."""

        return not self.in_cooldown() and not self.fresh_ok()

    def mark_ok(self) -> None:
        self._last_ok_at = self._clock()
        self._cooldown_until = None

    def mark_failure(self) -> None:
        self._cooldown_until = self._clock() + self._cooldown
        self._last_ok_at = None


@dataclass
class RuntimeProfile:
    model: str
    provider: str
    base_url: str
    api_key: str
    api_mode: str
    client: object = None
    anthropic_client: object = None
    client_kwargs: dict | None = None
    anthropic_api_key: str = ""
    anthropic_base_url: str | None = None
    is_anthropic_oauth: bool = False


_RUNTIME_SNAPSHOT_FIELDS = (
    "model",
    "provider",
    "base_url",
    "api_key",
    "api_mode",
    "client",
    "_anthropic_client",
    "_client_kwargs",
    "_anthropic_api_key",
    "_anthropic_base_url",
    "_is_anthropic_oauth",
    "_config_context_length",
    "_use_prompt_caching",
    "_use_native_cache_layout",
)
_MISSING = object()


def apply_runtime_profile(agent, profile: RuntimeProfile) -> None:
    """Swap per-request runtime fields onto a live agent with rollback."""

    snapshot = {field: getattr(agent, field, _MISSING) for field in _RUNTIME_SNAPSHOT_FIELDS}
    try:
        agent.model = profile.model
        agent.provider = profile.provider
        agent.base_url = profile.base_url
        agent.api_key = profile.api_key
        agent.api_mode = profile.api_mode
        agent._config_context_length = None
        agent._client_kwargs = dict(profile.client_kwargs or {})
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        if profile.api_mode == "anthropic_messages":
            agent._anthropic_client = profile.anthropic_client
            agent._anthropic_api_key = profile.anthropic_api_key
            agent._anthropic_base_url = profile.anthropic_base_url
            agent._is_anthropic_oauth = profile.is_anthropic_oauth
            agent.client = None
        else:
            agent.client = profile.client

        policy = getattr(agent, "_anthropic_prompt_cache_policy", None)
        if callable(policy):
            agent._use_prompt_caching, agent._use_native_cache_layout = policy(
                provider=profile.provider,
                base_url=profile.base_url,
                api_mode=profile.api_mode,
                model=profile.model,
            )
    except Exception:
        for field, value in snapshot.items():
            if value is not _MISSING:
                setattr(agent, field, value)
        raise


class HybridRouter:
    def __init__(
        self,
        local_profile: RuntimeProfile,
        cloud_profile: RuntimeProfile,
        *,
        fingerprint: LocalFingerprint | None = None,
        knobs: dict | None = None,
        health: HealthGate | None = None,
    ):
        self.local = local_profile
        self.cloud = cloud_profile
        self.fingerprint = fingerprint
        self.knobs = knobs or {}
        self.health = health

    def decide_pre(
        self,
        *,
        tools,
        api_messages,
        fingerprint: LocalFingerprint,
        is_first_step: bool = False,
        force_cloud: bool = False,
    ) -> RouteDecision:
        if force_cloud:
            return RouteDecision("cloud", Reason.REDO_CLOUD)
        if tools and not fingerprint.tool_capable:
            return RouteDecision("cloud", Reason.SESSION_TOOL_INCAPABLE)
        if messages_have_image(api_messages) and not fingerprint.vision_capable:
            return RouteDecision("cloud", Reason.MULTIMODAL)
        if self.health is not None and self.health.in_cooldown():
            return RouteDecision("cloud", Reason.ENDPOINT_DOWN)
        if is_first_step and self.knobs.get("first_step") == "cloud":
            return RouteDecision("cloud", Reason.FIRST_STEP_KNOB)
        return RouteDecision("local", DEFAULT_LOCAL)

    def decide_post(
        self,
        *,
        assistant_message,
        user_text,
        tools,
        last_tool_failed: bool = False,
    ) -> RouteDecision:
        tool_calls = getattr(assistant_message, "tool_calls", None)
        if tool_calls:
            return route_tool_calls(tool_calls)
        if last_tool_failed:
            return RouteDecision("cloud", Reason.PRIOR_TOOL_ERROR)
        content = getattr(assistant_message, "content", "") or ""
        return accept_local_text(
            user_text,
            content,
            tools_available=bool(tools),
            final_answer_knob=self.knobs.get("final_answer", "local"),
        )


def maybe_build_hybrid_router(agent, env, *, clock=None, knobs=None):
    """Build a HybridRouter from injected env, or None when not hybrid."""

    fingerprint = LocalFingerprint.from_env(env)
    if not fingerprint.base_url or not fingerprint.model:
        return None

    cloud = RuntimeProfile(
        model=getattr(agent, "model", ""),
        provider=getattr(agent, "provider", ""),
        base_url=getattr(agent, "base_url", ""),
        api_key=getattr(agent, "api_key", ""),
        api_mode=getattr(agent, "api_mode", "chat_completions"),
        client=getattr(agent, "client", None),
        anthropic_client=getattr(agent, "_anthropic_client", None),
        client_kwargs=dict(getattr(agent, "_client_kwargs", {}) or {}),
        anthropic_api_key=getattr(agent, "_anthropic_api_key", ""),
        anthropic_base_url=getattr(agent, "_anthropic_base_url", None),
        is_anthropic_oauth=getattr(agent, "_is_anthropic_oauth", False),
    )

    local_kwargs = {"api_key": "no-key-required", "base_url": fingerprint.base_url}
    local_client = agent._create_openai_client(
        local_kwargs,
        reason="hybrid_local",
        shared=True,
    )
    local = RuntimeProfile(
        model=fingerprint.model,
        provider="mint-local",
        base_url=fingerprint.base_url,
        api_key="no-key-required",
        api_mode="chat_completions",
        client=local_client,
        anthropic_client=None,
        client_kwargs=dict(local_kwargs),
    )
    return HybridRouter(
        local,
        cloud,
        fingerprint=fingerprint,
        knobs=knobs or {},
        health=HealthGate(clock or _time.monotonic),
    )
