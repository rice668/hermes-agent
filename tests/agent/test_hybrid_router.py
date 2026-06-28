"""Tests for agent/hybrid_router.py - edge/cloud per-step routing decisions."""
from __future__ import annotations

import pytest
import time as _time
from types import SimpleNamespace

from agent.hybrid_router import (
    Reason,
    RouteDecision,
    HealthGate,
    HybridRouter,
    LocalFingerprint,
    RuntimeProfile,
    accept_local_text,
    apply_runtime_profile,
    classify_tool,
    context_overflow,
    maybe_build_hybrid_router,
    messages_have_image,
    route_tool_calls,
)


class TestRouteDecision:
    def test_decision_holds_backend_and_reason(self):
        d = RouteDecision("local", "default_local")
        assert d.backend == "local"
        assert d.reason == "default_local"

    def test_reason_codes_match_spec_enum(self):
        # Spec §5.7 escalation reason enum; these strings are stats contract.
        assert Reason.SESSION_TOOL_INCAPABLE == "session_tool_incapable"
        assert Reason.MULTIMODAL == "multimodal"
        assert Reason.CONTEXT_OVERFLOW == "context_overflow"
        assert Reason.UNKNOWN_CONTEXT == "unknown_context"
        assert Reason.ENDPOINT_DOWN == "endpoint_down"
        assert Reason.LOCAL_API_ERROR == "local_api_error"
        assert Reason.DANGEROUS_TOOL == "dangerous_tool"
        assert Reason.PRIOR_TOOL_ERROR == "prior_tool_error"
        assert Reason.FINAL_ANSWER_KNOB == "final_answer_knob"
        assert Reason.FIRST_STEP_KNOB == "first_step_knob"
        assert Reason.REDO_CLOUD == "redo_cloud"


def _tc(name, arguments=""):
    """Mimic a normalized OpenAI tool_call object: tc.function.name/.arguments."""
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


class TestClassifyTool:
    def test_idempotent_tools_are_local(self):
        for n in (
            "read_file",
            "search_files",
            "web_search",
            "session_search",
            "mcp_filesystem_read_file",
            "mcp_filesystem_list_directory",
        ):
            assert classify_tool(n, {}) == "local", n

    def test_mutating_tools_are_cloud(self):
        for n in (
            "write_file",
            "patch",
            "terminal",
            "execute_code",
            "memory",
            "todo",
            "skill_manage",
            "browser_navigate",
            "browser_scroll",
            "send_message",
            "delegate_task",
            "cronjob",
        ):
            assert classify_tool(n, {}) == "cloud", n

    def test_unknown_tools_default_cloud(self):
        for n in (
            "kanban_create",
            "discord_send",
            "feishu_drive_add",
            "image_generate",
            "video_generate",
            "tts",
            "browser_back",
            "mcp_unknown_thing",
            "vision_analyze",
            "skill_view",
        ):
            assert classify_tool(n, {}) == "cloud", n

    def test_process_read_actions_local_else_cloud(self):
        for a in ("list", "poll", "log", "wait"):
            assert classify_tool("process", {"action": a}) == "local", a
        for a in ("kill", "write", "submit", "close"):
            assert classify_tool("process", {"action": a}) == "cloud", a
        assert classify_tool("process", {}) == "cloud"

    def test_kanban_read_variants_local(self):
        assert classify_tool("kanban_show", {}) == "local"
        assert classify_tool("kanban_list", {}) == "local"
        assert classify_tool("kanban_complete", {}) == "cloud"

    def test_browser_console_tightened(self):
        assert classify_tool("browser_console", {}) == "local"
        assert classify_tool("browser_console", {"clear": False}) == "local"
        assert classify_tool("browser_console", {"expression": "1+1"}) == "cloud"
        assert classify_tool("browser_console", {"clear": True}) == "cloud"


class TestRouteToolCalls:
    def test_all_local_accepts_local(self):
        d = route_tool_calls([_tc("read_file"), _tc("search_files")])
        assert d.backend == "local"

    def test_any_cloud_escalates(self):
        d = route_tool_calls([_tc("read_file"), _tc("write_file")])
        assert d.backend == "cloud"
        assert d.reason == "dangerous_tool"

    def test_args_parsed_from_json_string(self):
        d = route_tool_calls([_tc("process", '{"action": "list"}')])
        assert d.backend == "local"
        d2 = route_tool_calls([_tc("process", '{"action": "kill"}')])
        assert d2.backend == "cloud"

    def test_malformed_json_args_treated_as_empty(self):
        d = route_tool_calls([_tc("process", "{not json")])
        assert d.backend == "cloud"


class TestMessagesHaveImage:
    def test_plain_text_messages_no_image(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert messages_have_image(msgs) is False

    def test_openai_image_url_part_detected(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
                ],
            }
        ]
        assert messages_have_image(msgs) is True

    def test_anthropic_image_block_detected(self):
        msgs = [{"role": "user", "content": [{"type": "image", "source": {}}]}]
        assert messages_have_image(msgs) is True

    def test_empty_or_none_safe(self):
        assert messages_have_image([]) is False
        assert messages_have_image(None) is False


class TestContextOverflow:
    def test_under_threshold_ok(self):
        assert context_overflow(10_000, 2_000, 65_536) is False

    def test_over_threshold_overflows(self):
        assert context_overflow(60_000, 4_000, 65_536) is True

    def test_unknown_context_is_overflow(self):
        assert context_overflow(100, 100, 0) is True
        assert context_overflow(100, 100, None) is True


class TestAcceptLocalText:
    def test_text_final_answers_accept_local_without_keyword_whitelist(self):
        cases = [
            ("帮我改写这句:今天天气不错", "今日天气晴好"),
            ("translate: good morning", "早上好"),
            ("润色一下这段话", "润色后的话"),
            ("summarize this paragraph", "短摘要"),
            ("what is a monad", "a monad is ..."),
            ("评审一下这个架构方案的取舍", "我觉得..."),
            ("你是什么模型", "我是本地模型"),
            ("who are you", "I am a local assistant"),
        ]
        for user_text, output_text in cases:
            d = accept_local_text(user_text, output_text, tools_available=True)
            assert d.backend == "local", user_text

    def test_action_words_do_not_force_cloud_after_local_final_text(self):
        d = accept_local_text("帮我运行测试看看", "好的...", tools_available=True)
        assert d.backend == "local"

    def test_grounding_words_do_not_force_cloud_after_local_final_text(self):
        d = accept_local_text("总结一下当前项目最新进展", "项目...", tools_available=True)
        assert d.backend == "local"

    def test_file_path_prompt_does_not_force_cloud_after_local_final_text(self):
        d = accept_local_text("看看 src/app/main.py 写了啥", "这个文件...", tools_available=True)
        assert d.backend == "local"

    def test_output_text_keywords_do_not_force_cloud_after_local_final_text(self):
        d = accept_local_text("翻译一下", "I don't have access to that file", tools_available=True)
        assert d.backend == "local"

    def test_final_answer_knob_forces_cloud(self):
        d = accept_local_text(
            "翻译一下",
            "你好",
            tools_available=True,
            final_answer_knob="cloud",
        )
        assert d.backend == "cloud"
        assert d.reason == "final_answer_knob"


class TestLocalFingerprint:
    def test_parses_full_env(self):
        fp = LocalFingerprint.from_env(
            {
                "MINT_LOCAL_LLM_URL": "http://127.0.0.1:8765/v1",
                "MINT_LOCAL_LLM_MODEL": "qwen3-4b",
                "MINT_LOCAL_LLM_TOOL_CAPABLE": "1",
                "MINT_LOCAL_LLM_VISION_CAPABLE": "false",
                "MINT_LOCAL_LLM_CONTEXT": "65536",
            }
        )
        assert fp.base_url == "http://127.0.0.1:8765/v1"
        assert fp.model == "qwen3-4b"
        assert fp.tool_capable is True
        assert fp.vision_capable is False
        assert fp.context == 65536

    def test_missing_env_is_conservative(self):
        fp = LocalFingerprint.from_env({})
        assert fp.base_url == ""
        assert fp.model == ""
        assert fp.tool_capable is False
        assert fp.vision_capable is False
        assert fp.context == 0

    def test_bool_variants(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            assert LocalFingerprint.from_env({"MINT_LOCAL_LLM_TOOL_CAPABLE": value}).tool_capable is True
        for value in ("0", "false", "no", "", "garbage"):
            assert LocalFingerprint.from_env({"MINT_LOCAL_LLM_TOOL_CAPABLE": value}).tool_capable is False

    def test_bad_context_is_zero(self):
        assert LocalFingerprint.from_env({"MINT_LOCAL_LLM_CONTEXT": "notanint"}).context == 0


class _FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestHealthGate:
    def test_needs_probe_initially(self):
        gate = HealthGate(_FakeClock(), ok_ttl=20.0, cooldown=30.0)
        assert gate.needs_probe() is True
        assert gate.in_cooldown() is False

    def test_fresh_ok_skips_probe_within_ttl(self):
        clock = _FakeClock()
        gate = HealthGate(clock, ok_ttl=20.0, cooldown=30.0)
        gate.mark_ok()
        assert gate.fresh_ok() is True
        assert gate.needs_probe() is False
        clock.advance(19)
        assert gate.needs_probe() is False
        clock.advance(2)
        assert gate.fresh_ok() is False
        assert gate.needs_probe() is True

    def test_failure_opens_cooldown(self):
        clock = _FakeClock()
        gate = HealthGate(clock, ok_ttl=20.0, cooldown=30.0)
        gate.mark_failure()
        assert gate.in_cooldown() is True
        assert gate.needs_probe() is False
        clock.advance(31)
        assert gate.in_cooldown() is False
        assert gate.needs_probe() is True

    def test_ok_clears_cooldown(self):
        clock = _FakeClock()
        gate = HealthGate(clock, ok_ttl=20.0, cooldown=30.0)
        gate.mark_failure()
        gate.mark_ok()
        assert gate.in_cooldown() is False
        assert gate.fresh_ok() is True


class _StubAgent:
    """Minimal agent surface that apply_runtime_profile touches."""

    def __init__(self):
        self.model = "old-model"
        self.provider = "old-prov"
        self.base_url = "http://old"
        self.api_key = "old-key"
        self.api_mode = "chat_completions"
        self.client = "OLD_CLIENT"
        self._anthropic_client = None
        self._client_kwargs = {"old": 1}
        self._anthropic_api_key = "old-akey"
        self._anthropic_base_url = "http://old-an"
        self._is_anthropic_oauth = False
        self._config_context_length = 4096
        self._transport_cache = {"x": 1}
        self._primary_runtime = "MUST_NOT_CHANGE"
        self._fallback_chain = ["a", "b"]
        self._use_prompt_caching = False
        self._use_native_cache_layout = False

    def _anthropic_prompt_cache_policy(self, **kw):
        return (True, True)


def _profile(**kw):
    base = dict(
        model="cloud-model",
        provider="mint",
        base_url="http://cloud/v1",
        api_key="jwt",
        api_mode="chat_completions",
        client="CLOUD_CLIENT",
        client_kwargs={"k": "v"},
    )
    base.update(kw)
    return RuntimeProfile(**base)


class TestApplyRuntimeProfile:
    def test_swaps_runtime_fields(self):
        agent = _StubAgent()
        apply_runtime_profile(agent, _profile())
        assert agent.model == "cloud-model"
        assert agent.provider == "mint"
        assert agent.base_url == "http://cloud/v1"
        assert agent.api_key == "jwt"
        assert agent.client == "CLOUD_CLIENT"
        assert agent._client_kwargs == {"k": "v"}
        assert agent._config_context_length is None
        assert agent._transport_cache == {}

    def test_does_not_touch_primary_runtime_or_fallback(self):
        agent = _StubAgent()
        apply_runtime_profile(agent, _profile())
        assert agent._primary_runtime == "MUST_NOT_CHANGE"
        assert agent._fallback_chain == ["a", "b"]

    def test_anthropic_profile_swaps_anthropic_client_and_creds(self):
        agent = _StubAgent()
        apply_runtime_profile(
            agent,
            _profile(
                api_mode="anthropic_messages",
                client=None,
                anthropic_client="ANTHRO",
                anthropic_api_key="new-akey",
                anthropic_base_url="http://new-an",
                is_anthropic_oauth=True,
            ),
        )
        assert agent._anthropic_client == "ANTHRO"
        assert agent.client is None
        assert agent._anthropic_api_key == "new-akey"
        assert agent._anthropic_base_url == "http://new-an"
        assert agent._is_anthropic_oauth is True

    def test_reevaluates_prompt_cache(self):
        agent = _StubAgent()
        apply_runtime_profile(agent, _profile())
        assert agent._use_prompt_caching is True
        assert agent._use_native_cache_layout is True

    def test_atomic_rollback_on_failure(self):
        agent = _StubAgent()

        def _boom(**kw):
            raise RuntimeError("build failed")

        agent._anthropic_prompt_cache_policy = _boom
        with pytest.raises(RuntimeError):
            apply_runtime_profile(agent, _profile())
        assert agent.model == "old-model"
        assert agent.provider == "old-prov"
        assert agent.base_url == "http://old"
        assert agent.client == "OLD_CLIENT"
        assert agent._config_context_length == 4096


def _router(**knobs):
    local_profile = _profile(
        model="qwen3-4b",
        provider="mint-local",
        base_url="http://127.0.0.1:8765/v1",
        api_key="no-key",
        client="LOCAL_CLIENT",
    )
    cloud_profile = _profile()
    return HybridRouter(local_profile, cloud_profile, knobs=knobs)


def _fp(tool_capable=True, vision_capable=False, context=65536):
    return LocalFingerprint(
        "http://127.0.0.1:8765/v1",
        "qwen3-4b",
        tool_capable,
        vision_capable,
        context,
    )


def _asst(tool_calls=None, content=""):
    return SimpleNamespace(tool_calls=tool_calls, content=content)


class TestDecidePre:
    def test_tools_but_local_not_tool_capable_session_cloud(self):
        d = _router().decide_pre(
            tools=[{"x": 1}],
            api_messages=[],
            fingerprint=_fp(tool_capable=False),
        )
        assert d.backend == "cloud"
        assert d.reason == "session_tool_incapable"

    def test_image_without_local_vision_cloud(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        d = _router().decide_pre(tools=[], api_messages=msgs, fingerprint=_fp(vision_capable=False))
        assert d.backend == "cloud"
        assert d.reason == "multimodal"

    def test_image_with_local_vision_local(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        d = _router().decide_pre(tools=[], api_messages=msgs, fingerprint=_fp(vision_capable=True))
        assert d.backend == "local"

    def test_cooldown_forces_cloud(self):
        clock = _FakeClock()
        router = _router()
        router.health = HealthGate(clock)
        router.health.mark_failure()
        d = router.decide_pre(tools=[], api_messages=[], fingerprint=_fp())
        assert d.backend == "cloud"
        assert d.reason == "endpoint_down"

    def test_first_step_knob_cloud(self):
        d = _router(first_step="cloud").decide_pre(
            tools=[],
            api_messages=[],
            fingerprint=_fp(),
            is_first_step=True,
        )
        assert d.backend == "cloud"
        assert d.reason == "first_step_knob"

    def test_default_local(self):
        d = _router().decide_pre(tools=[], api_messages=[], fingerprint=_fp())
        assert d.backend == "local"


class TestDecidePost:
    def test_local_tool_calls_accept(self):
        d = _router().decide_post(
            assistant_message=_asst(tool_calls=[_tc("read_file")]),
            user_text="x",
            tools=[1],
            last_tool_failed=False,
        )
        assert d.backend == "local"

    def test_dangerous_tool_escalates(self):
        d = _router().decide_post(
            assistant_message=_asst(tool_calls=[_tc("write_file")]),
            user_text="x",
            tools=[1],
            last_tool_failed=False,
        )
        assert d.backend == "cloud"
        assert d.reason == "dangerous_tool"

    def test_text_simple_accept(self):
        d = _router().decide_post(
            assistant_message=_asst(content="你好世界"),
            user_text="翻译 hello",
            tools=[1],
            last_tool_failed=False,
        )
        assert d.backend == "local"

    def test_text_complex_accepts_local_without_keyword_gate(self):
        d = _router().decide_post(
            assistant_message=_asst(content="我觉得..."),
            user_text="评审这个实现计划",
            tools=[1],
            last_tool_failed=False,
        )
        assert d.backend == "local"

    def test_prior_tool_failure_then_text_escalates(self):
        d = _router().decide_post(
            assistant_message=_asst(content="搞定了"),
            user_text="翻译 hello",
            tools=[1],
            last_tool_failed=True,
        )
        assert d.backend == "cloud"
        assert d.reason == "prior_tool_error"


class _InitStubAgent:
    def __init__(self):
        self.model = "deepseek/deepseek-v4-pro"
        self.provider = "mint"
        self.base_url = "http://localhost:8090/api/ai/openai-compat/v1"
        self.api_key = "jwt"
        self.api_mode = "chat_completions"
        self.client = "CLOUD_CLIENT"
        self._anthropic_client = None
        self._client_kwargs = {"api_key": "jwt", "base_url": self.base_url}
        self._anthropic_api_key = ""
        self._anthropic_base_url = None
        self._is_anthropic_oauth = False
        self._created = []

    def _create_openai_client(self, kwargs, *, reason, shared):
        self._created.append((kwargs, reason))
        return f"CLIENT[{kwargs.get('base_url')}]"


class TestMaybeBuildHybridRouter:
    def test_no_hybrid_env_returns_none(self):
        assert maybe_build_hybrid_router(_InitStubAgent(), {}) is None

    def test_builds_profiles_with_full_fields(self):
        env = {
            "MINT_LOCAL_LLM_URL": "http://127.0.0.1:8765/v1",
            "MINT_LOCAL_LLM_MODEL": "qwen3-4b",
            "MINT_LOCAL_LLM_TOOL_CAPABLE": "1",
            "MINT_LOCAL_LLM_CONTEXT": "65536",
        }
        agent = _InitStubAgent()
        router = maybe_build_hybrid_router(agent, env, clock=_time.monotonic)
        assert router is not None
        assert router.cloud.client == "CLOUD_CLIENT"
        assert router.cloud.client_kwargs == {"api_key": "jwt", "base_url": agent.base_url}
        assert router.local.client == "CLIENT[http://127.0.0.1:8765/v1]"
        assert router.local.client_kwargs["base_url"] == "http://127.0.0.1:8765/v1"
        assert router.fingerprint.tool_capable is True
        assert router.health is not None


class TestRedoMachineUnit:
    def test_decide_pre_force_cloud(self):
        d = _router().decide_pre(
            tools=[],
            api_messages=[],
            fingerprint=_fp(),
            force_cloud=True,
        )
        assert d.backend == "cloud"
        assert d.reason == "redo_cloud"

    def test_escalate_helper_sets_state_and_refunds(self):
        from agent.conversation_loop import _hybrid_escalate

        class _Budget:
            def __init__(self):
                self.refunded = 0

            def refund(self):
                self.refunded += 1

        class _Agent:
            pass

        agent = _Agent()
        agent._ephemeral_max_output_tokens = 99
        agent.iteration_budget = _Budget()
        agent.hybrid_router = _router()
        agent.hybrid_router.health = HealthGate(lambda: 0.0)

        _hybrid_escalate(agent, "local_api_error", ephemeral_snap=None)

        assert agent._hybrid_redo_on_cloud is True
        assert agent._hybrid_pending_reason == "local_api_error"
        assert agent._ephemeral_max_output_tokens is None
        assert agent.iteration_budget.refunded == 1
        assert agent.hybrid_router.health.in_cooldown() is True
        assert agent._hybrid_attempts[-1]["reason"] == "local_api_error"


class TestHybridTelemetryUnit:
    def test_record_accepted_local_attempt_counts_and_clears_pending_reason(self):
        from agent.conversation_loop import _hybrid_record_accepted_attempt

        class _Agent:
            pass

        agent = _Agent()
        agent.hybrid_router = object()
        agent._hybrid_attempt_backend = "local"
        agent._hybrid_pending_reason = "dangerous_tool"

        _hybrid_record_accepted_attempt(agent, 1.234)

        assert agent._hybrid_attempts == [
            {
                "backend": "local",
                "reason": "accept",
                "accepted": True,
                "duration_ms": 1234,
            }
        ]
        assert agent._hybrid_local_calls == 1
        assert getattr(agent, "_hybrid_cloud_calls", 0) == 0
        assert agent._hybrid_pending_reason is None

    def test_record_accepted_cloud_attempt_uses_pending_reason(self):
        from agent.conversation_loop import _hybrid_record_accepted_attempt

        class _Agent:
            pass

        agent = _Agent()
        agent.hybrid_router = object()
        agent._hybrid_attempt_backend = "cloud"
        agent._hybrid_pending_reason = "dangerous_tool"

        _hybrid_record_accepted_attempt(agent, 0.2)

        assert agent._hybrid_attempts == [
            {
                "backend": "cloud",
                "reason": "dangerous_tool",
                "accepted": True,
                "duration_ms": 200,
            }
        ]
        assert getattr(agent, "_hybrid_local_calls", 0) == 0
        assert agent._hybrid_cloud_calls == 1
        assert agent._hybrid_pending_reason is None

    def test_attach_hybrid_stats_to_turn_result(self):
        from agent.turn_finalizer import attach_hybrid_stats

        class _Agent:
            pass

        attempts = [{"backend": "local", "reason": "accept"}]
        agent = _Agent()
        agent.hybrid_router = object()
        agent._hybrid_local_calls = 2
        agent._hybrid_cloud_calls = 1
        agent._hybrid_attempts = attempts
        result = {}

        attach_hybrid_stats(result, agent)

        assert result["hybrid"] == {
            "local_calls": 2,
            "cloud_calls": 1,
            "attempts": attempts,
        }

    def test_hybrid_api_request_id_adds_attempt_backend_suffix(self):
        from agent.conversation_loop import _hybrid_api_request_id

        class _Agent:
            pass

        agent = _Agent()
        agent.hybrid_router = object()
        agent._hybrid_attempt_backend = "local"

        assert _hybrid_api_request_id(agent, "turn-1", 3) == (
            "turn-1:api:3:attempt:local"
        )

    def test_prepare_cloud_for_internal_work_switches_profile(self):
        from agent.conversation_loop import _hybrid_prepare_cloud_for_internal_work

        router = _router()

        class _Agent:
            pass

        agent = _Agent()
        agent.model = router.local.model
        agent.provider = router.local.provider
        agent.base_url = router.local.base_url
        agent.api_key = router.local.api_key
        agent.api_mode = router.local.api_mode
        agent.client = router.local.client
        agent._anthropic_client = None
        agent._client_kwargs = {}
        agent._anthropic_api_key = ""
        agent._anthropic_base_url = None
        agent._is_anthropic_oauth = False
        agent._hybrid_attempt_backend = "local"
        agent._hybrid_local_attempt = True

        _hybrid_prepare_cloud_for_internal_work(agent, router)

        assert agent.model == router.cloud.model
        assert agent.client == router.cloud.client
        assert agent._hybrid_attempt_backend == "cloud"
        assert agent._hybrid_local_attempt is False
