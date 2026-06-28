"""Integration: decide_post tier-gate + apply_runtime_profile switch contract."""

from __future__ import annotations

from types import SimpleNamespace

from agent.hybrid_router import (
    HealthGate,
    HybridRouter,
    LocalFingerprint,
    RuntimeProfile,
    apply_runtime_profile,
)


def _router():
    local_profile = RuntimeProfile(
        "qwen3-4b",
        "mint-local",
        "http://127.0.0.1:8765/v1",
        "no-key",
        "chat_completions",
        client="LOCAL",
        client_kwargs={},
    )
    cloud_profile = RuntimeProfile(
        "cloud-model",
        "mint",
        "http://cloud/v1",
        "jwt",
        "chat_completions",
        client="CLOUD",
        client_kwargs={"k": 1},
    )
    fingerprint = LocalFingerprint(
        "http://127.0.0.1:8765/v1",
        "qwen3-4b",
        True,
        False,
        65536,
    )
    return HybridRouter(
        local_profile,
        cloud_profile,
        fingerprint=fingerprint,
        knobs={},
        health=HealthGate(lambda: 0.0),
    )


def _tc(name):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments="{}"))


class TestEscalationContract:
    def test_local_readfile_accepts(self):
        decision = _router().decide_post(
            assistant_message=SimpleNamespace(
                tool_calls=[_tc("read_file")],
                content="",
            ),
            user_text="read it",
            tools=[1],
            last_tool_failed=False,
        )

        assert decision.backend == "local"

    def test_local_writefile_escalates(self):
        decision = _router().decide_post(
            assistant_message=SimpleNamespace(
                tool_calls=[_tc("write_file")],
                content="",
            ),
            user_text="change it",
            tools=[1],
            last_tool_failed=False,
        )

        assert decision.backend == "cloud"
        assert decision.reason == "dangerous_tool"

    def test_apply_profile_switches_client_and_kwargs(self):
        router = _router()
        agent = SimpleNamespace(
            model="qwen3-4b",
            provider="mint-local",
            base_url="http://127.0.0.1:8765/v1",
            api_key="no-key",
            api_mode="chat_completions",
            client="LOCAL",
            _anthropic_client=None,
            _client_kwargs={},
            _anthropic_api_key="",
            _anthropic_base_url=None,
            _is_anthropic_oauth=False,
            _config_context_length=64000,
            _transport_cache={},
            _primary_runtime="KEEP",
        )

        apply_runtime_profile(agent, router.cloud)

        assert agent.client == "CLOUD"
        assert agent.model == "cloud-model"
        assert agent._client_kwargs == {"k": 1}
        assert agent._primary_runtime == "KEEP"

    def test_hybrid_local_attempt_rewrites_cached_cloud_prompt_identity(self):
        from agent.conversation_loop import _hybrid_active_system_prompt

        router = _router()
        agent = SimpleNamespace(
            hybrid_router=router,
            _hybrid_attempt_backend="local",
            model="qwen3-4b",
            provider="mint-local",
        )
        prompt = (
            "Identity block\n\n"
            "Conversation started: Tuesday, June 23, 2026\n"
            "Model: deepseek/deepseek-v4-pro\n"
            "Provider: custom"
        )

        rewritten = _hybrid_active_system_prompt(agent, prompt)

        assert "Model: qwen3-4b" in rewritten
        assert "Provider: mint-local" in rewritten
        assert "deepseek/deepseek-v4-pro" not in rewritten
        assert "Provider: custom" not in rewritten
