"""Claude 생각 수준(effort) 전달, 최대 출력 한도, 잘린 답 알림."""
import asyncio
import json

import pytest


def efforts_by_stage(core, ai):
    by_system = {core.HUB_BRIEF_SYSTEM: "brief", core.HUB_VERDICT_SYSTEM: "verdict",
                 core.JUDGE_SYSTEM: "judge", core.DEBATE_SYSTEM: "debate", core.CLARIFY_SYSTEM: "clarify"}
    return {(by_system[s], m): e for s, m, e in ai.efforts}


def test_effort_per_stage(core, run, ai):
    run({"hub": "Claude", "a": "Gemini", "b": "Claude"},
        {"hub": "claude-sonnet-5-5", "a": "gemini-3.8-flash", "b": "claude-haiku-5-5"})
    e = efforts_by_stage(core, ai)
    assert e[("brief", "claude-sonnet-5-5")] == "medium"
    assert e[("verdict", "claude-sonnet-5-5")] == "high"
    assert e[("judge", "claude-haiku-5-5")] is None and e[("judge", "gemini-3.8-flash")] is None
    ai.efforts.clear()
    asyncio.run(core.hub_clarify(core.UserInput(prompt="q"), "Claude", "claude-sonnet-5-5", {}))
    assert efforts_by_stage(core, ai)[("clarify", "claude-haiku-5-5")] == "low"


def test_effort_not_sent_to_non_claude_hub(core, run, ai):
    run({"hub": "Gemini", "a": "Groq", "b": "Mistral"},
        {"hub": "gemini-3.8-flash", "a": "openai/gpt-oss-120b", "b": "mistral-small-latest"})
    assert all(e is None for _, _, e in ai.efforts)


def test_truncated_answer_is_reported(run, ai):
    ai.truncate = ("Claude", "claude-sonnet-5-5")
    r = run()
    assert any("잘렸을 수 있습니다" in n for n in r["notices"])


# ── 실제 SDK 요청 내용 ──
def sdk_core(core, handler):
    httpx = pytest.importorskip("httpx2", reason="이 SDK 버전은 httpx2를 씀")
    from anthropic import AsyncAnthropic

    class Fake(AsyncAnthropic):
        def __init__(self, **kw):
            super().__init__(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)
    core["AsyncAnthropic"] = Fake
    return httpx


def message(text, stop="end_turn"):
    return {"id": "m", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_sdk_request_has_effort_and_max_tokens_and_detects_truncation(core):
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json=message("잘린 답", stop="max_tokens"))
    httpx = sdk_core(core, handler)
    out, meta = asyncio.run(core.call_anthropic(core.PROVIDERS["Claude"], "claude-sonnet-5-5", "sys", "q", [],
                                                search=True, effort="medium"))
    assert bodies[0]["output_config"] == {"effort": "medium"}
    assert bodies[0]["max_tokens"] == core.CLAUDE_MAX_TOKENS
    assert meta["truncated"] is True


def test_sdk_retries_without_effort_when_model_rejects_it(core):
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if "output_config" in body:
            return httpx.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": "output_config.effort is not supported for this model"}})
        return httpx.Response(200, json=message("정상 답"))
    httpx = sdk_core(core, handler)
    out, meta = asyncio.run(core.call_anthropic(core.PROVIDERS["Claude"], "claude-haiku-4-5", "sys", "q", [],
                                                effort="low"))
    assert out == "정상 답" and "output_config" in bodies[0] and "output_config" not in bodies[-1]
    assert meta["truncated"] is False
