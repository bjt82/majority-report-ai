"""Claude 사용량·비용 계산, 허브와 같은 AI의 판단자(Haiku) 배치."""
import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from conftest import DAILY, MODELS

ROLES = {"hub": "Claude", "a": "Gemini", "b": "Claude"}
MODELS_AB = {"hub": "claude-sonnet-5-5", "a": "gemini-3.8-flash", "b": "claude-haiku-5-5"}


def test_usage_and_cost(core):
    r1 = NS(usage=NS(input_tokens=10000, output_tokens=2000, cache_creation_input_tokens=0,
                     cache_read_input_tokens=0, server_tool_use=NS(web_search_requests=2)))
    r2 = NS(usage=NS(input_tokens=5000, output_tokens=1000, cache_creation_input_tokens=None,
                     cache_read_input_tokens=None, server_tool_use=None))
    u = core.usage_of([r1, r2])
    assert u == {"input": 15000, "output": 3000, "searches": 2}
    assert core.cost_of("claude-sonnet-5-5", u) == pytest.approx((15000 * 2 + 3000 * 10) / 1e6 + 0.02)
    assert core.cost_of("claude-haiku-5-5", {"input": 10000, "output": 2000, "searches": 0}) == pytest.approx(0.002)
    assert core.cost_of("gemini-3.8-flash", u) == 0 and core.cost_of("claude-sonnet-5-5", None) == 0


def test_judge_model_on_hub_provider_is_lower_model(core):
    assert core.judge_model("Claude", "Claude", "claude-sonnet-5-5") == "claude-haiku-5-5"
    assert core.judge_model("Gemini", "Claude", "claude-sonnet-5-5") == "gemini-3.8-flash"


def test_turn_cost_and_haiku_judge(core, run, ai):
    r = run(ROLES, MODELS_AB, search=True)
    assert r["firsts"][1]["model"] == "claude-haiku-5-5" and not r["notices"]
    u = {**ai.usage, "searches": 0}
    expected = 2 * core.cost_of("claude-sonnet-5-5", u) + 0.01 + 2 * core.cost_of("claude-haiku-5-5", u)
    assert r["cost"] == pytest.approx(expected)


def test_free_judge_exhausted_never_duplicates_haiku(run, ai):
    for m in MODELS["Gemini"] + ["gemini-flash-lite-latest", "gemini-flash-latest"]:
        ai.behavior[("Gemini", m)] = DAILY
    assert run(ROLES, MODELS_AB)["firsts"][0]["name"] == "Groq"
    for m in MODELS["Groq"] + ["qwen/qwen3-32b"]:
        ai.behavior[("Groq", m)] = DAILY
    for m in MODELS["Mistral"]:
        ai.behavior[("Mistral", m)] = DAILY
    r = run(ROLES, MODELS_AB)
    assert (r["firsts"][0]["name"], r["firsts"][0]["model"]) == ("Claude", "claude-sonnet-5-5")
    assert ai.calls.count(("Claude", "claude-haiku-5-5")) == 2      # Haiku는 판단 B로만 (1차 + 반론)


def test_haiku_overload_goes_to_free_ai(run, ai):
    ai.behavior[("Claude", "claude-haiku-5-5")] = "529 overloaded"
    assert run(ROLES, MODELS_AB)["firsts"][1]["name"] == "Groq"


def test_call_anthropic_collects_usage_through_real_sdk(core):
    """실제 Anthropic SDK를 모의 HTTP 서버에 연결해 일반·스트리밍 응답의 usage 수집을 확인."""
    httpx = pytest.importorskip("httpx2", reason="이 SDK 버전은 httpx2를 씀")
    from anthropic import AsyncAnthropic

    def msg(text, usage, stop="end_turn"):
        return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": text}], "stop_reason": stop, "stop_sequence": None,
                "usage": usage}

    def handler(req):
        if json.loads(req.content).get("stream"):
            start = {**msg("", {"input_tokens": 1200, "output_tokens": 1}), "content": [], "stop_reason": None}
            ev = [("message_start", {"type": "message_start", "message": start}),
                  ("content_block_start", {"type": "content_block_start", "index": 0,
                                           "content_block": {"type": "text", "text": ""}}),
                  ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "text_delta", "text": "판정 결과"}}),
                  ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                  ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                     "usage": {"output_tokens": 800}}),
                  ("message_stop", {"type": "message_stop"})]
            data = "".join(f"event: {e}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n" for e, d in ev)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data.encode())
        usage = {"input_tokens": 20000, "output_tokens": 3000,
                 "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0}}
        return httpx.Response(200, json=msg("정리 결과", usage))

    class Fake(AsyncAnthropic):
        def __init__(self, **kw):
            super().__init__(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)
    core["AsyncAnthropic"] = Fake
    p = core.PROVIDERS["Claude"]
    out, meta = asyncio.run(core.call_anthropic(p, "claude-sonnet-5-5", "sys", "q", [], search=True))
    assert out == "정리 결과" and meta["usage"] == {"input": 20000, "output": 3000, "searches": 2}
    chunks = []
    out, meta = asyncio.run(core.call_anthropic(p, "claude-sonnet-5-5", "sys", "q", [], on_text=chunks.append))
    assert out == "판정 결과" and chunks == ["판정 결과"] and meta["usage"] == {"input": 1200, "output": 800, "searches": 0}
