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
    assert core.cost_of("gemini-3.8-flash", u) == pytest.approx((15000 * .75 + 3000 * 3.75) / 1e6)
    assert core.cost_of("claude-sonnet-5-5", None) == 0


def test_judge_model_on_hub_provider_is_lower_model(core):
    assert core.judge_model("Claude", "Claude", "claude-sonnet-5-5") == "claude-haiku-5-5"
    assert core.judge_model("Gemini", "Claude", "claude-sonnet-5-5") == "gemini-3.8-flash"


def test_turn_cost_and_haiku_judge(core, run, ai):
    r = run(ROLES, MODELS_AB, search=True)
    assert r["firsts"][1]["model"] == "claude-haiku-5-5" and not r["notices"]
    u = {**ai.usage, "searches": 0}
    expected = 2 * core.cost_of("claude-sonnet-5-5", u) + 0.01 + 2 * core.cost_of("claude-haiku-5-5", u)
    assert r["costs"]["Claude"] == pytest.approx(expected)
    assert r["costs"]["Gemini"] == pytest.approx(2 * core.cost_of("gemini-3.8-flash", u), abs=1e-6)
    assert r["cost"] == pytest.approx(sum(r["costs"].values()), abs=2e-6)


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


def test_pretty_model_names(core):
    assert core.pretty("claude-sonnet-5-5") == "Sonnet 5.5"
    assert core.pretty("claude-haiku-5-5") == "Haiku 5.5"
    assert core.pretty("gemini-3.8-flash") == "Gemini 3.8 Flash"
    assert core.pretty("openai/gpt-oss-120b") == "gpt-oss-120b"
    assert core.pretty("") == ""


def test_gemini_thinking_cache_and_price_transition(core):
    usage = core.gemini_usage(NS(usage_metadata=NS(prompt_token_count=10000,
        candidates_token_count=500, thoughts_token_count=1500, cached_content_token_count=4000)))
    assert usage == {"input": 10000, "output": 2000, "cached_input": 4000, "searches": 0}
    core["today"] = lambda: "2026-12-31"
    assert core.cost_of("gemini-3.8-flash", usage) == pytest.approx((6000*.75 + 4000*.075 + 2000*3.75)/1e6)
    core["today"] = lambda: "2027-01-01"
    assert core.cost_of("gemini-3.8-flash", usage) == pytest.approx((6000*1.5 + 4000*.15 + 2000*7.5)/1e6)
    old_secret = core.secret
    core["secret"] = lambda k, d=None: False if k == "GEMINI_PAID_TIER" else old_secret(k, d)
    assert core.cost_of("gemini-3.8-flash", usage) == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_gemini_sdk_usage_and_alias(core, monkeypatch, streaming):
    """일반·스트리밍 usage, 생각 토큰, 별칭 모델의 실제 단가를 SDK 경계에서 확인."""
    def response(text, output):
        return NS(text=text, candidates=None, model_version="gemini-3.8-flash",
                  usage_metadata=NS(prompt_token_count=1000, candidates_token_count=output,
                                    thoughts_token_count=100, cached_content_token_count=200))

    class Models:
        async def generate_content(self, **kw):
            return response("완성", 200)

        async def generate_content_stream(self, **kw):
            async def chunks():
                yield response("완", 50)
                yield response("성", 200)
            return chunks()

    monkeypatch.setattr(core.genai, "Client", lambda **kw: NS(aio=NS(models=Models())))
    chunks = []
    core["CALLERS"]["gemini"] = core.call_gemini
    r = asyncio.run(core.ask("판정", "Gemini", "gemini-flash-latest", "sys", "q", [],
                            on_text=chunks.append if streaming else None))
    assert r["ok"] and r["text"] == "완성"
    assert r["usage"] == {"input": 1000, "output": 300, "cached_input": 200, "searches": 0}
    assert r["billing_model"] == "gemini-3.8-flash" and not r["cost_note"]
    assert r["cost"] == pytest.approx((800*.75 + 200*.075 + 300*3.75)/1e6)
    if streaming:
        assert chunks == [None, "완", "성"]


def test_unknown_gemini_price_is_reported(core):
    async def unknown(*a, **kw):
        return "응답", {"usage": {"input": 1000, "output": 100, "searches": 0}}
    core.CALLERS["gemini"] = unknown
    r = asyncio.run(core.ask("판단", "Gemini", "gemini-unknown", "sys", "q", []))
    assert r["ok"] and "단가가 없어" in r["cost_note"]


def test_monthly_totals_preserve_old_claude_records(core):
    old = {"cost": {"2026-11": .25, "2026-10": 3}}
    new = {"cost": {"2026-11": .15},
           "cost_by_provider": {"Claude": {"2026-11": .1}, "Gemini": {"2026-11": .05}}}
    assert core.monthly_costs([old, new], "2026-11") == pytest.approx({"Claude": .35, "Gemini": .05})
    assert core.result_costs({"cost": .25}) == {"Claude": .25, "Gemini": 0}


def test_pipeline_updates_live_provider_costs(core, run):
    class Board(core.NullBoard):
        def __init__(self):
            self.updates = []
        def update_cost(self, costs):
            self.updates.append(dict(costs))
    board = Board()
    r = run(ROLES, MODELS_AB, board=board)
    assert len(board.updates) == 6
    assert board.updates[-1] == r["costs"]
    assert r["costs"]["Claude"] > 0 and r["costs"]["Gemini"] > 0


def test_clarify_cost_is_attributed_to_gemini(core, ai):
    core["fallbacks"] = lambda name, model: []
    details = {}
    _, cost = asyncio.run(core.hub_clarify(core.UserInput(prompt="q"), "Gemini",
                                         "gemini-3.8-flash", {}, cost_details=details))
    assert cost > 0 and details == {"Gemini": cost}


def test_claude_cache_rates_and_long_haiku(core):
    usage = {"input": 10000, "output": 1000, "searches": 0,
             "cache_read": 3000, "cache_write": 2000, "cache_write_1h": 1000}
    expected = (5000*2 + 3000*.1 + 1000*2.5 + 1000*4 + 1000*10)/1e6
    assert core.cost_of("claude-sonnet-5-5", usage) == pytest.approx(expected)
    assert core.cost_of("claude-haiku-5-5", {"input": 100001, "output": 2000, "searches": 0}) == pytest.approx(
        (100001*.5 + 2000*2.5)/1e6)
