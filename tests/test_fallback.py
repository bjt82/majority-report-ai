"""한도 초과·오류 시 판단자 교체, 모델 단위 자동 제외, 재시도·시간 제한."""
import asyncio
import time

from conftest import DAILY, MODELS


class QuotaWithHeader(Exception):
    def __init__(self, msg, retry_after):
        super().__init__(msg)
        self.response = type("R", (), {"headers": {"retry-after": str(retry_after)}})()


def test_fallback_chain_uses_real_model_list(core):
    assert core.chain("Gemini", "gemini-3.8-flash") == ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.5-flash"]
    assert "qwen/qwen3-32b" not in core.chain("Groq", "openai/gpt-oss-120b")   # 목록에 없는 모델은 제외


def test_normal_run(run):
    r = run()
    assert r["final"]["ok"] and not r["notices"] and len(r["debates"]) == 2


def test_daily_quota_switches_to_lower_model_and_cools_that_model_only(run, ai):
    ai.behavior[("Gemini", "gemini-3.8-flash")] = DAILY
    cd = {}
    r = run(cooldowns=cd)
    assert r["firsts"][0]["model"] == "gemini-3.8-flash-lite" and r["firsts"][0]["ok"]
    assert "Gemini/gemini-3.8-flash" in cd and "Gemini/gemini-3.8-flash-lite" not in cd
    assert r["debates"][0]["model"] == "gemini-3.8-flash-lite"
    assert ai.calls.count(("Gemini", "gemini-3.8-flash")) == 1          # 하루 한도는 재시도하지 않음
    # 다음 질문: 제외 중인 모델은 부르지 않고 바로 낮은 모델
    run(cooldowns=cd)
    assert not ai.called("Gemini", "gemini-3.8-flash")
    # 제외가 풀리면 원래 모델로 복귀
    cd["Gemini/gemini-3.8-flash"] = time.time() - 1
    ai.behavior.clear()
    r = run(cooldowns=cd)
    assert r["firsts"][0]["model"] == "gemini-3.8-flash" and not r["notices"]


def test_exhausted_provider_moves_to_other_ai_then_hub_other_model(run, ai):
    for m in MODELS["Gemini"][:3]:
        ai.behavior[("Gemini", m)] = DAILY
    assert run()["firsts"][0]["name"] == "Mistral"
    for m in MODELS["Mistral"]:
        ai.behavior[("Mistral", m)] = DAILY
    r = run()
    assert (r["firsts"][0]["name"], r["firsts"][0]["model"]) == ("Claude", "claude-haiku-5-5")


def test_minute_quota_with_long_retry_after_switches_immediately(run, ai):
    async def minute():
        raise QuotaWithHeader("429 rate_limit_exceeded tokens per minute", 45)
    ai.behavior[("Groq", "openai/gpt-oss-120b")] = minute
    cd = {}
    t = time.time()
    r = run(cooldowns=cd)
    assert r["firsts"][1]["model"] == "openai/gpt-oss-20b"
    assert time.time() - t < 3
    assert 40 < cd["Groq/openai/gpt-oss-120b"] - time.time() <= 61


def test_judge_timeout_is_total_cap(run, ai):
    async def slow():
        await asyncio.sleep(5)
        return "late", {}
    ai.behavior[("Gemini", "gemini-3.8-flash")] = slow
    t = time.time()
    r = run(judge_timeout=1.5)
    assert time.time() - t < 4
    assert not r["firsts"][0]["ok"] and r["firsts"][0].get("timeout")


def test_retry_after_parsing(core):
    ra = core.retry_after
    assert ra(Exception("Please retry in 23.45s.")) == 23.45
    assert ra(Exception("'retryDelay': '23s'")) == 23.0
    assert ra(Exception("Please try again in 7.66s.")) == 7.66
    assert ra(Exception("try again in 500ms")) is None


def test_hub_model_cooled_uses_hub_lower_model(run):
    cd = {"Claude/claude-sonnet-5-5": time.time() + 600}
    r = run(cooldowns=cd)
    assert r["brief"]["model"] == "claude-haiku-5-5" and r["final"]["model"] == "claude-haiku-5-5"


def test_overload_prefers_same_provider_lower_model(run, ai):
    ai.behavior[("Gemini", "gemini-3.8-flash")] = "ServerError: 503 UNAVAILABLE. The model is overloaded."
    cd = {}
    r = run(cooldowns=cd)
    assert r["firsts"][0]["model"] == "gemini-3.8-flash-lite"
    assert "서버 과부하" in r["notices"][0] and "Gemini/gemini-3.8-flash" not in cd   # 과부하는 제외하지 않음


def test_model_list_failure_uses_guess_and_aliases_and_remembers_404(core, run, ai):
    core["model_ids"] = lambda p: None if p == "Gemini" else MODELS.get(p)
    assert core.chain("Gemini", "gemini-3.8-flash") == [
        "gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]
    ai.behavior[("Gemini", "gemini-3.8-flash")] = DAILY
    ai.behavior[("Gemini", "gemini-3.8-flash-lite")] = "404 NOT_FOUND models/gemini-3.8-flash-lite is not found"
    cd = {}
    r = run(cooldowns=cd)
    assert (r["firsts"][0]["name"], r["firsts"][0]["model"]) == ("Gemini", "gemini-flash-lite-latest")
    assert core.is_missing("Gemini", "gemini-3.8-flash-lite")
    run(cooldowns=cd)
    assert not ai.called("Gemini", "gemini-3.8-flash-lite")


def test_notice_explains_when_lower_models_all_cooled(run, ai):
    ai.behavior[("Gemini", "gemini-3.8-flash")] = "503 UNAVAILABLE overloaded"
    cd = {"Gemini/gemini-3.8-flash-lite": time.time() + 600, "Gemini/gemini-3.5-flash": time.time() + 600}
    r = run(cooldowns=cd)
    assert r["firsts"][0]["name"] == "Mistral" and "낮은 모델도 사용 불가" in r["notices"][0]


def test_no_judge_at_all_still_returns_verdict(core, ai):
    import asyncio
    for name, ms in MODELS.items():
        if name != "Claude":
            for m in ms + ["gemini-flash-lite-latest", "gemini-flash-latest", "qwen/qwen3-32b"]:
                ai.behavior[(name, m)] = DAILY
    # Claude는 판단자(대타 포함)로 불릴 때만 실패하고, 허브의 정리·판정은 성공하게 함
    orig = core.CALLERS["anthropic"]

    async def anthropic(p, model, system, text, images, search=False, on_text=None, effort=None):
        if system in (core.JUDGE_SYSTEM, core.DEBATE_SYSTEM):
            raise Exception(DAILY)
        return await orig(p, model, system, text, images, search=search, on_text=on_text, effort=effort)
    core.CALLERS["anthropic"] = anthropic
    ui = core.UserInput(prompt="q")
    r = asyncio.run(core.run_pipeline(ui, {"hub": "Claude", "a": "Gemini", "b": "Groq"},
                                      {"hub": "claude-sonnet-5-5", "a": "gemini-3.8-flash", "b": "openai/gpt-oss-120b"},
                                      lambda m: None, {}))
    assert not any(f["ok"] for f in r["firsts"])
    assert r["final"]["ok"] and r["debates"] == []
    assert any("허브가 혼자 답합니다" in n for n in r["notices"])
