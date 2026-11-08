"""실제 입력 기준 판단 B 선택, 무료 대기 예산, Haiku 전환과 취소 검증."""
import asyncio
import time

import pytest

from conftest import DAILY

ROLES = {"hub": "Claude", "a": "Gemini", "b": "Claude"}
MODELS = {"hub": "claude-sonnet-5-5", "a": "gemini-3.8-flash", "b": "claude-haiku-5-5"}


def invoke(core, ui=None, **kw):
    return asyncio.run(core.run_pipeline(ui or core.UserInput(prompt="짧은 질문"), ROLES, MODELS,
                                         lambda m: None, kw.pop("cooldowns", {}), auto_judge_b=True, **kw))


@pytest.mark.parametrize("kind", ["document", "image", "history", "question"])
def test_full_input_routes_directly_to_haiku(core, ai, kind):
    ui = core.UserInput(prompt="자료를 검토해 줘")
    if kind == "document":
        ui.doc_text = "긴 문서 " * 1500
    elif kind == "image":
        ui.images = [core.Attachment(b"image", "image/png")]
    elif kind == "history":
        ui.history = [("이전 질문", "이전 답변 " * 1500)]
    else:
        ui.prompt = "긴 질문 " * 1500
    r = invoke(core, ui)
    assert r["judge_b_route"]["mode"] == "full"
    assert r["firsts"][1]["model"] == "claude-haiku-5-5"
    assert not any(n in ("Groq", "Mistral") for n, _ in ai.calls)


def test_short_text_uses_third_provider_in_both_stages(core, ai):
    r = invoke(core)
    assert r["firsts"][1]["name"] == "Groq"
    assert r["debates"][1]["name"] == "Groq"
    assert r["firsts"][0]["name"] == "Gemini" and r["final"]["name"] == "Claude"
    assert r["costs"]["Claude"] > 0 and r["costs"]["Gemini"] > 0


def test_groq_quota_moves_directly_to_mistral(core, ai):
    ai.behavior[("Groq", "openai/gpt-oss-120b")] = DAILY
    cd = {}
    r = invoke(core, cooldowns=cd)
    assert r["firsts"][1]["name"] == "Mistral" and r["debates"][1]["name"] == "Mistral"
    assert "Groq/openai/gpt-oss-120b" in cd
    assert not ai.called("Groq", "openai/gpt-oss-20b")


def test_free_errors_fall_back_to_haiku_and_keep_costs(core, ai):
    ai.behavior[("Groq", "openai/gpt-oss-120b")] = DAILY
    ai.behavior[("Mistral", "mistral-small-latest")] = "503 unavailable"
    r = invoke(core)
    assert r["firsts"][1]["model"] == r["debates"][1]["model"] == "claude-haiku-5-5"
    assert ai.calls.count(("Claude", "claude-haiku-5-5")) == 2
    assert r["costs"]["Claude"] == pytest.approx(.0147)


def test_timeouts_cancel_free_calls_and_leave_time_for_haiku(core, ai):
    core["AUTO_B_FREE_BUDGET_SEC"] = .10
    core["AUTO_B_FREE_CALL_SEC"] = .05
    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    ai.behavior[("Groq", "openai/gpt-oss-120b")] = slow
    ai.behavior[("Mistral", "mistral-small-latest")] = slow
    started = time.perf_counter()
    r = invoke(core, judge_timeout=.6)
    assert time.perf_counter() - started < 1
    assert len(cancelled) == 2
    assert r["firsts"][1]["ok"] and r["firsts"][1]["model"] == "claude-haiku-5-5"


def test_free_budget_is_shared_between_first_and_debate(core, ai, monkeypatch):
    # Windows 타이머/시스템 부하가 30ms 대기를 넘겨도 예산 검증 결과는 같아야 함.
    from types import SimpleNamespace
    import majority.pipeline as pipeline
    elapsed = 0.0
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(perf_counter=lambda: elapsed, time=time.time))
    core["AUTO_B_FREE_BUDGET_SEC"] = .09
    core["AUTO_B_FREE_CALL_SEC"] = .06
    calls = []
    original = core.ask

    async def record(*a, **kw):
        nonlocal elapsed
        if a[0].startswith("판단 B") and a[1] in ("Groq", "Mistral"):
            calls.append(kw["timeout"])
            if len(calls) == 1:
                elapsed += .04
            else:
                elapsed += kw["timeout"]
                return {"role": a[0], "name": a[1], "model": a[2], "ok": False,
                        "timeout": True, "text": "시험용 시간 초과", "sec": kw["timeout"]}
        return await original(*a, **kw)

    core["ask"] = record
    r = invoke(core, judge_timeout=.6)
    assert r["firsts"][1]["name"] == "Groq"
    assert r["debates"][1]["model"] == "claude-haiku-5-5"
    assert calls[1] < .06  # 1차 호출에 쓴 시간을 반론 단계에서 다시 받지 않음


def test_missing_keys_cooled_models_and_duplicate_a_are_skipped(core):
    core["available_providers"] = lambda: ["Claude", "Gemini"]
    ui = core.UserInput(prompt="q")
    assert core.judge_b_plan(ui, ROLES, MODELS, {})["candidates"][0] == ("Claude", "claude-haiku-5-5")
    core["available_providers"] = lambda: ["Claude", "Gemini", "Groq", "Mistral"]
    cd = {"Groq/openai/gpt-oss-120b": time.time() + 60}
    assert core.judge_b_plan(ui, ROLES, MODELS, cd)["candidates"][0][0] == "Mistral"
    roles, models = {**ROLES, "a": "Groq"}, {**MODELS, "a": "openai/gpt-oss-120b"}
    assert core.judge_b_plan(ui, roles, models, {})["candidates"][0][0] == "Mistral"


def test_manual_b_still_uses_selected_haiku(core, run):
    r = run(ROLES, MODELS, auto_judge_b=False)
    assert r["firsts"][1]["model"] == "claude-haiku-5-5" and r["judge_b_route"] is None


def test_stage_deadline_also_bounds_paid_fallback(core, ai):
    core["AUTO_B_FREE_BUDGET_SEC"] = .10
    core["AUTO_B_FREE_CALL_SEC"] = .05
    async def slow():
        await asyncio.sleep(5)
        return "느린 응답"
    for pair in [("Groq", "openai/gpt-oss-120b"), ("Mistral", "mistral-small-latest"),
                 ("Claude", "claude-haiku-5-5")]:
        ai.behavior[pair] = slow
    started = time.perf_counter()
    r = invoke(core, judge_timeout=.2, fast=True)
    assert time.perf_counter() - started < .6
    assert not r["firsts"][1]["ok"] and r["final"]["ok"]


def test_cancel_interrupts_free_candidate_without_starting_paid_backup(core, ai):
    cancelled = []
    async def slow():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
    ai.behavior[("Groq", "openai/gpt-oss-120b")] = slow

    async def scenario():
        task = asyncio.create_task(core.run_pipeline(core.UserInput(prompt="q"), ROLES, MODELS,
                                                     lambda m: None, {}, auto_judge_b=True))
        await asyncio.sleep(.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert cancelled and not ai.called("Claude", "claude-haiku-5-5")
