"""중단: 진행판을 그리다 Streamlit 중단 신호가 오면 진행 중인 토론(AI 호출)을 즉시 취소."""
import asyncio
import time

import pytest


class StopSignal(BaseException):
    """Streamlit의 StopException/RerunException처럼 BaseException 계열."""


def test_stop_signal_cancels_job_immediately(core):
    state = {"renders": 0, "cancelled": False, "finished": False}

    async def slow_job():
        try:
            await asyncio.sleep(10)               # 오래 걸리는 AI 호출 흉내
            state["finished"] = True
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    def render():
        state["renders"] += 1
        if state["renders"] == 3:                 # 약 2초 뒤 사용자가 중단
            raise StopSignal()

    t = time.time()
    with pytest.raises(StopSignal):
        asyncio.run(core.run_with_ticker(slow_job(), render, interval=0.05))
    assert time.time() - t < 1
    assert state["cancelled"] and not state["finished"]


def test_normal_completion_returns_result_and_errors_propagate(core):
    async def job():
        await asyncio.sleep(0.12)
        return {"ok": 1}
    renders = []
    assert asyncio.run(core.run_with_ticker(job(), lambda: renders.append(1), interval=0.05)) == {"ok": 1}
    assert len(renders) >= 2                      # 기다리는 동안 진행판을 계속 그림

    async def failing():
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        asyncio.run(core.run_with_ticker(failing(), lambda: None, interval=0.05))


def test_pipeline_is_cancelled_mid_call(core, ai):
    """실제 파이프라인: 판단자 호출 중에 중단하면 이후 단계(반론·판정)는 시작되지 않음."""
    async def slow():
        await asyncio.sleep(5)
        return "late", {}
    ai.behavior[("Gemini", "gemini-3.8-flash")] = slow
    ticks = {"n": 0}

    def render():
        ticks["n"] += 1
        if ticks["n"] == 4:
            raise StopSignal()
    ui = core.UserInput(prompt="오타 질문")
    job = core.run_pipeline(ui, {"hub": "Claude", "a": "Gemini", "b": "Groq"},
                            {"hub": "claude-sonnet-5-5", "a": "gemini-3.8-flash", "b": "openai/gpt-oss-120b"},
                            lambda m: None, {})
    t = time.time()
    with pytest.raises(StopSignal):
        asyncio.run(core.run_with_ticker(job, render, interval=0.05))
    assert time.time() - t < 1
    assert ("Claude", "claude-sonnet-5-5") in ai.calls             # 정리는 했고
    assert ai.calls.count(("Claude", "claude-sonnet-5-5")) == 1    # 판정은 시작하지 않음
