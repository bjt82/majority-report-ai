"""시험 공통 준비물.

핵심 로직은 majority/ 패키지에 있고, 시험마다 모듈을 새로 불러와(reload) 앞 시험에서 바꾼 값이 남지 않게 합니다.
실제 AI API는 부르지 않고 FakeAI가 대신 답하거나 오류를 냅니다.
"""
import asyncio
import importlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
sys.path.insert(0, str(ROOT))
MODULES = ["majority.settings", "majority.models", "majority.llm",
           "majority.prompts", "majority.pipeline", "majority.store"]   # 의존 순서
KEYS = {"ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"}
# 각 AI의 'API 모델 목록'을 흉내 냄 (낮은 모델 목록은 여기 있는 것만 쓰임)
MODELS = {
    "Claude": ["claude-sonnet-5-5", "claude-haiku-5-5"],
    "Gemini": ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.5-flash", "gemini-embedding-001"],
    "Groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
    "Mistral": ["mistral-small-latest", "ministral-8b-latest"],
}
DAILY = "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"


class Core:
    """majority 모듈들을 한 이름공간처럼 다룸. core.ask / core["ask"]로 꺼내고,
    core["secret"] = 가짜 처럼 바꾸면 그 이름을 가져다 쓰는 모든 모듈에 함께 적용됨 (별 import로 이름이 복사되므로)."""

    def __init__(self, mods):
        object.__setattr__(self, "_mods", mods)

    def _find(self, name):
        for m in reversed(self._mods):
            if hasattr(m, name):
                return getattr(m, name)
        raise KeyError(name)

    def __getitem__(self, name):
        return self._find(name)

    def __getattr__(self, name):
        try:
            return self._find(name)
        except KeyError as e:
            raise AttributeError(name) from e

    def __setitem__(self, name, value):
        old = self._find(name)
        for m in self._mods:
            if getattr(m, name, None) is old:
                setattr(m, name, value)


def load_core(fake_model_list=True) -> Core:
    mods = [importlib.reload(importlib.import_module(n)) for n in MODULES]
    core = Core(mods)
    core["secret"] = lambda k, d=None: "test-key" if k in KEYS else d
    if fake_model_list:
        core["model_ids"] = lambda p: MODELS.get(p)
    core["RETRY_WAIT_SEC"] = 0.2
    core.model_list_failures().clear()
    core.missing_models().clear()
    return core


class FakeAI:
    """모든 AI 호출을 가로채는 가짜. behavior[(AI, 모델)]에 오류 문구나 async 함수를 넣어 상황을 만듦."""

    def __init__(self, core: Core):
        self.core, self.behavior, self.calls, self.efforts = core, {}, [], []
        self.usage = {"input": 1000, "output": 500}
        for kind in ("anthropic", "gemini", "openai"):
            core["CALLERS"][kind] = self._make(kind)

    def reply(self, name, model, system, text):
        return f"answer from {name}/{model}"

    def _make(self, kind):
        async def caller(p, model, system, text, images, search=False, on_text=None, effort=None):
            name = next(n for n, q in self.core["PROVIDERS"].items() if q is p)
            self.calls.append((name, model))
            self.efforts.append((system, model, effort))
            b = self.behavior.get((name, model))
            if isinstance(b, str):
                raise Exception(b)
            out = await b() if b else self.reply(name, model, system, text)
            if on_text:
                on_text(out)
            if kind == "anthropic":
                return out, {"usage": {**self.usage, "searches": 1 if search else 0},
                             "truncated": getattr(self, "truncate", None) == (name, model)}
            return out if kind == "openai" else (out, {"usage": {**self.usage, "cached_input": 0, "searches": 0}})
        return caller

    def called(self, name, model):
        return (name, model) in self.calls


@pytest.fixture
def core():
    return load_core()


@pytest.fixture
def ai(core):
    return FakeAI(core)


@pytest.fixture
def run(core, ai):
    """run(roles, models, cooldowns, **kw) → 파이프라인 결과. 기본은 허브 Claude · Gemini vs Groq."""
    def _run(roles=None, models=None, cooldowns=None, prompt="테스트 질문", **kw):
        roles = roles or {"hub": "Claude", "a": "Gemini", "b": "Groq"}
        models = models or {"hub": "claude-sonnet-5-5", "a": "gemini-3.8-flash", "b": "openai/gpt-oss-120b"}
        ai.calls.clear()
        ui = core.UserInput(prompt=prompt)
        cd = {} if cooldowns is None else cooldowns
        return asyncio.run(core.run_pipeline(ui, roles, models, lambda m: None, cd, **kw))
    return _run
