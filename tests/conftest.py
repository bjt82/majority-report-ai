"""시험 공통 준비물.

app.py는 Streamlit 화면 코드까지 한 파일에 들어 있어서, 로직 시험은 화면 부분('5) UI' 앞)만 불러와 씁니다.
실제 AI API는 부르지 않고 FakeAI가 대신 답하거나 오류를 냅니다.
"""
import asyncio
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"
UI_MARK = "# ───────────────────────── 5) UI"
KEYS = {"ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"}
# 각 AI의 'API 모델 목록'을 흉내 냄 (낮은 모델 목록은 여기 있는 것만 쓰임)
MODELS = {
    "Claude": ["claude-sonnet-5-5", "claude-haiku-5-5"],
    "Gemini": ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.5-flash", "gemini-embedding-001"],
    "Groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
    "Mistral": ["mistral-small-latest", "ministral-8b-latest"],
}
DAILY = "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"


class Core(dict):
    """app.py 로직 부분의 전역 이름공간. core.ask처럼 점으로도 꺼낼 수 있음."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def load_core() -> Core:
    src = APP.read_text(encoding="utf-8").split(UI_MARK)[0]
    ns = Core(__file__=str(APP), __name__="app_core")
    exec(compile(src, str(APP), "exec"), ns)
    ns["secret"] = lambda k, d=None: "test-key" if k in KEYS else d
    ns["model_ids"] = lambda p: MODELS.get(p)
    ns["RETRY_WAIT_SEC"] = 0.2
    ns["model_list_failures"]().clear()
    ns["missing_models"]().clear()
    return ns


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
            return out if kind == "openai" else (out, {})
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
