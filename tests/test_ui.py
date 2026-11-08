"""Streamlit 화면 흐름 (AI 호출만 가짜로 바꾼 앱 사본을 AppTest로 조작)."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP

CALLERS_LINE = 'CALLERS = {"anthropic": call_anthropic, "gemini": call_gemini, "openai": call_openai_compat}'
MOCK = '''
async def _mock(p, model, system, text, images, search=False, on_text=None, effort=None):
    if system == CLARIFY_SYSTEM:
        out = ('{"ask": true, "questions": [{"q": "어느 지역 기준인가요?", "options": ["서울", "부산"], "multi": false},'
               ' {"q": "관심 장르는?", "options": ["액션", "로맨스", "애니"], "multi": true}]}'
               if "되묻기테스트" in text else '{"ask": false, "questions": []}')
    else:
        out = f"[{model}] " + ("CLARIFIED " if "<clarification>" in text else "PLAIN ") + "답변"
    if on_text:
        on_text(out)
    if p["kind"] == "anthropic":
        return out, {"usage": {"input": 10000, "output": 2000, "searches": 1 if search else 0}}
    return out if p["kind"] == "openai" else (out, {})
CALLERS = {"anthropic": _mock, "gemini": _mock, "openai": _mock}
def model_ids(provider):
    return {"Claude": ["claude-sonnet-5-5", "claude-haiku-5-5"], "Gemini": ["gemini-3.8-flash", "gemini-3.8-flash-lite"],
            "Groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]}.get(provider)
'''
KEYS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")


@pytest.fixture
def mock_app(tmp_path):
    src = APP.read_text(encoding="utf-8")
    assert CALLERS_LINE in src
    path = tmp_path / "app.py"                # 대화 기록은 tmp_path/data 에 저장됨
    path.write_text(src.replace(CALLERS_LINE, MOCK), encoding="utf-8")
    return path


def start(path, **secrets):
    at = AppTest.from_file(str(path), default_timeout=60)
    for k in KEYS:
        at.secrets[k] = "dummy"
    for k, v in secrets.items():
        at.secrets[k] = v
    return at.run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


# ── 비밀번호 ──
def test_missing_password_blocks_app(mock_app):
    at = start(mock_app)
    assert not at.exception
    assert any("접속 비밀번호가 설정되지 않아" in e.value for e in at.error)
    assert len(at.chat_input) == 0


def test_password_gate_and_lockout(mock_app):
    at = start(mock_app, APP_PASSWORD="secret-pw")
    assert len(at.chat_input) == 0
    for _ in range(5):
        at.text_input[0].set_value("").run()
        at.text_input[0].set_value("wrong").run()
    at.run()                                  # 잠금 안내는 다음 화면 갱신부터 표시
    assert any("잠겼습니다" in e.value for e in at.error) and len(at.text_input) == 0
    at = start(mock_app, APP_PASSWORD="secret-pw")       # 새 세션
    at.text_input[0].set_value("secret-pw").run()
    assert len(at.chat_input) == 1


# ── 기본 배치·비용 ──
def test_default_roles_and_cost(mock_app, tmp_path):
    at = start(mock_app, ALLOW_NO_PASSWORD=True)
    assert not at.exception
    assert at.selectbox(key="r_a").value == "Gemini" and at.selectbox(key="r_b").value == "Claude"
    assert any("Claude(Sonnet 5.5) 주재 · ⚖️ Gemini vs Claude(Haiku 5.5)" in c.value for c in at.sidebar.caption)
    engines = [c.value for c in at.sidebar.caption if c.value.startswith("🧠")]
    assert engines == ["🧠 Sonnet 5.5", "🧠 Gemini 3.8 Flash", "🧠 Haiku 5.5"]   # 허브·A·B 모델 표시
    at.toggle(key="r_clarify").set_value(False).run()
    at.chat_input[0].set_value("1+1은?").run()
    assert not at.exception
    turn = at.session_state["turns"][0]
    assert turn["result"]["firsts"][1]["model"] == "claude-haiku-5-5"
    expected = 2 * (10000 * 2 + 2000 * 10) / 1e6 + 0.01 + 2 * (10000 * 0.1 + 2000 * 0.5) / 1e6
    assert turn["result"]["cost"] == pytest.approx(expected) and turn.get("at")
    assert any("💰 Claude 약 $" in c.value for c in at.caption)
    idx = json.loads((tmp_path / "data" / "index.json").read_text(encoding="utf-8"))
    assert next(iter(idx.values()))["cost"][turn["at"][:7]] == pytest.approx(expected)


# ── 되묻기 ──
def test_clarify_answer_flows_into_debate(mock_app):
    at = start(mock_app, ALLOW_NO_PASSWORD=True)
    at.chat_input[0].set_value("되묻기테스트 다음주 인기 영화는?").run()
    assert "pending" in at.session_state and len(at.radio) == 1 and len(at.multiselect) == 1
    assert len(at.session_state["turns"]) == 0
    at.radio[0].set_value("서울")
    at.multiselect[0].set_value(["액션", "애니"])
    next(w for w in at.text_input if w.label.startswith("기타")).set_value("가족과 볼 영화")
    button(at, "답하고 토론 시작").click().run()
    assert not at.exception
    t = at.session_state["turns"][0]
    assert [a["a"] for a in t["clarify"]] == ["서울", "액션, 애니", "가족과 볼 영화"]
    r = t["result"]
    assert all("CLARIFIED" in x["text"] for x in r["firsts"]) and "CLARIFIED" in r["final"]["text"]
    assert "pending" not in at.session_state
    assert any("어느 지역 기준인가요? → **서울**" in c.value for c in at.caption)
    assert not [m for m in at.main.get("chat_message") if len(m.children) == 0]      # 빈 말풍선 없음


def test_clarify_skip_and_unambiguous_and_toggle(mock_app):
    at = start(mock_app, ALLOW_NO_PASSWORD=True)
    at.chat_input[0].set_value("되묻기테스트 영화").run()
    button(at, "건너뛰고 그대로 진행").click().run()
    t = at.session_state["turns"][0]
    assert t["clarify"] == [] and "PLAIN" in t["result"]["firsts"][0]["text"]

    at = start(mock_app, ALLOW_NO_PASSWORD=True)
    at.chat_input[0].set_value("1+1은?").run()
    assert "pending" not in at.session_state and len(at.session_state["turns"]) == 1

    at = start(mock_app, ALLOW_NO_PASSWORD=True)
    at.toggle(key="r_clarify").set_value(False).run()
    at.chat_input[0].set_value("되묻기테스트 영화").run()
    assert "pending" not in at.session_state and len(at.session_state["turns"]) == 1


def test_new_message_while_pending_discards_form(mock_app):
    at = start(mock_app, ALLOW_NO_PASSWORD=True)
    at.chat_input[0].set_value("되묻기테스트 A").run()
    at.chat_input[0].set_value("1+1은?").run()
    assert "pending" not in at.session_state and at.session_state["turns"][0]["q"] == "1+1은?"
