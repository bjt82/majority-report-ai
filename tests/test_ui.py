"""Streamlit 화면 흐름 (AI 호출만 가짜로 바꾼 앱 사본을 AppTest로 조작)."""
import importlib
import json
import time

import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP, MODULES, ROOT

MOCK = '''
import sys
sys.path.insert(0, {root!r})
import majority.llm as _llm
import majority.models as _models
from majority.prompts import CLARIFY_SYSTEM


_llm.SEEN_IMAGES = getattr(_llm, "SEEN_IMAGES", [])     # 호출마다 전달된 이미지 수 (시험에서 확인)


async def _mock(p, model, system, text, images, search=False, on_text=None, effort=None):
    _llm.SEEN_IMAGES.append(len(images))
    if system == CLARIFY_SYSTEM:
        out = ('{{"ask": true, "questions": [{{"q": "어느 지역 기준인가요?", "options": ["서울", "부산"], "multi": false}},'
               ' {{"q": "관심 장르는?", "options": ["액션", "로맨스", "애니"], "multi": true}}]}}'
               if "되묻기테스트" in text else '{{"ask": false, "questions": []}}')
    else:
        out = f"[{{model}}] " + ("CLARIFIED " if "<clarification>" in text else "PLAIN ") + "답변"
    if on_text:
        on_text(out)
    if p["kind"] == "anthropic":
        return out, {{"usage": {{"input": 10000, "output": 2000, "searches": 1 if search else 0}}}}
    return out if p["kind"] == "openai" else (out, {{"usage": {{"input": 10000, "output": 2000, "cached_input": 0, "searches": 0}}}})


def _model_ids(provider):
    return {{"Claude": ["claude-sonnet-5-5", "claude-haiku-5-5"], "Gemini": ["gemini-3.8-flash", "gemini-3.8-flash-lite"],
            "Groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]}}.get(provider)


_llm.CALLERS.update({{"anthropic": _mock, "gemini": _mock, "openai": _mock}})
_models.model_ids = _model_ids
exec(compile(open({app!r}, encoding="utf-8").read(), {app!r}, "exec"))
'''
KEYS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")


@pytest.fixture
def mock_app(tmp_path):
    """AI 호출만 가짜로 바꾸고 실제 app.py를 그대로 실행하는 시험용 진입 파일."""
    for name in MODULES:                      # 로직 시험이 바꿔 둔 가짜(secret 등)가 남지 않게 새로 불러옴
        importlib.reload(importlib.import_module(name))
    path = tmp_path / "app_under_test.py"
    path.write_text(MOCK.format(root=str(ROOT), app=str(APP)), encoding="utf-8")
    return path


def start(path, **secrets):
    at = AppTest.from_file(str(path), default_timeout=60)
    for k in KEYS:
        at.secrets[k] = "dummy"
    at.secrets["LOCAL_DATA_DIR"] = str(path.parent / "data")      # 대화 기록은 시험 임시 폴더에
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
    at.toggle(key="r_auto_b").set_value(False).run()  # 수동 Haiku 배치의 비용 회귀 시험
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
    claude_expected = expected
    gemini_expected = 2 * (10000 * .75 + 2000 * 3.75) / 1e6
    expected += gemini_expected
    assert turn["result"]["cost"] == pytest.approx(expected) and turn.get("at")
    assert any("💰 Claude 약 $" in c.value for c in at.caption)
    index = tmp_path / "data" / "index.json"          # 저장은 뒤에서 하므로 잠깐 기다림
    for _ in range(50):
        if index.exists():
            break
        time.sleep(0.1)
    idx = json.loads(index.read_text(encoding="utf-8"))
    assert next(iter(idx.values()))["cost"][turn["at"][:7]] == pytest.approx(expected)
    assert turn["result"]["costs"] == pytest.approx({"Claude": claude_expected, "Gemini": gemini_expected})
    entry = next(iter(idx.values()))
    assert entry["cost_by_provider"]["Gemini"][turn["at"][:7]] == pytest.approx(gemini_expected)
    assert any("이번 달 Gemini" in p.proto.text for p in at.sidebar.get("progress"))
    assert any("Gemini 약 $0.0300" in c.value for c in at.caption)


def test_auto_judge_b_routes_short_and_long_input(mock_app):
    at = start(mock_app, ALLOW_NO_PASSWORD=True)
    assert at.toggle(key="r_auto_b").value and at.selectbox(key="r_b").disabled
    assert any("자동 판단 B" in c.value for c in at.sidebar.caption)
    at.toggle(key="r_clarify").set_value(False).run()
    at.chat_input[0].set_value("짧은 질문").run()
    assert not at.exception
    short = at.session_state["turns"][-1]["result"]
    assert short["firsts"][1]["name"] == "Groq" and short["judge_b_route"]["mode"] == "free_first"
    at.session_state["conv_docs"] = "긴 자료 " * 1500
    at.chat_input[0].set_value("이 문서의 세부 내용을 검토해 줘").run()
    assert not at.exception
    long = at.session_state["turns"][-1]["result"]
    assert long["firsts"][1]["model"] == "claude-haiku-5-5" and long["judge_b_route"]["mode"] == "full"
    at.toggle(key="r_auto_b").set_value(False).run()
    assert not at.selectbox(key="r_b").disabled


def test_pinned_input_does_not_force_page_autoscroll(mock_app):
    """하단 입력창은 유지하되 펼침·긴 답변의 높이 변화가 자동 스크롤을 켜지 않음."""
    at = start(mock_app, ALLOW_NO_PASSWORD=True)
    assert not at.exception and len(at.chat_input) == 1
    assert not at.chat_input[0].proto.is_auto_positioned_at_bottom
    at.toggle(key="r_clarify").set_value(False).run()
    at.chat_input[0].set_value("스크롤 검증").run()
    assert not at.exception
    result = at.session_state["turns"][0]["result"]
    result["final"]["text"] = "긴 답변 단락\n\n" * 200 + "최종 답변 끝"
    result["brief"]["text"] = "긴 토론 단락\n\n" * 200 + "토론 과정 끝"
    at.run()
    assert not at.exception and not at.chat_input[0].proto.is_auto_positioned_at_bottom
    assert any("최종 답변 끝" in m.value for m in at.markdown)
    assert any("토론 과정 끝" in m.value for m in at.markdown)


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


def test_background_save_failure_is_reported(mock_app):
    """저장소(GitHub)에 저장이 실패하면 답변은 그대로 두고 다음 화면 사이드바에 알림."""
    at = start(mock_app, ALLOW_NO_PASSWORD=True, GITHUB_TOKEN="bad-token", DATA_REPO="nobody/does-not-exist-xyz")
    at.toggle(key="r_clarify").set_value(False).run()
    at.chat_input[0].set_value("1+1은?").run()
    assert not at.exception and len(at.session_state["turns"]) == 1
    for _ in range(40):                               # 뒤에서 저장 시도가 끝나길 기다림
        at.run()
        if any("대화 기록 저장 실패" in w.value for w in at.sidebar.warning):
            break
        time.sleep(0.25)
    assert any("대화 기록 저장 실패" in w.value for w in at.sidebar.warning)


# ── 첨부 관리·알림 구분 ──
def seen_images():
    import majority.llm as llm
    return llm.SEEN_IMAGES


def with_earlier_image(at):
    from majority.prompts import Attachment
    at.session_state["conv_images"] = [Attachment(b"fake-png", "image/png")]
    at.session_state["conv_files"] = ["image.png"]
    return at


def test_earlier_image_not_resent_by_default(mock_app):
    at = with_earlier_image(start(mock_app, ALLOW_NO_PASSWORD=True))
    at.toggle(key="r_clarify").set_value(False).run()
    seen_images().clear()
    at.chat_input[0].set_value("1+1은?").run()
    assert not at.exception
    assert seen_images() and all(n == 0 for n in seen_images())          # 이전 이미지는 보내지 않음
    route = at.session_state["turns"][0]["result"]["judge_b_route"]
    assert route["reason"] == "짧은 텍스트"                             # 그래서 판단 B도 무료 AI 우선
    # 일상 안내(🧭)는 노란 경고가 아니라 회색 글씨
    assert not any(w.value.startswith("🧭") for w in at.warning)
    assert any(c.value.startswith("🧭") for c in at.caption)


def test_keep_images_toggle_resends(mock_app):
    at = with_earlier_image(start(mock_app, ALLOW_NO_PASSWORD=True))
    at.toggle(key="r_clarify").set_value(False).run()
    at.toggle(key="keep_images").set_value(True).run()
    seen_images().clear()
    at.chat_input[0].set_value("이 그림 다시 봐줘").run()
    assert max(seen_images()) == 1
    assert at.session_state["turns"][0]["result"]["judge_b_route"]["reason"] == "이미지 포함"


def test_clear_attachments(mock_app):
    at = with_earlier_image(start(mock_app, ALLOW_NO_PASSWORD=True))
    at.session_state["conv_docs"] = "[파일: a.txt] 내용"
    at.run()
    button(at, "첨부 해제").click().run()
    assert at.session_state["conv_docs"] == "" and at.session_state["conv_images"] == []
    assert at.session_state["conv_files"] == []
