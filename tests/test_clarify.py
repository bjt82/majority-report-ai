"""되묻기: 허브 답(JSON) 해석, 가벼운 모델 우선, 답을 질문에 붙이기."""
import asyncio
import json


def test_parse_clarify(core):
    raw = ('```json\n{"ask": true, "questions": ['
           '{"q": "어느 지역 기준인가요?", "options": ["서울", "부산", "서울", "기타"], "multi": false},'
           '{"q": "관심 장르?", "options": ["액션", "로맨스", "애니"], "multi": true},'
           '{"q": "선택지 하나", "options": ["a"]}]}\n```')
    r = core.parse_clarify(raw)
    assert len(r) == 2 and r[0]["options"] == ["서울", "부산"] and r[1]["multi"]
    assert core.parse_clarify('{"ask": false, "questions": []}') == []
    assert core.parse_clarify("잘 모르겠습니다") == [] and core.parse_clarify("{broken") == []
    many = json.dumps({"ask": True, "questions": [{"q": f"q{i}", "options": list("abcdef")} for i in range(5)]})
    r = core.parse_clarify(many)
    assert len(r) == 3 and len(r[0]["options"]) == 5


def test_hub_clarify_prefers_light_model_and_reports_cost(core, ai):
    ai.reply = lambda *a: '{"ask": true, "questions": [{"q": "기간은?", "options": ["이번 주", "다음 주"]}]}'
    ui = core.UserInput(prompt="다음주 영화 추천")
    cd = {}
    qs, cost = asyncio.run(core.hub_clarify(ui, "Claude", "claude-sonnet-5-5", cd))
    assert ai.calls == [("Claude", "claude-haiku-5-5")] and qs[0]["q"] == "기간은?"
    assert cost == core.cost_of("claude-haiku-5-5", {**ai.usage, "searches": 0})

    ai.calls.clear()
    core["RETRY_WAIT_SEC"] = 20               # 실제 기본값: 대기 시간을 모르면 재시도 대신 바로 다음 모델로
    ai.behavior[("Claude", "claude-haiku-5-5")] = "429 rate limit"
    qs, _ = asyncio.run(core.hub_clarify(ui, "Claude", "claude-sonnet-5-5", cd))
    assert ai.calls == [("Claude", "claude-haiku-5-5"), ("Claude", "claude-sonnet-5-5")] and qs
    assert "Claude/claude-haiku-5-5" in cd

    ai.calls.clear()
    asyncio.run(core.hub_clarify(ui, "Claude", "claude-sonnet-5-5", cd))
    assert ai.calls == [("Claude", "claude-sonnet-5-5")]          # 제외 중인 Haiku는 건너뜀


def test_with_clarification(core):
    w = core.with_clarification("영화 추천", [{"q": "기간은?", "a": "다음 주"}])
    assert "<clarification>" in w and "기간은? → 다음 주" in w
    assert core.with_clarification("x", []) == "x"
