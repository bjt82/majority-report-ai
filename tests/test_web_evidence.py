"""검색 발췌를 넘어 본문 조회: 실제 API를 부르지 않는 SDK 회귀 시험."""
import asyncio
import json

import pytest

from test_effort import message, sdk_core


@pytest.mark.parametrize("current,tomorrow,yesterday", [
    ("2026-11-08", "2026-11-09", "2026-11-07"),
    ("2026-12-31", "2027-01-01", "2026-12-30"),
    ("2028-03-01", "2028-03-02", "2028-02-29"),
])
def test_relative_dates_follow_app_date(core, current, tomorrow, yesterday):
    core["today"] = lambda: current
    text = core.UserInput(prompt="내일 배정고등학교 석식은?").base_text()
    assert f"오늘 날짜: {current}" in text
    assert f"내일: {tomorrow}" in text
    assert f"어제: {yesterday}" in text


def fetched(url="https://school.example/menu", title="11월 식단표"):
    return {"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_fetch",
            "content": {"type": "web_fetch_result", "url": url,
                        "retrieved_at": "2026-11-08T14:00:00Z",
                        "content": {"type": "document", "title": title,
                                    "source": {"type": "text", "media_type": "text/plain",
                                               "data": "2026-11-09 석식: 메뉴"}}}}


def test_fetch_request_and_sources_reach_pipeline_result(core):
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        response = message("해당 날짜의 식단을 본문에서 확인")
        response["content"] = [
            {"type": "server_tool_use", "id": "srvtoolu_fetch", "name": "web_fetch",
             "input": {"url": "https://school.example/menu"}},
            fetched(), *response["content"],
        ]
        response["usage"]["server_tool_use"] = {"web_fetch_requests": 1, "web_search_requests": 0}
        return httpx.Response(200, json=response)

    httpx = sdk_core(core, handler)
    result = asyncio.run(core.ask("허브 정리", "Claude", "claude-sonnet-5-5", "sys", "q", [], search=True))
    assert result["ok"], result
    assert result["fetches"] == 1 and result["searches"] == 0
    assert result["sources"] == [{"title": "11월 식단표", "url": "https://school.example/menu"}]
    tools = {t["name"]: t for t in bodies[0]["tools"]}
    assert tools["web_fetch"]["use_cache"] is False
    assert tools["web_fetch"]["max_content_tokens"] == core.FETCH_MAX_CONTENT_TOKENS
    assert tools["web_fetch"]["max_uses"] == core.FETCH_MAX_USES
    assert tools["web_search"]["max_uses"] == core.SEARCH_MAX_USES
    assert result["usage"]["searches"] == 0  # 본문 조회를 유료 검색 횟수로 세지 않음


@pytest.mark.parametrize("rejected", ["web_fetch", "web_search"])
def test_rejecting_one_tool_preserves_the_other(core, rejected):
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if any(t["name"] == rejected for t in body.get("tools", [])):
            return httpx.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": f"{rejected} is not enabled"}})
        return httpx.Response(200, json=message("도구 제한을 반영한 답"))

    httpx = sdk_core(core, handler)
    out, meta = asyncio.run(core.call_anthropic(core.PROVIDERS["Claude"], "claude-sonnet-5-5",
                                                "sys", "q", [], search=True))
    assert out == "도구 제한을 반영한 답"
    assert len(bodies) == 3
    surviving = "web_search" if rejected == "web_fetch" else "web_fetch"
    assert [t["name"] for t in bodies[-1]["tools"]] == [surviving, "browser_read"]
    assert meta["search_note"]
    if rejected == "web_fetch":
        legacy = next(t for t in bodies[1]["tools"] if t["name"] == "web_fetch")
        assert legacy["type"] == "web_fetch_20250910"
        assert "use_cache" not in legacy and "allowed_callers" not in legacy


def test_fetch_error_is_reported_without_inventing_a_source(core):
    def handler(req):
        response = message("본문을 확인하지 못했습니다")
        response["content"].insert(0, {"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_fetch",
                                       "content": {"type": "web_fetch_tool_result_error",
                                                   "error_code": "url_not_accessible"}})
        return httpx.Response(200, json=response)

    httpx = sdk_core(core, handler)
    _, meta = asyncio.run(core.call_anthropic(core.PROVIDERS["Claude"], "claude-sonnet-5-5",
                                             "sys", "q", [], search=True))
    assert "url_not_accessible" in meta["search_note"]
    assert meta["sources"] == []


def test_no_network_tools_when_search_is_disabled(core):
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json=message("답"))

    httpx = sdk_core(core, handler)
    asyncio.run(core.call_anthropic(core.PROVIDERS["Claude"], "claude-sonnet-5-5", "sys", "q", []))
    assert "tools" not in bodies[0]
