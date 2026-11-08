"""렌더링·프록시·도구 왕복 검증. 실제 AI API나 외부 사이트는 사용하지 않음."""
import asyncio
import base64
import json
import os
import socket

import pytest

from majority import browser
from test_effort import message, sdk_core


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://user:pass@example.com/",
                                "https://example.com:8080/", "javascript:alert(1)"])
def test_only_public_web_url_forms(url):
    with pytest.raises(ValueError):
        browser.public_target(url)


@pytest.mark.parametrize("addresses", [["127.0.0.1"], ["169.254.169.254"], ["10.0.0.1"],
                                     ["::1"], ["::ffff:127.0.0.1"], ["93.184.216.34", "192.168.1.1"]])
def test_dns_rejects_private_and_mixed_answers(monkeypatch, addresses):
    async def run():
        async def resolve(*args, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in addresses]
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
        with pytest.raises(ValueError):
            await browser.public_addresses("public-looking.example", 443)
    asyncio.run(run())


def test_connection_uses_validated_ip_not_hostname(monkeypatch):
    calls = []

    async def addresses(host, port):
        return ["93.184.216.34"]

    async def connect(host, port):
        calls.append((host, port))
        return "reader", "writer"

    monkeypatch.setattr(browser, "public_addresses", addresses)
    monkeypatch.setattr(asyncio, "open_connection", connect)
    assert asyncio.run(browser.public_connection("public.example", 443)) == ("reader", "writer")
    assert calls == [("93.184.216.34", 443)]


def test_proxy_blocks_internal_connect_and_write_requests():
    async def run():
        for request in [b"CONNECT 127.0.0.1:443 HTTP/1.1\r\n\r\n",
                        b"POST http://example.com/ HTTP/1.1\r\n\r\n"]:
            async with browser.public_proxy() as url:
                from urllib.parse import urlsplit
                target = urlsplit(url)
                reader, writer = await asyncio.open_connection(target.hostname, target.port)
                writer.write(request)
                await writer.drain()
                assert (await reader.read()).startswith(b"HTTP/1.1 403")
                writer.close()
    asyncio.run(run())


def test_browser_tool_result_goes_back_to_model_without_persisting_image(core):
    bodies = []

    async def read(arguments):
        assert arguments["url"] == "https://public.example/"
        return {"ok": True, "url": arguments["url"], "title": "Rendered", "method": "browser",
                "retrieved_at": "2026-11-09T00:00:00Z", "text": "Fresh after JS",
                "screenshot": base64.b64encode(b"fake image").decode()}

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if len(bodies) == 1:
            result = message("화면을 열어 확인합니다", stop="tool_use")
            result["content"].append({"type": "tool_use", "id": "tool_browser", "name": "browser_read",
                                      "input": {"url": "https://public.example/"}})
            return httpx.Response(200, json=result)
        return httpx.Response(200, json=message("브라우저에서 최신 내용 확인"))

    core["browser_read"] = read
    httpx = sdk_core(core, handler)
    result = asyncio.run(core.ask("허브 정리", "Claude", "claude-sonnet-5-5", "sys", "q", [], search=True))
    assert result["ok"], result
    tool_result = bodies[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tool_browser"
    assert "Fresh after JS" in tool_result["content"][0]["text"]
    assert tool_result["content"][1]["type"] == "image"
    assert result["sources"] == [{"title": "Rendered", "url": "https://public.example/"}]
    assert result["browser_visits"][0]["method"] == "browser"
    assert "screenshot" not in json.dumps(result)
    assert result["usage"]["input"] == 20  # 이어진 두 SDK 응답의 사용량 모두 반영


def test_browser_limit_returns_tool_error_without_extra_visits(core):
    bodies = []
    calls = []

    async def read(arguments):
        calls.append(arguments)
        return {"ok": False, "error": "not_accessible", "message": "접근 실패"}

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if len(bodies) <= 4:
            result = message("다시 확인", stop="tool_use")
            result["content"] = [{"type": "tool_use", "id": f"browser_{len(bodies)}", "name": "browser_read",
                                  "input": {"url": "https://public.example/"}}]
        else:
            result = message("확인하지 못했습니다")
        return httpx.Response(200, json=result)

    core["browser_read"] = read
    httpx = sdk_core(core, handler)
    _, meta = asyncio.run(core.call_anthropic(core.PROVIDERS["Claude"], "claude-sonnet-5-5",
                                             "sys", "q", [], search=True))
    assert len(calls) == browser.BROWSER_MAX_USES
    assert bodies[-1]["messages"][-1]["content"][0]["is_error"] is True
    assert "한도" in meta["search_note"]
    assert meta["sources"] == []


def test_rejected_browser_preserves_search_and_fetch(core):
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if any(t["name"] == "browser_read" for t in body.get("tools", [])):
            return httpx.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": "browser_read tool is not supported"}})
        return httpx.Response(200, json=message("검색과 본문으로 확인"))

    httpx = sdk_core(core, handler)
    _, meta = asyncio.run(core.call_anthropic(core.PROVIDERS["Claude"], "claude-sonnet-5-5",
                                             "sys", "q", [], search=True))
    assert [t["name"] for t in bodies[-1]["tools"]] == ["web_search", "web_fetch"]
    assert len(bodies) == 2 and "브라우저 도구" in meta["search_note"]


def test_browser_can_be_disabled_without_disabling_search(core):
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json=message("답"))

    original_secret = core.secret
    core["secret"] = lambda key, default=None: False if key == "BROWSER_READING" else original_secret(key, default)
    httpx = sdk_core(core, handler)
    asyncio.run(core.call_anthropic(core.PROVIDERS["Claude"], "claude-sonnet-5-5", "sys", "q", [], search=True))
    assert [t["name"] for t in bodies[0]["tools"]] == ["web_search", "web_fetch"]


@pytest.mark.skipif(os.environ.get("BROWSER_TESTS") != "1", reason="설치된 Chromium으로 통합 시험: BROWSER_TESTS=1")
def test_real_chromium_renders_js_and_interacts(monkeypatch):
    async def run():
        requests = []

        async def serve(reader, writer):
            header = await reader.readuntil(b"\r\n\r\n")
            method, path, _ = header.decode().split("\r\n")[0].split(" ")
            requests.append((method, path))
            if path == "/data":
                body = b'{"value":"Fresh JavaScript content"}'
            elif path == "/other":
                body = b"<html><title>Other</title><body>Linked page text</body></html>"
            else:
                body = b"""<html><title>Dynamic fixture</title><body>
                  <main id="result">Loading...</main>
                  <button role="tab" onclick="document.querySelector('#result').textContent='Tab content'">Details</button>
                  <select aria-label="Period" onchange="document.querySelector('#result').textContent=this.value">
                    <option value="today">Today</option><option value="tomorrow">Tomorrow</option></select>
                  <input type="date" onchange="document.querySelector('#result').textContent=this.value">
                  <a href="/other">More</a>
                  <script>setTimeout(async()=>{
                    let data=await(await fetch('/data')).json();
                    document.querySelector('#result').textContent=data.value;
                    fetch('/write',{method:'POST'}).catch(()=>{});
                  },150);</script></body></html>"""
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: "
                         + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def fixture_connection(host, remote_port):
            assert host == "fixture.test"
            return await asyncio.open_connection("127.0.0.1", port)

        monkeypatch.setattr(browser, "public_connection", fixture_connection)
        try:
            initial = await browser.render_page("http://fixture.test/")
            assert "Fresh JavaScript content" in initial["text"]
            assert initial["controls"]["tabs"] == ["Details"]
            assert base64.b64decode(initial["screenshot"]).startswith(b"\xff\xd8")
            tab = await browser.render_page("http://fixture.test/", [{"kind": "tab", "name": "Details"}])
            assert "Tab content" in tab["text"]
            selected = await browser.render_page("http://fixture.test/", [
                {"kind": "select", "index": 0, "value": "tomorrow"},
                {"kind": "wait", "text": "tomorrow"},
            ])
            assert "tomorrow" in selected["text"]
            dated = await browser.render_page("http://fixture.test/", [{"kind": "date", "index": 0,
                                                                       "value": "2026-11-09"}])
            assert "2026-11-09" in dated["text"]
            linked = await browser.render_page("http://fixture.test/", [{"kind": "link", "name": "More"}])
            assert linked["url"] == "http://fixture.test/other" and "Linked page text" in linked["text"]
            assert not any(method == "POST" for method, _ in requests)
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(run())
