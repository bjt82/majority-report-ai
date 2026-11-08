"""공개 페이지를 격리된 Chromium으로 렌더링하는 읽기 도구.

브라우저의 모든 연결은 공개 IP에만 연결하는 임시 프록시를 거칩니다.
사용자 프로필·쿠키·앱의 인증 키를 브라우저로 넘기지 않습니다.
"""
import asyncio
import base64
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
import ipaddress
import json
import shutil
import socket
from urllib.parse import urlsplit


BROWSER_MAX_USES = 3
BROWSER_TIMEOUT_SEC = 30
BROWSER_MAX_CHARS = 24_000
BROWSER_MAX_ACTIONS = 4

BROWSER_TOOL = {
    "name": "browser_read",
    "description": (
        "공개 웹페이지를 실제 Chromium으로 열어 JavaScript 로딩 후 보이는 텍스트·화면·링크·탭·선택항목을 읽습니다. "
        "검색 발췌/정적 본문이 오래됐거나 비어 있거나 사용자가 보는 내용과 다르면 사용하세요. "
        "첫 조회의 controls를 보고 필요하면 같은 URL에 actions를 지정해 다시 조회하세요. "
        "매 호출은 새 브라우저이므로 이전 조작은 유지되지 않습니다. "
        "읽기용 탭·일반 링크 이동·select/date 변경·스크롤·텍스트 대기만 지원합니다. "
        "로그인·제출·결제·다운로드·접근 제한 우회는 지원하지 않습니다. 최대 3회."
    ),
    "input_schema": {
        "type": "object", "properties": {
            "url": {"type": "string", "description": "사용자나 검색 결과에서 확인한 http(s) 공개 URL"},
            "actions": {"type": "array", "maxItems": BROWSER_MAX_ACTIONS, "items": {
                "type": "object", "properties": {
                    "kind": {"type": "string", "enum": ["tab", "link", "select", "date", "scroll", "wait"]},
                    "name": {"type": "string", "description": "tab/link의 화면 표시 이름(정확히 일치)"},
                    "index": {"type": "integer", "minimum": 0, "description": "controls에 표시된 select/date 인덱스"},
                    "value": {"type": "string", "description": "select의 option value 또는 date의 YYYY-MM-DD"},
                    "text": {"type": "string", "description": "wait: 화면에 나타날 텍스트"},
                }, "required": ["kind"], "additionalProperties": False,
            }},
        }, "required": ["url"], "additionalProperties": False,
    },
}


def public_target(url):
    """URL 문법 검증. DNS/IP 검증은 실제 연결 직전에 프록시가 수행."""
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 80, 443) or len(url) > 4096):
        raise ValueError("표준 http(s) 공개 URL만 읽을 수 있습니다.")
    return parsed


async def public_addresses(host, port):
    infos = await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM), 5)
    addresses = list(dict.fromkeys(info[4][0] for info in infos))
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError("내부·로컬 네트워크에는 접근할 수 없습니다.")
    return addresses


async def public_connection(host, port):
    for address in await public_addresses(host, port):
        try:
            return await asyncio.wait_for(asyncio.open_connection(address, port), 5)
        except OSError:
            continue
    raise OSError("공개 서버에 연결하지 못했습니다.")


@asynccontextmanager
async def public_proxy():
    """DNS 결과를 검사한 뒤 그 IP로 직접 연결해 DNS 재해석을 피함."""
    active = set()

    async def copy_stream(source, target):
        while chunk := await source.read(65536):
            target.write(chunk)
            await target.drain()

    async def handle(reader, writer):
        task = asyncio.current_task()
        active.add(task)
        remote = None
        pumps = []
        connected = False
        try:
            header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            lines = header.decode("latin-1").split("\r\n")
            method, destination, version = lines[0].split(" ", 2)
            if method == "CONNECT":
                target = public_target("https://" + destination)
                port = target.port or 443
            elif method in ("GET", "HEAD"):
                target = public_target(destination)
                port = target.port or (443 if target.scheme == "https" else 80)
            else:
                raise ValueError("읽기 요청만 허용합니다.")
            incoming, remote = await public_connection(target.hostname, port)
            if method == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            else:
                path = (target.path or "/") + (("?" + target.query) if target.query else "")
                headers = [line for line in lines[1:] if line and line.split(":", 1)[0].lower()
                           not in ("host", "proxy-connection", "connection", "proxy-authorization")]
                outgoing = [f"{method} {path} {version}", f"Host: {target.netloc}", "Connection: close", *headers]
                remote.write(("\r\n".join(outgoing) + "\r\n\r\n").encode("latin-1"))
                await remote.drain()
            await writer.drain()
            connected = True
            pumps = [asyncio.create_task(copy_stream(reader, remote)),
                     asyncio.create_task(copy_stream(incoming, writer))]
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        except (ValueError, OSError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            if not connected:
                writer.write(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
                with suppress(OSError):
                    await writer.drain()
        finally:
            for pump in pumps:
                pump.cancel()
            if pumps:
                await asyncio.gather(*pumps, return_exceptions=True)
            if remote is not None:
                remote.close()
            writer.close()
            active.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0, limit=16384)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        server.close()
        await server.wait_closed()
        pending = list(active)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def settle(page):
    from playwright.async_api import TimeoutError as PlaywrightTimeout
    with suppress(PlaywrightTimeout):
        await page.wait_for_load_state("networkidle", timeout=4000)
    await page.wait_for_timeout(400)


async def apply_action(page, action):
    kind = action["kind"]
    if kind in ("tab", "link"):
        locator = page.get_by_role(kind, name=action["name"], exact=True)
        if await locator.count() != 1:
            raise ValueError("같은 이름의 항목이 여러 개이거나 없습니다. controls를 확인하세요.")
        if kind == "link":
            href = await locator.get_attribute("href")
            from urllib.parse import urljoin
            url = urljoin(page.url, href or "")
            public_target(url)
            if not href or href.lower().startswith("javascript:"):
                raise ValueError("通常のURLリンクのみ移動できます。")
            await page.goto(url, wait_until="domcontentloaded")
        else:
            await locator.click()
    elif kind == "select":
        await page.locator("select:visible").nth(int(action["index"])).select_option(action["value"])
    elif kind == "date":
        from datetime import date
        date.fromisoformat(action["value"])
        await page.locator('input[type="date"]:visible').nth(int(action["index"])).fill(action["value"])
    elif kind == "scroll":
        await page.mouse.wheel(0, 800)
    elif kind == "wait":
        await page.get_by_text(action["text"], exact=False).first.wait_for(state="visible", timeout=6000)
    else:
        raise ValueError("지원하지 않는 읽기 조작입니다.")
    await settle(page)


async def render_page(url, actions=(), executable_path=None):
    """텍스트·controls·뷰포트 스크린샷. 쿠키/저장소를 공유하지 않는 1회성 세션."""
    public_target(url)
    if not isinstance(actions, (list, tuple)) or len(actions) > BROWSER_MAX_ACTIONS:
        raise ValueError("읽기 조작은 최대 4개입니다.")
    from playwright.async_api import async_playwright
    async with asyncio.timeout(BROWSER_TIMEOUT_SEC), public_proxy() as proxy, async_playwright() as pw:
        executable = executable_path or shutil.which("chromium") or shutil.which("chromium-browser")
        launch = {"headless": True, "proxy": {"server": proxy}, "args": [
            "--proxy-bypass-list=<-loopback>", "--disable-quic",
            "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
        ]}
        if executable:
            launch["executable_path"] = executable
        browser = await pw.chromium.launch(**launch)
        try:
            context = await browser.new_context(locale="ko-KR", timezone_id="Asia/Seoul",
                                                viewport={"width": 1280, "height": 900},
                                                service_workers="block", accept_downloads=False)
            context.set_default_timeout(6000)
            context.set_default_navigation_timeout(12000)

            async def guard(route):
                request = route.request
                try:
                    public_target(request.url)
                    if request.method not in ("GET", "HEAD") or request.resource_type == "media":
                        raise ValueError("읽기 외 요청")
                    await route.continue_()
                except ValueError:
                    await route.abort()

            await context.route("**/*", guard)
            await context.route_web_socket("**/*", lambda ws: ws.close())
            page = await context.new_page()
            context.on("page", lambda popup: popup.close())
            page.on("dialog", lambda dialog: dialog.dismiss())
            status = {"http_status": None}

            def response_status(response):
                if response.request.is_navigation_request() and response.frame == page.main_frame:
                    status["http_status"] = response.status

            page.on("response", response_status)
            await page.goto(url, wait_until="domcontentloaded")
            await settle(page)
            for action in actions:
                await apply_action(page, action)
            public_target(page.url)
            text = await page.locator("body").inner_text()
            for frame in page.frames[1:5]:
                with suppress(Exception):
                    text += f"\n[iframe {frame.url}]\n" + await frame.locator("body").inner_text(timeout=1000)
            controls = await page.evaluate("""() => {
                const visible = e => !!(e.getClientRects().length);
                const name = e => (e.getAttribute('aria-label') || e.innerText || '').trim().slice(0,120);
                return {
                    links: [...document.querySelectorAll('a[href]')].filter(visible).slice(0,40)
                        .map(e => ({name:name(e), url:e.href})),
                    tabs: [...document.querySelectorAll('[role=tab]')].filter(visible).slice(0,30).map(name),
                    selects: [...document.querySelectorAll('select')].filter(visible).slice(0,15)
                        .map((e,index) => ({index, name:e.getAttribute('aria-label') || e.name || e.id,
                            options:[...e.options].slice(0,50).map(o=>({name:o.text.slice(0,120),value:o.value.slice(0,200)}))})),
                    dates: [...document.querySelectorAll('input[type=date]')].filter(visible).slice(0,10)
                        .map((e,index)=>({index,value:e.value}))
                };
            }""")
            controls_truncated = False
            while len(json.dumps(controls, ensure_ascii=False)) > 12_000:
                group = max(controls.values(), key=lambda items: len(json.dumps(items, ensure_ascii=False)))
                group.pop()
                controls_truncated = True
            screenshot = await page.screenshot(type="jpeg", quality=55, timeout=4000)
            return {"ok": True, "url": page.url, "title": await page.title(), "method": "browser",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(), "actions": list(actions),
                    "text": text[:BROWSER_MAX_CHARS], "truncated": len(text) > BROWSER_MAX_CHARS,
                    "controls": controls, "screenshot": base64.b64encode(screenshot).decode(),
                    "controls_truncated": controls_truncated,
                    **status,
                    "limitations": "로그인 없는 읽기 전용 조회. POST·WebSocket·다운로드는 사용하지 않습니다."}
        finally:
            await browser.close()


async def browser_read(arguments):
    try:
        return await render_page(arguments["url"], arguments.get("actions", []))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        reason = "브라우저나 사이트 응답을 처리하지 못했습니다."
        if isinstance(exc, ImportError) or "Executable doesn't exist" in str(exc):
            reason = "Playwright 또는 Chromium이 설치되어 있지 않습니다."
        elif isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or type(exc).__name__ == "TimeoutError":
            reason = "페이지 로딩 또는 지정한 항목 대기 시간이 초과됐습니다."
        elif isinstance(exc, ValueError):
            reason = str(exc)
        elif "net::ERR_" in str(exc):
            reason = "페이지 연결에 실패했습니다(네트워크·접근 제한·차단된 요청 가능)."
        return {"ok": False, "method": "browser", "error": type(exc).__name__,
                "message": reason + " 페이지 부재의 증거가 아닙니다."}
