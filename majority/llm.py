"""AI 공급자별 비동기 호출, 재시도·한도 판별, Claude 사용량·비용 계산."""
import asyncio
import base64
import re
import time
from types import SimpleNamespace
import requests
from anthropic import Anthropic, AsyncAnthropic
from google import genai
from google.genai import types
from openai import AsyncOpenAI, OpenAI
from majority.settings import *  # noqa: F401,F403
from majority.models import *  # noqa: F401,F403


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def search_disabled_error(e: Exception) -> bool:
    """웹 검색 도구가 거부된 오류인지 (조직 설정에서 꺼짐, 도구 버전 미지원 등)."""
    msg = str(e).lower()
    return "web_search" in msg or "web_fetch" in msg or ("tool" in msg and ("not enabled" in msg or "not supported" in msg))


def usage_of(resps) -> dict:
    """Claude 응답들의 실제 사용량 합계: 입력·출력 토큰과 웹 검색 횟수."""
    u = {"input": 0, "output": 0, "searches": 0}
    for r in resps:
        x = getattr(r, "usage", None)
        if not x:
            continue
        u["input"] += (x.input_tokens or 0) + (getattr(x, "cache_creation_input_tokens", 0) or 0) \
            + (getattr(x, "cache_read_input_tokens", 0) or 0)
        u["output"] += x.output_tokens or 0
        for key, attr in (("cache_read", "cache_read_input_tokens"),
                          ("cache_write", "cache_creation_input_tokens")):
            value = getattr(x, attr, 0) or 0
            if value:
                u[key] = u.get(key, 0) + value
        cache = getattr(x, "cache_creation", None)
        hour = getattr(cache, "ephemeral_1h_input_tokens", 0) or 0
        if hour:
            u["cache_write_1h"] = u.get("cache_write_1h", 0) + hour
        stu = getattr(x, "server_tool_use", None)
        u["searches"] += (getattr(stu, "web_search_requests", 0) or 0) if stu else 0
    return u


def cost_of(model: str, usage) -> float:
    """사용량 → 달러. 미등록 단가는 별도 cost_note로 알리고 합계에서는 제외."""
    if not usage:
        return 0.0
    if model.startswith("gemini-"):
        prices = gemini_prices(model)
        if prices is None or not secret("GEMINI_PAID_TIER", True):
            return 0.0
        pin, pout, pcache = prices
        cached = min(usage.get("cached_input", 0), usage["input"])
        return ((usage["input"] - cached) * pin + cached * pcache + usage["output"] * pout) / 1_000_000
    if model not in CLAUDE_PRICES:
        return 0.0
    pin, pout = CLAUDE_PRICES[model]
    if model == "claude-haiku-5-5" and usage["input"] > 100_000:
        pin, pout = 0.50, 2.50
    read, write = usage.get("cache_read", 0), usage.get("cache_write", 0)
    hour = min(usage.get("cache_write_1h", 0), write)
    read_factor = 0.05 if model in ("claude-sonnet-5-5", "claude-opus-5-5") else 0.1
    input_cost = (usage["input"] - read - write) * pin + read * pin * read_factor
    input_cost += (write - hour) * pin * 1.25 + hour * pin * 2
    return (input_cost + usage["output"] * pout) / 1_000_000 + usage["searches"] * WEB_SEARCH_PRICE


def gemini_prices(model):
    """확인된 모델 또는 Secrets의 명시적 단가만 사용. 모르는 별칭은 추정하지 않음."""
    custom = secret("GEMINI_PRICES", {})
    if model in custom:
        try:
            rates = tuple(float(x) for x in custom[model])
            if len(rates) == 3 and all(0 <= x < float("inf") for x in rates):
                return rates
        except (TypeError, ValueError):
            pass
        return None
    if model == "gemini-3.8-flash" and today() >= "2027-01-01":
        return (1.50, 7.50, 0.15)
    return GEMINI_PRICES.get(model)


def gemini_usage(resp):
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return None
    return {"input": getattr(u, "prompt_token_count", 0) or 0,
            "output": (getattr(u, "candidates_token_count", 0) or 0)
                      + (getattr(u, "thoughts_token_count", 0) or 0),
            "cached_input": getattr(u, "cached_content_token_count", 0) or 0,
            "searches": 0}


def result_costs(result):
    """예전 기록의 cost는 Claude만 포함. 새 기록은 공급자별 합계를 유지."""
    return result.get("costs") or {"Claude": result.get("cost", 0), "Gemini": 0}


def monthly_costs(conversations, month):
    totals = {"Claude": 0.0, "Gemini": 0.0}
    for c in conversations:
        if "cost_by_provider" in c:
            for name in totals:
                totals[name] += c["cost_by_provider"].get(name, {}).get(month, 0)
        else:
            totals["Claude"] += (c.get("cost") or {}).get(month, 0)
    return totals


def cost_caption(costs):
    return (f"Claude 약 ${costs.get('Claude', 0):.4f} · Gemini 약 ${costs.get('Gemini', 0):.4f}"
            f" · 합계 약 ${sum(costs.values()):.4f}")


def effort_rejected(e: Exception) -> bool:
    """이 모델이 생각 수준(effort) 설정을 받지 않는다는 오류인지."""
    msg = str(e).lower()
    return "effort" in msg or "output_config" in msg


async def call_anthropic(p, model, system, text, images, search=False, on_text=None, effort=None):
    content = [{"type": "image",
                "source": {"type": "base64", "media_type": img.mime, "data": b64(img.data)}}
               for img in images]
    content.append({"type": "text", "text": text})
    extra = {"output_config": {"effort": effort}} if effort else {}
    if on_text and not search:            # 최종 판정: 글자가 나오는 대로 화면에 표시
        async with AsyncAnthropic(api_key=secret(p["key"])) as client:
            while True:
                try:
                    async with client.messages.stream(model=model, max_tokens=CLAUDE_MAX_TOKENS_STREAM,
                                                      system=system, **extra,
                                                      messages=[{"role": "user", "content": content}]) as stream:
                        out = []
                        async for chunk in stream.text_stream:
                            out.append(chunk)
                            on_text(chunk)
                        final = await stream.get_final_message()
                        return "".join(out), {"usage": usage_of([final]),
                                              "truncated": final.stop_reason == "max_tokens"}
                except Exception as e:
                    if extra and effort_rejected(e):      # 생각 수준을 지원하지 않는 모델이면 빼고 다시
                        extra = {}
                        on_text(None)
                        continue
                    raise
    meta = {"sources": [], "searches": 0, "fetches": 0, "search_note": ""}
    versions = list(CLAUDE_SEARCH_TOOLS) if search else []
    fetch_versions = list(CLAUDE_FETCH_TOOLS) if search else []

    def tool_note(message):
        meta["search_note"] = " · ".join(filter(None, (meta["search_note"], message)))

    async with AsyncAnthropic(api_key=secret(p["key"])) as client:
        while True:
            tool = versions[0] if versions else None
            fetch_tool = fetch_versions[0] if fetch_versions else None
            kwargs = dict(extra)
            tools = []
            if tool:
                tools.append({"type": tool, "name": "web_search", "max_uses": SEARCH_MAX_USES,
                              "allowed_callers": ["direct"],
                              "user_location": {"type": "approximate", "country": "KR",
                                                "timezone": "Asia/Seoul"}})
                if tool == "web_search_20250305":
                    tools[0].pop("allowed_callers")
            if fetch_tool:
                tools.append({"type": fetch_tool, "name": "web_fetch", "max_uses": FETCH_MAX_USES,
                              "max_content_tokens": FETCH_MAX_CONTENT_TOKENS,
                              "citations": {"enabled": True}})
                if fetch_tool != "web_fetch_20250910":
                    tools[-1].update(allowed_callers=["direct"], use_cache=False)
            if tools:
                kwargs["tools"] = tools
            messages = [{"role": "user", "content": content}]
            try:
                blocks, resps = [], []
                for _ in range(4):        # 서버 도구가 길어지면 pause_turn으로 끊어 오므로 이어서 요청
                    resp = await client.messages.create(model=model, max_tokens=CLAUDE_MAX_TOKENS,
                                                        system=system, messages=messages, **kwargs)
                    resps.append(resp)
                    blocks += resp.content
                    if resp.stop_reason != "pause_turn":
                        break
                    messages = messages + [{"role": "assistant",
                                            "content": [b.model_dump(exclude_none=True) for b in resp.content]}]
                break
            except Exception as e:
                if extra and effort_rejected(e):          # 생각 수준을 지원하지 않는 모델이면 빼고 다시
                    extra = {}
                    continue
                if (tool or fetch_tool) and search_disabled_error(e):
                    msg = str(e).lower()
                    if fetch_tool and (not tool or "web_fetch" in msg or "tools.1" in msg or "tools[1]" in msg):
                        fetch_versions.pop(0)
                        if not fetch_versions:
                            tool_note("페이지 본문 직접 조회 도구를 쓸 수 없습니다. 검색 발췌만으로는 최신 내용을 확인하지 못할 수 있습니다.")
                    else:
                        versions.pop(0)
                        if not versions:
                            tool_note(f"Claude 웹 검색을 쓸 수 없어 검색 없이 진행했습니다 ({str(e)[:80]})")
                    continue
                raise

    seen = set()
    for b in blocks:
        if b.type == "server_tool_use" and getattr(b, "name", "") == "web_search":
            meta["searches"] += 1
        elif b.type == "web_search_tool_result":
            res = b.content
            if isinstance(res, list):
                for r in res:
                    if r.url not in seen:
                        seen.add(r.url)
                        meta["sources"].append({"title": r.title or r.url, "url": r.url})
            else:
                tool_note(f"웹 검색 오류: {getattr(res, 'error_code', res)}")
        elif b.type == "server_tool_use" and getattr(b, "name", "") == "web_fetch":
            meta["fetches"] += 1
        elif b.type == "web_fetch_tool_result":
            res = b.content
            if getattr(res, "type", "") == "web_fetch_result":
                if res.url not in seen:
                    seen.add(res.url)
                    title = getattr(getattr(res, "content", None), "title", None) or res.url
                    meta["sources"].append({"title": title, "url": res.url})
            else:
                tool_note(f"페이지 본문 조회 실패: {getattr(res, 'error_code', 'unknown')}")
    out = "".join(b.text for b in blocks if b.type == "text")
    meta["usage"] = usage_of(resps)
    meta["truncated"] = bool(resps) and resps[-1].stop_reason == "max_tokens"
    return out, meta


async def call_gemini(p, model, system, text, images, search=False, on_text=None, effort=None):
    client = genai.Client(api_key=secret(p["key"]))
    parts = [types.Part.from_bytes(data=img.data, mime_type=img.mime) for img in images]
    parts.append(text)
    meta = {"sources": [], "searches": 0, "search_note": ""}
    level = RUNTIME.get("gemini_thinking")

    def config(with_search, with_thinking):
        cfg = dict(system_instruction=system)
        if with_search:
            cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        if with_thinking and level:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        return types.GenerateContentConfig(**cfg)

    async def run(cfg):
        if on_text:
            out = []
            last_usage, resolved_model = None, model
            async for chunk in await client.aio.models.generate_content_stream(
                    model=model, contents=parts, config=cfg):
                # 스트림 usage는 누적값이므로 더하지 않고 마지막 값만 보관.
                if getattr(chunk, "usage_metadata", None) is not None:
                    last_usage = chunk.usage_metadata
                resolved_model = getattr(chunk, "model_version", None) or resolved_model
                if chunk.text:
                    out.append(chunk.text)
                    on_text(chunk.text)
            return SimpleNamespace(text="".join(out), candidates=None,
                                   usage_metadata=last_usage, model_version=resolved_model)
        return await client.aio.models.generate_content(model=model, contents=parts, config=cfg)

    use_search, use_thinking = search, True
    while True:
        try:
            resp = await run(config(use_search, use_thinking))
            break
        except Exception as e:
            msg = str(e).lower()
            if use_thinking and level and "thinking" in msg:      # 생각 수준을 지원하지 않는 모델
                use_thinking = False
                continue
            if use_search:
                meta["search_note"] = f"Gemini 검색 연동을 쓸 수 없어 검색 없이 진행했습니다 ({str(e)[:80]})"
                use_search = False
                continue
            raise
    if use_search and resp.candidates:
        gm = resp.candidates[0].grounding_metadata
        if gm:
            meta["searches"] = len(gm.web_search_queries or [])
            for ch in gm.grounding_chunks or []:
                if ch.web and ch.web.uri:
                    meta["sources"].append({"title": ch.web.title or ch.web.uri, "url": ch.web.uri})
    meta["usage"] = gemini_usage(resp)
    meta["billing_model"] = getattr(resp, "model_version", None) or model
    if use_search:
        meta["cost_note"] = "Gemini 검색 요금은 무료 허용량에 따라 달라 토큰 비용만 계산했습니다."
    return resp.text or "", meta


async def call_openai_compat(p, model, system, text, images, search=False, on_text=None, effort=None):
    async with AsyncOpenAI(api_key=secret(p["key"]), base_url=p["base_url"]) as client:
        if on_text:
            stream = await client.chat.completions.create(
                model=model, max_tokens=p.get("max_tokens", 8000), stream=True,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": text}])
            out = []
            async for ev in stream:
                piece = ev.choices[0].delta.content if ev.choices else None
                if piece:
                    out.append(piece)
                    on_text(piece)
            return "".join(out)
        resp = await client.chat.completions.create(
            model=model, max_tokens=p.get("max_tokens", 8000),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": text}],
        )
        return resp.choices[0].message.content or ""


CALLERS = {"anthropic": call_anthropic, "gemini": call_gemini, "openai": call_openai_compat}


QUOTA_WORDS = ("429", "resource_exhausted", "quota", "rate limit", "rate_limit", "ratelimit",
               "too many requests", "credit balance")


def is_quota(msg: str) -> bool:
    """사용량·요청 한도 초과 여부."""
    return any(k in msg.lower() for k in QUOTA_WORDS)


def is_daily(msg: str) -> bool:
    """하루 한도 초과로 보이는지 (이 경우 기다려도 소용없으니 재시도하지 않음)."""
    m = msg.lower().replace("_", "").replace(" ", "")
    return any(k in m for k in ("perday", "daily", "creditbalance"))


def is_retryable(e: Exception) -> bool:
    msg = f"{type(e).__name__} {e}"
    if is_daily(msg):
        return False
    return is_quota(msg) or any(k in msg.lower() for k in ("overloaded", "529", "503"))


RETRY_AFTER_RE = re.compile(
    r"(?:retry(?:delay)?['\"]?\s*(?:in|:)|try again in)\s*['\"]?(\d+(?:\.\d+)?)\s*s\b", re.I)


def retry_after(e: Exception):
    """API가 알려준 재시도 대기 시간(초). Retry-After 헤더 → 오류 메시지의 'retry in 23s' 순으로 찾음."""
    headers = getattr(getattr(e, "response", None), "headers", None)
    if headers:
        try:
            return float(headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    m = RETRY_AFTER_RE.search(str(e))
    return float(m.group(1)) if m else None


async def ask(role, name, model, system, text, images, log=None, search=False,
              timeout=CALL_TIMEOUT_SEC, on_text=None, max_wait=RETRY_WAIT_SEC, effort=None):
    """예외를 삼키고 결과 dict로 돌려줘서 gather 전체가 깨지지 않게 함.
    timeout은 재시도까지 포함한 전체 시간 상한. 한도 초과·과부하면 API가 알려준 시간만큼 기다려 1회 재시도하되,
    max_wait보다 오래 기다려야 하면 재시도하지 않고 바로 실패를 돌려줌(호출한 쪽이 다른 모델로 교체)."""
    p = PROVIDERS[name]
    if images and not p["vision"]:
        text += (f"\n\n(참고: 이미지 {len(images)}장은 이 모델에 직접 전달되지 않았습니다. "
                 "<brief>의 이미지 설명을 근거로 답하세요.)")
        images = []
    t0 = time.perf_counter()
    deadline = t0 + timeout
    for attempt in (1, 2):
        try:
            kw = {"search": True} if search else {}
            if effort and p["kind"] == "anthropic":   # 생각 수준은 Claude에만 적용
                kw["effort"] = effort
            if on_text:
                on_text(None)                 # 재시도면 화면에 쓰던 글을 비움
                kw["on_text"] = on_text
            left = deadline - time.perf_counter()
            if left <= 0:
                raise asyncio.TimeoutError
            out = await asyncio.wait_for(CALLERS[p["kind"]](p, model, system, text, images, **kw), left)
            meta = {}
            if isinstance(out, tuple):
                out, meta = out
            ok = bool(out.strip())
            usage = meta.get("usage")
            billing_model = meta.get("billing_model", model)
            cost_note = meta.get("cost_note", "")
            if name in ("Claude", "Gemini") and ok:
                if usage is None:
                    cost_note = f"{name} 사용량을 받지 못해 일부 비용을 계산하지 못했습니다."
                elif name == "Gemini" and secret("GEMINI_PAID_TIER", True) and gemini_prices(billing_model) is None:
                    cost_note = f"Gemini {billing_model} 단가가 없어 일부 비용을 계산하지 못했습니다."
                elif name == "Claude" and billing_model not in CLAUDE_PRICES:
                    cost_note = f"Claude {billing_model} 단가가 없어 일부 비용을 계산하지 못했습니다."
            return {"role": role, "name": name, "model": model, "ok": ok,
                    "text": out if ok else "빈 응답", "sec": round(time.perf_counter() - t0, 1),
                    "sources": meta.get("sources", []), "searches": meta.get("searches", 0),
                    "fetches": meta.get("fetches", 0),
                    "search_note": meta.get("search_note", ""),
                    "usage": usage, "billing_model": billing_model, "cost_note": cost_note,
                    "cost": round(cost_of(billing_model, usage), 6),
                    "truncated": bool(meta.get("truncated"))}
        except (asyncio.TimeoutError, TimeoutError):
            return {"role": role, "name": name, "model": model, "ok": False, "timeout": True,
                    "text": f"{timeout}초 안에 응답하지 않았습니다.", "sec": round(time.perf_counter() - t0, 1)}
        except Exception as e:
            wait = retry_after(e)
            pause = wait if wait is not None else RETRY_WAIT_SEC
            room = deadline - time.perf_counter() - pause      # 기다린 뒤 남는 응답 시간
            if attempt == 1 and is_retryable(e) and pause <= max_wait and room >= 15:
                if log:
                    log(f"⏳ {name} 요청 한도/과부하 → {pause:.0f}초 후 재시도")
                await asyncio.sleep(pause)
                continue
            msg = f"{type(e).__name__}: {e}"
            low = msg.lower()
            too_large = any(k in low for k in ("413", "too large", "context_length", "maximum context",
                                                "context window", "too long"))
            return {"role": role, "name": name, "model": model, "ok": False, "text": msg,
                    "quota": is_quota(msg) and not too_large, "daily": is_daily(msg), "too_large": too_large,
                    "retry_after": wait, "sec": round(time.perf_counter() - t0, 1)}


def error_reason(text: str) -> str:
    """오류 메시지를 화면용 짧은 이유로."""
    low = text.lower()
    if any(k in low for k in ("503", "unavailable", "overloaded", "529")):
        return "서버 과부하(일시적)"
    if any(k in low for k in ("404", "not_found", "not found", "does not exist")):
        return "모델을 찾을 수 없음"
    if any(k in low for k in ("401", "403", "permission", "api key", "unauthorized")):
        return "API 키·권한 오류"
    if any(k in low for k in ("500", "internal")):
        return "서버 오류(일시적)"
    return f"오류 ({text[:60]})"
