"""토론 진행(되묻기 → 정리 → 1차 판단 → 반론 → 판정)과 한도 초과 모델 자동 제외."""
import asyncio
import contextlib
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from majority.settings import *  # noqa: F401,F403
from majority.models import *  # noqa: F401,F403
from majority.llm import *  # noqa: F401,F403
from majority.prompts import *  # noqa: F401,F403


async def hub_clarify(ui: UserInput, hub: str, hub_model: str, cooldowns: dict, log=None, cost_details=None):
    """되물을 질문 목록. 빠르게 끝나도록 허브 AI의 가벼운 모델(예: Haiku)을 먼저 쓰고, 안 되면 허브 모델.
    실패하면 [] → 되묻지 않고 바로 토론. 반환: (질문 목록, 이 단계 비용 달러)."""
    cands = [m for m in fallbacks(hub, hub_model)[:1] + [hub_model] if not cooled(hub, m, cooldowns)]
    cost = 0.0
    for m in dict.fromkeys(cands):
        r = await ask("허브 확인", hub, m, CLARIFY_SYSTEM, clarify_prompt(ui), ui.images, log,
                      effort=CLAUDE_EFFORT["clarify"],
                      timeout=CLARIFY_TIMEOUT_SEC, max_wait=QUICK_RETRY_SEC)
        cost += r.get("cost", 0)
        if cost_details is not None:
            cost_details[hub] = round(cost_details.get(hub, 0) + r.get("cost", 0), 6)
            if r.get("cost_note") and log:
                log(r["cost_note"])
        if r["ok"]:
            return parse_clarify(r["text"]), cost
        if r.get("quota"):
            cooldowns[ckey(hub, m)] = cooldown_until(hub, r)
    return [], cost


def clock(ts: float) -> str:
    """제외 해제 시각을 사용자 시간대(기본 한국)로 표시. 서버는 UTC로 돌기 때문."""
    tz = ZoneInfo(secret("APP_TIMEZONE", "Asia/Seoul"))
    when = datetime.fromtimestamp(ts, tz)
    day = (when.date() - datetime.now(tz).date()).days
    return ("" if day <= 0 else "내일 " if day == 1 else when.strftime("%m/%d ")) + when.strftime("%H:%M")


def cooldown_until(name: str, r: dict) -> float:
    """한도 초과한 모델을 언제까지 뺄지. Gemini 하루 한도는 태평양 시간 자정에 초기화됨.
    분당 한도는 API가 알려준 대기 시간이 있으면 그만큼만(최소 1분) 뺌."""
    if r.get("daily") and PROVIDERS[name]["kind"] == "gemini":
        pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        reset = datetime.combine(pt.date() + timedelta(days=1), datetime.min.time(), tzinfo=pt.tzinfo)
        return reset.timestamp() + 60
    if r.get("daily"):
        return time.time() + COOLDOWN_DAY_SEC
    if r.get("retry_after"):
        return time.time() + max(r["retry_after"], 60)
    return time.time() + COOLDOWN_MIN_SEC


def ckey(name: str, model: str) -> str:
    """제외 기록의 키. 한도는 모델마다 따로라서 'AI/모델' 단위로 기록."""
    return f"{name}/{model}"


def cooled(name: str, model: str, cooldowns: dict) -> bool:
    return cooldowns.get(ckey(name, model), 0) > time.time()


def judge_b_plan(ui: UserInput, roles: dict, models: dict, cooldowns: dict):
    """API 호출 없이 실제 입력으로 판단 B 경로 결정. 키 없는 AI·제외/404 모델은 건너뜀."""
    full = bool(ui.images) or len(ui.base_text()) > AUTO_B_INPUT_CHARS
    reason = "이미지 포함" if ui.images else (f"전체 입력 {AUTO_B_INPUT_CHARS:,}자 초과" if full else "짧은 텍스트")
    available = available_providers()
    candidates = []
    if not full:
        candidates += [(n, PROVIDERS[n]["model"]) for n in ("Groq", "Mistral") if n in available]
    if "Claude" in available:
        # 허브 쪽 가벼운 모델 (ANTHROPIC_FALLBACK 설정을 따름, 기본 Haiku 5.5)
        candidates.append(("Claude", judge_model("Claude", "Claude", PROVIDERS["Claude"]["model"])))
        # Haiku도 불가능하면 마지막으로 기존 Claude 기본 모델 사용.
        candidates.append(("Claude", PROVIDERS["Claude"]["model"]))
    elif full:
        # Claude 키가 없는 구성에서는 원문·이미지를 받을 수 있는 기존 판단 B만 허용.
        if PROVIDERS[roles["b"]]["vision"]:
            candidates.append((roles["b"], models["b"]))
    pair_a = (roles["a"], models["a"])
    candidates = [(n, m) for n, m in dict.fromkeys(candidates)
                  if m and (n, m) != pair_a and not cooled(n, m, cooldowns) and not is_missing(n, m)]
    return {"mode": "full" if full else "free_first", "reason": reason, "candidates": candidates}


class NullBoard:
    """진행판이 없을 때(테스트 등) 쓰는 빈 진행판."""
    stream = None
    def start(self, *a, **k): pass
    def done(self, *a, **k): pass
    def skip(self, *a, **k): pass


async def run_pipeline(ui: UserInput, roles: dict, models: dict, log, cooldowns: dict, search=False,
                       board=None, fast=False, judge_timeout=JUDGE_TIMEOUT_DEFAULT, auto_judge_b=False):
    """cooldowns: {'AI/모델': 제외 해제 시각}. 사용량 초과가 나면 여기에 기록되어 다음 질문에도 자동 제외.
    board: 단계별 진행 상황을 그리는 진행판. fast: 상호 토론(②)을 건너뛰는 빠른 모드.

    판단자가 한도 초과·오류면 아래 순서로 교체 (상호 검토가 유지되는 쪽부터):
      1) 같은 AI의 낮은 모델  2) 다른 AI  3) 허브 AI의 다른 모델(예: Haiku)  4) 허브가 같은 모델로 겸함"""
    board = board or NullBoard()
    spent = []                                        # 이번 질문에서 성공·실패 상관없이 청구된 비용
    costs = {"Claude": 0.0, "Gemini": 0.0}

    async def tracked_ask(*a, **k):                   # 모든 호출의 비용을 모으고, 잘린 답은 알림
        r = await ask(*a, **k)
        spent.append(r.get("cost", 0) or 0)
        name = r["name"]
        if name in costs:
            costs[name] = round(costs[name] + (r.get("cost", 0) or 0), 6)
        if r.get("cost_note"):
            note("💰 " + r["cost_note"])
        if hasattr(board, "update_cost"):
            board.update_cost(costs)
        if r.get("ok") and r.get("truncated"):
            note(f"✂️ {r['role']}({r['name']})의 답이 최대 길이에 닿아 끝부분이 잘렸을 수 있습니다.")
        return r

    hub = roles["hub"]
    names = {"a": roles["a"], "b": roles["b"]}
    mdl = {"a": models["a"], "b": models["b"]}
    prim = dict(mdl)                                  # 각 자리의 '기본 모델' (낮은 모델 목록의 출발점)
    tried = {k: {(names[k], mdl[k])} for k in ("a", "b")}
    other_slot = {"a": "b", "b": "a"}
    role_label = {"a": "판단 A", "b": "판단 B"}
    imgs = ui.images
    notices = []

    def note(msg):
        notices.append(msg)
        log(msg)

    def pick_substitute(slot):
        """slot에 넣을 (AI, 모델). 이번 질문에서 이미 써 본 것·제외 중인 것·상대 판단자와 같은 것은 건너뜀."""
        cur, other = names[slot], names[other_slot[slot]]
        cands = [(cur, m) for m in chain(cur, prim[slot])]
        for n in available_providers():
            if n not in (cur, other, hub):
                cands += [(n, m) for m in chain(n, resolve_model(n))]
        cands += [(hub, m) for m in chain(hub, models["hub"]) if m != models["hub"]]
        cands.append((hub, models["hub"]))             # 최후: 허브가 같은 모델로 겸함
        for n, m in cands:
            if (m and (n, m) not in tried[slot] and (n, m) != (other, mdl[other_slot[slot]])
                    and not cooled(n, m, cooldowns)):
                return n, m
        return None

    def bench(slot, why, key=None):
        """slot의 판단자를 교체. 성공하면 True."""
        old, old_m = names[slot], mdl[slot]
        who = f"{role_label[slot]}({old} {short(old_m)})"
        sub = pick_substitute(slot)
        if not sub:
            note(f"⚠️ {who} {why} → 대신할 AI가 없어 이 판단자 없이 진행합니다.")
            return False
        n, m = sub
        same = [x for x in chain(old, prim[slot]) if x != old_m]   # 교체 전 기준의 같은 AI 낮은 모델
        names[slot], mdl[slot] = n, m
        tried[slot].add(sub)
        if n != old:
            prim[slot] = m
        if n != old and not same:
            why += f" · {old}에 낮은 모델 없음"
        elif n != old:
            why += f" · {old}의 낮은 모델도 사용 불가"
        if n == old:
            how = f"같은 AI의 낮은 모델 {short(m)}로 진행"
        elif n == hub and m == models["hub"]:
            how = "허브가 판단자를 겸해 진행합니다 (같은 모델이라 상호 검토 효과가 줄어듭니다)"
        elif n == hub:
            how = f"허브 쪽 다른 모델 {n} {short(m)}로 진행"
        else:
            how = f"대타: {n} {short(m)}"
        if key:
            board.start(key, f"{n} · {short(m)}", f"{old} {short(old_m)} 대신")
        note(f"⚠️ {who} {why} → {how}")
        return True

    def limit_of(name, factor=1.0):
        return int(PROVIDERS[name].get("max_input_chars", 0) * factor)

    plan = judge_b_plan(ui, roles, models, cooldowns) if auto_judge_b else None
    b_index, free_left = 0, AUTO_B_FREE_BUDGET_SEC
    if plan:
        route = " → ".join(f"{n} {pretty(m)}" for n, m in plan["candidates"])
        note(f"🧭 판단 B 자동 선택 · {plan['reason']} · {route or '사용 가능한 판단자 없음'}")

    async def adaptive_b_call(label, system, make, key):
        """무료 후보는 질문 전체 15초만 사용. 단계 전체 상한 안에서 Haiku 응답 시간을 확보."""
        nonlocal b_index, free_left
        end = time.perf_counter() + judge_timeout
        last = {"role": label, "name": names["b"], "model": mdl["b"], "ok": False,
                "text": "판단 B에 사용할 수 있는 모델이 없습니다.", "sec": 0}
        while b_index < len(plan["candidates"]):
            name, model = plan["candidates"][b_index]
            if (cooled(name, model, cooldowns) or is_missing(name, model)
                    or (name, model) == (names["a"], mdl["a"])):
                b_index += 1
                continue
            remaining = end - time.perf_counter()
            if remaining <= 0:
                break
            free = name in ("Groq", "Mistral")
            # 무료 시도로 단계 시간을 모두 쓰지 않게 최대 30초 또는 단계 시간의 절반을 남김.
            reserve = min(30.0, judge_timeout / 2)
            timeout = min(AUTO_B_FREE_CALL_SEC, free_left, remaining - reserve) if free else remaining
            if timeout <= 0:
                note(f"⌛ 판단 B · 무료 AI 대기 예산을 마쳐 {name} 대신 다음 후보로 진행합니다.")
                b_index += 1
                continue
            names["b"], mdl["b"] = name, model
            board.start(key, f"{name} · {pretty(model)}", "자동 선택")
            t0 = time.perf_counter()
            last = await tracked_ask(label, name, model, system, make(limit_of(name)), imgs, log,
                                     timeout=timeout, max_wait=0 if free else QUICK_RETRY_SEC)
            if free:
                free_left = max(0, free_left - (time.perf_counter() - t0))
            if last["ok"]:
                board.done(key, last)
                return last
            if last.get("quota"):
                cooldowns[ckey(name, model)] = cooldown_until(name, last)
                why = f"사용량 한도 초과 ({clock(cooldowns[ckey(name, model)])}까지 자동 제외)"
            elif last.get("timeout"):
                why = f"{timeout:g}초 대기 초과"
                if free:
                    cooldowns[ckey(name, model)] = time.time() + 60
            else:
                why = error_reason(last["text"])
                if why == "모델을 찾을 수 없음":
                    missing_models()[ckey(name, model)] = time.time()
            b_index += 1
            note(f"⚠️ {label}({name} {pretty(model)}) {why} → 다음 후보로 전환합니다.")
        board.done(key, last)
        return last

    async def judge_call(slot, label, system, make, key):
        """make(입력 한도) → 그 판단자에게 맞는 크기의 프롬프트.
        judge_timeout은 교체·재시도까지 포함한 이 단계 전체의 시간 상한."""
        if slot == "b" and plan is not None:
            return await adaptive_b_call(label, system, make, key)
        end = time.perf_counter() + judge_timeout

        def call(limit):
            return tracked_ask(label, names[slot], mdl[slot], system, make(limit), imgs, log,
                       timeout=max(end - time.perf_counter(), 1), max_wait=QUICK_RETRY_SEC)

        board.start(key, names[slot] if mdl[slot] == prim[slot] and names[slot] != hub
                    else f"{names[slot]} · {short(mdl[slot])}")
        r = await call(limit_of(names[slot]))
        shrunk = False
        while not r["ok"]:
            failed, fmodel = names[slot], mdl[slot]
            if r.get("too_large") and not shrunk:     # 요청이 너무 크면 더 줄여서 한 번 더
                shrunk = True
                small = limit_of(failed, 0.5) or 6000
                log(f"✂️ {label}({failed}) 요청이 커서 {small:,}자로 줄여 다시 시도")
                r = await call(small)
                continue
            if r.get("timeout") or end - time.perf_counter() < 10:   # 느린 판단자는 기다리지 않고 빼고 진행
                note(f"⌛ {label}({failed})가 {judge_timeout}초 안에 답하지 않아 빼고 진행합니다.")
                board.done(key, r)
                return r
            if r.get("quota"):
                cooldowns[ckey(failed, fmodel)] = cooldown_until(failed, r)
                why = f"사용량 한도 초과 ({clock(cooldowns[ckey(failed, fmodel)])}까지 자동 제외)"
            elif r.get("too_large"):
                why = "요청이 이 AI의 무료 한도보다 큼"
            else:
                why = error_reason(r["text"])
                if why == "모델을 찾을 수 없음":   # 다음 질문부터는 시도하지 않음
                    missing_models()[f"{failed}/{fmodel}"] = time.time()
            if not bench(slot, why, key):
                board.done(key, r)
                return r
            shrunk = False
            r = await call(limit_of(names[slot]))
        board.done(key, r)
        return r

    # 사용량 초과로 제외 중인 모델은 시작 전에 미리 교체 (해제되면 다음 질문부터 원래 모델로 돌아감)
    for slot in ("a", "b"):
        if slot == "b" and plan is not None:
            continue
        if cooled(names[slot], mdl[slot], cooldowns):
            until = clock(cooldowns[ckey(names[slot], mdl[slot])])
            bench(slot, f"사용량 초과로 {until}까지 제외 중")

    # ⓪ 허브의 질문 정리 (허브 모델이 한도 초과면 허브 AI의 낮은 모델로)
    search = search and PROVIDERS[hub]["kind"] in ("anthropic", "gemini")
    hub_models = [m for m in chain(hub, models["hub"]) if not cooled(hub, m, cooldowns)] or [models["hub"]]
    if hub_models[0] != models["hub"]:
        note(f"⚠️ 허브({hub} {short(models['hub'])}) 사용량 초과로 제외 중 → {short(hub_models[0])}로 정리합니다")
    for i, hm in enumerate(hub_models):
        board.start("brief", hub if hm == models["hub"] else f"{hub} · {short(hm)}",
                    "필요하면 웹 검색" if search else "")
        brief_res = await tracked_ask("허브 정리", hub, hm, HUB_BRIEF_SYSTEM, brief_prompt(ui, search),
                                      imgs, log, search=search, effort=CLAUDE_EFFORT["brief"])
        board.done("brief", brief_res)
        if brief_res["ok"] or not brief_res.get("quota") or i == len(hub_models) - 1:
            break
        cooldowns[ckey(hub, hm)] = cooldown_until(hub, brief_res)
        note(f"⚠️ 허브({hub} {short(hm)}) 사용량 한도 초과 → {short(hub_models[i + 1])}로 다시 정리합니다")
    brief = brief_res["text"] if brief_res["ok"] else ""
    sources = brief_res.get("sources", [])
    if brief_res.get("search_note"):
        note(f"🔎 {brief_res['search_note']}")
    if brief_res.get("searches"):
        log(f"🔎 웹 검색 {brief_res['searches']}회, 출처 {len(sources)}개 확인")
    if brief_res.get("fetches"):
        log(f"🔎 페이지 본문 조회 {brief_res['fetches']}회 시도 (요청 날짜의 정보 확인 여부는 브리프 참고)")
    visits = brief_res.get("browser_visits", [])
    if visits:
        succeeded = sum(bool(v.get("ok")) for v in visits)
        log(f"🔎 브라우저 조회 {len(visits)}회 중 {succeeded}회 완료 (내용 일치 여부는 별도 검증)")
        evidence = "\n".join(
            f"- {v.get('url', '(주소 확인 실패)')}: "
            + (f"브라우저 렌더링 완료, 조회 시각 {v.get('retrieved_at', '')}"
               + (", 텍스트 일부 생략" if v.get("truncated") else "")
               if v.get("ok") else f"브라우저 조회 실패 ({v.get('error', 'unknown')})")
            for v in visits
        )
        brief += ("\n\n<browser_retrieval_log>\n" + evidence + "\n</browser_retrieval_log>\n"
                  "(앱이 기록한 실제 조회 방식입니다. 렌더링 완료 자체는 요청 정보의 검증 성공을 뜻하지 않습니다.)")
    if not brief:
        log(f"허브 정리 실패 → 원래 질문으로 토론 진행 ({brief_res['text'][:100]})")

    # ① 1차 판단
    def jp(limit):
        return judge_prompt(ui, brief, limit)

    fa, fb = await asyncio.gather(
        judge_call("a", "판단 A", JUDGE_SYSTEM, jp, "a1"),
        judge_call("b", "판단 B", JUDGE_SYSTEM, jp, "b1"),
    )
    firsts = [d for d in (fa, fb) if d["ok"]]
    if not firsts:                    # 둘 다 실패해도 답은 냄: 허브가 혼자 답하고 그 사실을 알림
        note(f"⚠️ 두 판단자 모두 답하지 못해(판단 A {fa['name']}: {error_reason(fa['text'])} · "
             f"판단 B {fb['name']}: {error_reason(fb['text'])}) 허브가 혼자 답합니다. 상호 검토가 빠진 답입니다.")

    # ② 상호 토론
    debates = []
    if len(firsts) == 2 and not fast:
        da, db = await asyncio.gather(
            judge_call("a", "판단 A 반론", DEBATE_SYSTEM, lambda n: debate_prompt(ui, brief, fa, fb, n), "a2"),
            judge_call("b", "판단 B 반론", DEBATE_SYSTEM, lambda n: debate_prompt(ui, brief, fb, fa, n), "b2"),
        )
        debates = [d for d in (da, db) if d["ok"]]
    else:
        why = "빠른 모드" if fast else ("판단자 모두 실패" if not firsts else "판단자 한쪽 실패")
        board.skip("a2", why)
        board.skip("b2", why)

    # ③ 최종 판정: 허브 → 허브 AI의 낮은 모델 → 판단자 순 (제외 중인 모델은 건너뜀)
    vp = verdict_prompt(ui, brief, firsts, debates, sources)
    candidates = [(hub, m) for m in chain(hub, models["hub"])]
    candidates += [(names[k], mdl[k]) for k in ("a", "b") if names[k] != hub]
    candidates = list(dict.fromkeys(candidates))
    candidates = [c for c in candidates if not cooled(c[0], c[1], cooldowns)] or candidates
    final = None
    for i, (name, model) in enumerate(candidates):
        is_hub_model = (name, model) == (hub, models["hub"])
        board.start("verdict", name if is_hub_model else f"{name} · {short(model)}",
                    "" if is_hub_model else "허브 대신 판정")
        final = await tracked_ask("최종 판정", name, model, HUB_VERDICT_SYSTEM, vp, imgs, log,
                                  on_text=getattr(board, "stream", None), effort=CLAUDE_EFFORT["verdict"])
        board.done("verdict", final)
        if final["ok"]:
            break
        if final.get("quota"):
            cooldowns[ckey(name, model)] = cooldown_until(name, final)
        if i + 1 < len(candidates):
            nxt = candidates[i + 1]
            note(f"⚠️ 판정({name} {short(model)}) {'사용량 한도 초과' if final.get('quota') else '실패'}"
                 f" → {nxt[0]} {short(nxt[1])}가 판정합니다.")
        else:
            note(f"⚠️ {name} 판정 실패: {final['text'][:80]}")

    return {"brief": brief_res, "firsts": [fa, fb], "debates": debates, "final": final,
            "notices": notices, "sources": sources, "searches": brief_res.get("searches", 0),
            "fast": fast, "cost": round(sum(spent), 6), "costs": costs,
            "judge_b_route": plan}


async def run_with_ticker(job, render, interval: float = 1.0):
    """job(코루틴)을 돌리면서 interval초마다 render()로 진행판을 그림.
    render()에서 Streamlit의 중단·새 질문 신호(BaseException)가 오면 job을 즉시 취소하고 그 신호를 다시 던짐.
    → 진행 중인 AI 호출이 끝나길 기다리지 않고 1초 안에 멈추며, 끝나지 않은 호출은 비용도 덜 듦."""
    task = asyncio.create_task(job)
    while not task.done():
        try:
            render()
        except BaseException:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            raise
        await asyncio.wait({task}, timeout=interval)
    return task.result()
