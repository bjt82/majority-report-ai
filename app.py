"""
Majority Report AI — 허브 AI가 주재하는 두 AI의 상호 토론

파이프라인 (질문 1회 = API 호출 6회, 허브는 2회만 사용)
  ⓪ 질문 정리  : 허브가 질문·첨부·이전 최종안을 분석해 토론용 브리프 작성
  ① 1차 판단   : 판단 A ∥ 판단 B 가 브리프를 보고 독립적으로 동시 답변
  ② 상호 토론  : A는 B의 답을, B는 A의 답을 반박·인정하며 자기 입장을 수정
  ③ 최종 판정  : 허브가 토론 전체를 보고
                  Majority Report(최종 결론) + Minority Report(쟁점·판정 근거) 작성

후속 질문은 이전 질문과 '최종 판정(Majority Report)'을 기준으로 이어집니다.

지원 AI (Secrets에 키가 있는 것만 표시)
  Claude(ANTHROPIC_API_KEY) · Gemini(GEMINI_API_KEY) · Groq(GROQ_API_KEY)
  Mistral(MISTRAL_API_KEY) · OpenRouter(OPENROUTER_API_KEY)

실행: streamlit run app.py
"""
import asyncio
import hmac
import json
import re
import secrets as pysecrets
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import streamlit as st
from majority.settings import *  # noqa: F401,F403
from majority.models import *  # noqa: F401,F403
from majority.llm import *  # noqa: F401,F403
from majority.prompts import *  # noqa: F401,F403
from majority.pipeline import *  # noqa: F401,F403
from majority.store import *  # noqa: F401,F403


def model_picker(label: str, provider: str, key: str, default=None):
    """API에서 목록을 불러와 드롭다운으로 표시. 실패하면 기본값만 표시."""
    default = default or PROVIDERS[provider]["model"]
    err = None
    try:
        options = list_models(provider, secret(PROVIDERS[provider]["key"]))
    except Exception as e:
        options, err = [], f"{type(e).__name__}: {e}"
    ids = [i for i, _ in options]
    if default and default not in ids:
        options.insert(0, (default, default))
        ids.insert(0, default)
    if not ids:
        st.error(f"{provider} 모델 목록이 비어 있습니다.")
        return None
    names = dict(options)
    choice = st.selectbox(
        label, ids, index=ids.index(default) if default in ids else 0, key=key,
        format_func=lambda i: i if names[i] == i else f"{names[i]}  ·  {i}",
    )
    if err:
        st.caption(f"⚠️ 목록 조회 실패, 기본값 사용 ({err[:80]})")
    return choice


# ───────────────────────── 5) UI ─────────────────────────
st.set_page_config(page_title=APP_NAME, page_icon="🔮", layout="wide")

# 공개 URL에서 API 요금 도용을 막는 비밀번호 게이트
PW_MAX_TRIES = 5                             # 연속으로 틀릴 수 있는 횟수
PW_LOCK_SEC = 60                             # 그 뒤 잠그는 시간
app_pw = secret("APP_PASSWORD")
if not app_pw and not secret("ALLOW_NO_PASSWORD", False):
    # 비밀번호가 없으면 링크를 아는 누구나 유료 API 크레딧을 쓸 수 있으므로 열지 않음
    st.title(f"🔮 {APP_NAME}")
    st.error("접속 비밀번호가 설정되지 않아 앱을 열 수 없습니다.\n\n"
             "Streamlit Cloud → 앱 **Settings → Secrets**에 `APP_PASSWORD = \"원하는-비밀번호\"`를 추가하세요. "
             "(혼자 쓰는 로컬 실행이라면 `ALLOW_NO_PASSWORD = true`로 끌 수 있습니다.)")
    st.stop()
if app_pw and not st.session_state.get("authed"):
    st.title(f"🔮 {APP_NAME}")
    lock_left = st.session_state.get("pw_lock_until", 0) - time.time()
    if lock_left > 0:
        st.error(f"비밀번호를 {PW_MAX_TRIES}번 틀려 {lock_left:.0f}초 동안 잠겼습니다. 잠시 후 새로고침하세요.")
        st.stop()
    entered = st.text_input("접속 비밀번호", type="password")
    if entered and hmac.compare_digest(entered, str(app_pw)):
        st.session_state.authed = True
        st.session_state.pop("pw_fails", None)
        st.rerun()
    elif entered:
        fails = st.session_state.get("pw_fails", 0) + 1
        st.session_state.pw_fails = fails
        if fails >= PW_MAX_TRIES:
            st.session_state.pw_lock_until = time.time() + PW_LOCK_SEC
            st.session_state.pw_fails = 0
        st.error("비밀번호가 틀렸습니다.")
    st.stop()


def pick_default(role: str, avail: list, taken=()):
    for n in DEFAULT_ROLES[role]:
        if n in avail and n not in taken:
            return n
    return next((n for n in avail if n not in taken), avail[0])


class ProgressBoard:
    """토론 진행판: 단계별로 누가 무엇을 하는지, 얼마나 걸리는지 1초마다 갱신."""
    STEPS = [("brief", "⓪ 질문 정리"), ("a1", "① 1차 판단 A"), ("b1", "① 1차 판단 B"),
             ("a2", "② 반론 A"), ("b2", "② 반론 B"), ("verdict", "③ 최종 판정")]
    ICON = {"wait": "⬜", "run": "⏳", "ok": "✅", "fail": "❌", "skip": "⏭️"}

    def __init__(self, roles, status, models=None, initial_costs=None):
        self.t0 = time.time()
        self.status = status
        models = models or {}

        def judge(k):                     # 허브와 같은 AI인 판단자는 모델명을 붙여 구분 (예: Claude · claude-haiku-5-5)
            n = roles[k]
            return f"{n} · {short(models[k])}" if n == roles["hub"] and models.get(k) else n
        who = {"brief": roles["hub"], "a1": judge("a"), "b1": judge("b"),
               "a2": judge("a"), "b2": judge("b"), "verdict": roles["hub"]}
        self.rows = {k: {"state": "wait", "name": who[k], "extra": "", "t": 0, "sec": 0, "preview": ""}
                     for k, _ in self.STEPS}
        self.expanded = True              # 1초마다 라벨을 갱신할 때 펼침 상태를 함께 보내야 접히지 않음
        self.bar = st.progress(0.0, text="시작하는 중…")
        self.box = st.empty()
        self.cost_box = st.empty()
        self.initial_costs = dict(initial_costs or {})
        self.notes = st.container()

    def update_cost(self, costs):
        combined = {n: costs.get(n, 0) + self.initial_costs.get(n, 0) for n in ("Claude", "Gemini")}
        self.cost_box.caption("💰 응답 완료분 · " + cost_caption(combined))

    def start(self, key, name, extra=""):
        self.rows[key].update(state="run", name=name, extra=extra, t=time.time())
        self.render()

    def done(self, key, res):
        row = self.rows[key]
        row.update(state="ok" if res["ok"] else "fail", sec=res["sec"])
        if res["ok"]:
            clean = (re.sub(r"[*_`#>|]", "", l).strip(" -") for l in res["text"].splitlines())
            line = next((l for l in clean if l), "")
            row["preview"] = line[:70] + ("…" if len(line) > 70 else "")
            if res.get("searches"):
                row["extra"] = f"웹 검색 {res['searches']}회 · 출처 {len(res.get('sources', []))}개"
        else:
            row["preview"] = res["text"][:70]
        self.render()

    def skip(self, key, why):
        self.rows[key].update(state="skip", extra=why)
        self.render()

    def note(self, msg):
        self.notes.caption(msg)

    live = None          # 최종 판정을 실시간으로 그릴 자리 (진행판 바깥, 답변 위치)

    def stream(self, chunk):
        """최종 판정 글자를 받는 대로 표시. chunk=None이면 새로 시작(재시도)."""
        if chunk is None:
            self.streamed, self.last_draw = "", 0.0
            self.expanded = False
            if self.status is not None:
                self.status.update(expanded=False)       # 진행판을 접고 답에 집중
            return
        self.streamed += chunk
        now = time.time()
        if self.live is not None and now - self.last_draw > 0.12:
            self.last_draw = now
            major = self.streamed.split(MINORITY_MARKER, 1)[0]
            self.live.markdown("##### 📜 Majority Report · 작성 중\n\n" + major + " ▌")

    def render(self):
        now = time.time()
        lines, finished, running = [], 0, []
        for key, label in self.STEPS:
            r = self.rows[key]
            if r["state"] == "run":
                status = f"{now - r['t']:.0f}초째 진행 중"
                running.append(label)
            elif r["state"] in ("ok", "fail"):
                status = f"{r['sec']}초"
                finished += 1
            elif r["state"] == "skip":
                status = "건너뜀"
                finished += 1
            else:
                status = "대기"
            extra = f" · {r['extra']}" if r["extra"] else ""
            line = f"{self.ICON[r['state']]} **{label}** · {r['name']} — {status}{extra}"
            if r["preview"]:
                line += f"  \n&nbsp;&nbsp;&nbsp;&nbsp;_{r['preview']}_"
            lines.append(line)
        elapsed = f"전체 {now - self.t0:.0f}초"
        stage = " · ".join(running) if running else ("판정 완료" if finished == len(self.STEPS) else "준비 중")
        self.bar.progress(min(finished / len(self.STEPS), 1.0), text=f"{stage} · {elapsed}")
        self.box.markdown("  \n".join(lines))
        if self.status is not None:
            self.status.update(label=f"{stage} · {elapsed}", expanded=self.expanded)


@st.cache_resource
def shared_cooldowns() -> dict:
    """{'AI/모델': 자동 제외 해제 시각}. 브라우저 세션이 아니라 앱 서버에 두어서
    새로고침하거나 다른 기기에서 접속해도 이미 막힌 모델을 다시 호출하지 않음 (1인 사용 기준)."""
    return {}


cooldowns = shared_cooldowns()

# 세션 상태 (사이드바보다 먼저 준비)
ss = st.session_state
ss.setdefault("turns", [])        # [{"q", "files", "result"}]
ss.setdefault("conv_docs", "")
ss.setdefault("conv_images", [])
ss.setdefault("conv_files", [])
ss.setdefault("conv_id", None)    # 저장소에 기록된 현재 대화 id
ss.setdefault("conv_created", None)
ss.setdefault("img_count", 0)     # 지난 대화에서 쓴 이미지 수 (이미지는 저장하지 않음)
ss.setdefault("turn_images", [])  # 이번 질문에 실제로 보낼 이미지 (기본: 이번에 첨부한 것만)
store = ConversationStore()

NEW_CONV_KEYS = ("turns", "conv_docs", "conv_images", "conv_files", "conv_id", "conv_created",
                 "img_count", "history_cache", "pending", "turn_images", "keep_images")


def clear_attachments():
    """대화에 쌓인 문서·이미지 첨부를 지움 (이미 끝난 질문의 기록은 그대로)."""
    ss.conv_docs, ss.conv_images, ss.conv_files, ss.turn_images, ss.img_count = "", [], [], [], 0


def reset_conversation():
    for k in NEW_CONV_KEYS:
        ss.pop(k, None)


def conversation_list(force=False):
    if force or "history_cache" not in ss:
        flush_saves()
        try:
            ss.history_cache = store.list()
        except Exception as e:
            st.sidebar.error(f"대화 기록을 불러오지 못했습니다: {str(e)[:80]}")
            ss.history_cache = []
    return ss.history_cache


@st.cache_resource
def save_worker() -> dict:
    """대화 저장을 뒤에서 하나씩 처리하는 일꾼. GitHub 저장(API 여러 번 호출)이 끝나길 기다리지 않고 바로 화면을 갱신.
    한 명이 순서대로 처리하므로 목록(index.json) 쓰기가 서로 꼬이지 않음."""
    return {"pool": ThreadPoolExecutor(max_workers=1, thread_name_prefix="save"), "last": None, "errors": []}


def _save_job(conv, errors):
    try:
        store.save(conv)
    except Exception as e:                    # 화면 코드는 다른 스레드에서 못 쓰므로 기록해 두고 다음 화면에서 알림
        errors.append(f"{conv['title'][:20]}: {str(e)[:80]}")


def flush_saves(timeout=30):
    """진행 중인 저장이 끝날 때까지 기다림 (대화 열기·삭제·목록 새로고침 전에 호출)."""
    last = save_worker()["last"]
    if last is not None:
        try:
            last.result(timeout=timeout)
        except Exception:
            pass


def save_conversation():
    """현재 대화를 저장소에 기록(뒤에서). 이미지 원본은 용량 때문에 저장하지 않음."""
    if not ss.conv_id:
        ss.conv_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S-") + pysecrets.token_hex(2)
        ss.conv_created = now_text()
    monthly = {}
    by_provider = {"Claude": {}, "Gemini": {}}
    for t in ss.turns:                      # 비용 기록이 생기기 전의 질문은 at·cost가 없어 빠짐
        if t.get("at") and t["result"].get("cost"):
            monthly[t["at"][:7]] = round(monthly.get(t["at"][:7], 0) + t["result"]["cost"], 6)
        if t.get("at"):
            m = t["at"][:7]
            for n, value in result_costs(t["result"]).items():
                if n in by_provider:
                    by_provider[n][m] = round(by_provider[n].get(m, 0) + value, 6)
    conv = {"id": ss.conv_id, "title": ss.turns[0]["q"][:40], "created": ss.conv_created,
            "updated": now_text(), "turns": ss.turns, "docs": ss.conv_docs,
            "files": ss.conv_files, "images": len(ss.conv_images) + ss.img_count, "cost": monthly,
            "cost_by_provider": by_provider}
    snapshot = json.loads(json.dumps(conv, ensure_ascii=False))     # 저장 중에 화면 쪽 데이터가 바뀌어도 안전하게 복사
    worker = save_worker()
    worker["last"] = worker["pool"].submit(_save_job, snapshot, worker["errors"])
    # 사이드바 목록은 저장 완료를 기다리지 않고 바로 반영
    entry = {"id": conv["id"], "title": conv["title"], "created": conv["created"], "updated": conv["updated"],
             "turns": len(conv["turns"]), "cost": monthly, "cost_by_provider": by_provider}
    if "history_cache" in ss:
        ss.history_cache = [entry] + [c for c in ss.history_cache if c["id"] != conv["id"]]


def open_conversation(cid):
    flush_saves()
    conv = store.load(cid)
    if not conv:
        st.sidebar.error("대화를 찾을 수 없습니다.")
        return
    reset_conversation()
    ss.turns, ss.conv_docs, ss.conv_files = conv["turns"], conv.get("docs", ""), conv.get("files", [])
    ss.conv_id, ss.conv_created, ss.img_count = conv["id"], conv["created"], conv.get("images", 0)


with st.sidebar:
    avail = available_providers()
    if len(avail) < 2:
        st.error("API 키가 2개 이상 필요합니다. Secrets에 다음 중 두 개 이상을 넣어 주세요:\n\n"
                 + "\n".join(f"- `{p['key']}`" for p in PROVIDERS.values()))
        st.stop()

    if st.button("🆕 새 대화", type="primary", use_container_width=True):
        reset_conversation()
        st.rerun()
    if ss.turns:
        st.caption(f"현재 대화: 질문 {len(ss.turns)}개"
                   + (f" · 📎 {len(ss.conv_files)}개" if ss.conv_files else "")
                   + (" · 저장됨" if ss.conv_id else ""))

    # 역할 구성: 한 줄 요약 + 접힌 설정
    summary = st.empty()
    with st.expander("⚙️ 역할·모델 설정"):
        hub = st.selectbox("허브 (정리·판정)", avail, index=avail.index(pick_default("hub", avail)), key="r_hub")
        engine = {"hub": st.empty()}      # 선택칸 바로 아래에 실제 모델명 (모델이 정해진 뒤 채움)
        judge_pool = avail                # 허브와 같은 AI도 판단자 가능 (다른 모델을 씀, 예: Haiku)
        c1, c2 = st.columns(2)
        a = c1.selectbox("판단 A", judge_pool, index=judge_pool.index(pick_default("a", judge_pool)), key="r_a")
        engine["a"] = c1.empty()
        auto_b = st.toggle("🧭 판단 B 자동 선택", value=bool(secret("AUTO_JUDGE_B", True)), key="r_auto_b",
                           help="짧은 입력은 Groq → Mistral → Claude Haiku 순서. "
                                "이미지나 전체 입력 5천 자 초과 시 Haiku를 우선합니다. "
                                "무료 AI 대기는 질문 전체 최대 15초이며, 끄면 판단 B를 직접 고릅니다.")
        b_pool = [n for n in judge_pool if n != a] or judge_pool
        b = c2.selectbox("판단 B", b_pool, index=b_pool.index(pick_default("b", b_pool, taken=(a,))), key="r_b",
                         disabled=auto_b)
        engine["b"] = c2.empty()
        can_search = PROVIDERS[hub]["kind"] in ("anthropic", "gemini")
        web_search = st.toggle("🔎 웹 검색", value=can_search and bool(secret("WEB_SEARCH", True)),
                               disabled=not can_search, key="r_search",
                               help=f"허브가 필요할 때만 질문당 최대 {SEARCH_MAX_USES}회 검색합니다. "
                                    "Claude는 1,000회당 10달러가 Team 크레딧에서 차감됩니다."
                                    if can_search else f"{hub}는 웹 검색을 지원하지 않습니다.")
        clarify_on = st.toggle("❓ 되묻기", value=bool(secret("CLARIFY", True)), key="r_clarify",
                               help="토론 전에 허브가 질문을 살펴, 답이 크게 달라질 만한 모호한 점이 있으면 "
                                    "선택형으로 먼저 물어봅니다. 물을 것이 없으면 바로 진행합니다.")
        fast = st.toggle("⚡ 빠른 모드 (토론 생략)", value=False, key="r_fast",
                         help="② 상호 토론을 건너뛰어 호출이 6회에서 4회로 줄고 더 빨리 끝납니다. "
                              "대신 서로의 오류를 바로잡는 과정이 빠집니다.")
        t1, t2 = st.columns(2)
        judge_timeout = t1.selectbox("판단 시간 제한", [60, 90, 120, 180],
                                     index=[60, 90, 120, 180].index(JUDGE_TIMEOUT_DEFAULT),
                                     format_func=lambda v: f"{v}초", key="r_timeout",
                                     help="판단자가 이 시간 안에 답하지 않으면 기다리지 않고 빼고 진행합니다.")
        levels = {"low": "낮음 (빠름)", "minimal": "최소", "medium": "보통", "high": "높음 (느림)", "": "모델 기본값"}
        RUNTIME["gemini_thinking"] = t2.selectbox(
            "Gemini 생각", list(levels), format_func=levels.get, key="r_think",
            help="Gemini가 답하기 전에 생각하는 정도입니다. 낮을수록 빠르고, 높을수록 꼼꼼하지만 느립니다.") or None
        pick_models = st.checkbox("모델 직접 고르기", key="r_pick",
                                  help="끄면 Secrets에 지정한 기본 모델을 씁니다.")
        if pick_models:
            hub_m = model_picker(f"허브 · {hub}", hub, f"m_hub_{hub}")
            models = {
                "hub": hub_m,
                "a": model_picker(f"판단 A · {a}", a, f"m_a_{a}", judge_model(a, hub, hub_m)),
                "b": ("claude-haiku-5-5" if b == "Claude" else resolve_model(b)) if auto_b else
                     model_picker(f"판단 B · {b}", b, f"m_b_{b}", judge_model(b, hub, hub_m)),
            }
            if st.button("🔄 모델 목록 새로고침", use_container_width=True):
                list_models.clear()
                model_list_failures().clear()
                missing_models().clear()
                st.rerun()
        else:
            hub_m = resolve_model(hub)
            models = {"hub": hub_m, "a": judge_model(a, hub, hub_m),
                      "b": (("claude-haiku-5-5" if b == "Claude" else resolve_model(b)) if auto_b
                            else judge_model(b, hub, hub_m))}
        for role in ("hub", "a", "b"):               # 같은 AI라도 모델이 다르면 여기서 구분됨
            engine[role].caption("🧠 자동 · 짧은 입력은 무료 AI, 긴 자료·이미지는 Haiku" if role == "b" and auto_b
                                 else f"🧠 {pretty(models[role]) or '모델 없음'}", help=models[role] or None)
        for who, n in (("A", a), ("B", b)):
            if who == "B" and auto_b:
                continue
            if n == hub and models[who.lower()] == models["hub"]:
                st.caption(f"⚠️ 판단 {who}가 허브와 같은 모델이라 상호 검토 효과가 줄어듭니다.")
        steps = []
        for who, n in (("A", a), ("B", b)):
            if who == "B" and auto_b:
                continue
            low = fallbacks(n, models[who.lower()] or "")
            if low:
                steps.append(f"판단 {who}: {short(models[who.lower()])} → " + " → ".join(map(short, low)))
        if steps:
            st.caption("한도 초과 시 같은 AI의 낮은 모델로 자동 교체  \n" + "  \n".join(steps))
        for n, (_, why) in model_list_failures().items():
            if n in (a, b):
                st.caption(f"⚠️ {n} 모델 목록을 불러오지 못해 추정한 이름으로 교체합니다 ({why})")
    calls = 4 if fast else 6
    shared = hub in (a, b)                    # 허브와 같은 AI의 판단자가 있으면 모델명으로 구분
    def tag(n, m):
        return f"{n}({pretty(m)})" if shared and n == hub else n
    b_summary = "자동 판단 B" if auto_b else tag(b, models["b"])
    summary.caption(f"🔮 {tag(hub, models['hub'])} 주재 · ⚖️ {tag(a, models['a'])} vs {b_summary}"
                    + (" · 🔎" if web_search else "") + (" · ❓" if clarify_on else "")
                    + (" · ⚡" if fast else "") + f" · 호출 {calls}회")

    active = {k: t for k, t in cooldowns.items() if t > time.time()}
    if active:
        st.warning("⏸ 자동 제외 중  \n" + "  \n".join(
            f"{k.split('/', 1)[0]} {short(k.split('/', 1)[1])} · {clock(t)}까지" for k, t in active.items()))
        if st.button("제외 해제", use_container_width=True):
            cooldowns.clear()
            st.rerun()

    # 공급자별 월 누적 (구버전 기록의 cost는 Claude로 해석)
    month = datetime.now(ZoneInfo(secret("APP_TIMEZONE", "Asia/Seoul"))).strftime("%Y-%m")
    budget = float(secret("CLAUDE_MONTHLY_BUDGET", 150))
    monthly = monthly_costs(conversation_list(), month)
    for name, limit in (("Claude", budget), ("Gemini", float(secret("GEMINI_MONTHLY_BUDGET", 10)))):
        st.progress(min(monthly[name] / limit, 1.0) if limit > 0 else 0.0,
                    text=f"💰 이번 달 {name} 약 ${monthly[name]:.2f} / ${limit:.0f}")
    st.caption(f"이번 달 합계 약 ${sum(monthly.values()):.2f}")
    st.caption("이 앱에 저장된 대화 기준 추정치입니다. 과거 Gemini 사용량·중단·삭제한 대화는 빠집니다. "
               "크레딧·세금·검색 무료 허용량은 반영하지 않습니다. 실제 청구는 Claude Console·AI Studio에서 확인하세요.")

    if ss.conv_docs or ss.conv_images:
        with st.expander(f"📎 첨부 관리 ({len(ss.conv_files)}개)"):
            for name in ss.conv_files:
                st.caption(f"· {name}")
            if ss.conv_docs:
                st.caption(f"문서 {len(ss.conv_docs):,}자는 이어지는 질문에서도 계속 참고합니다.")
            if ss.conv_images:
                st.toggle(f"이전 이미지 {len(ss.conv_images)}장도 계속 보내기", key="keep_images",
                          help="끄면(기본) 이미지를 첨부한 질문에만 보냅니다. 켜면 이후 질문마다 모든 AI에 다시 보내 "
                               "비용과 시간이 늘고, 판단 B는 Haiku로 고정됩니다.")
            if st.button("첨부 해제", use_container_width=True, on_click=clear_attachments,
                         help="이 대화에 쌓인 문서·이미지를 지워 다음 질문부터 보내지 않습니다."):
                pass

    st.divider()
    st.markdown("**📚 대화 기록**")
    save_errors = save_worker()["errors"]
    while save_errors:
        st.warning(f"대화 기록 저장 실패 · {save_errors.pop(0)}")
    if not store.remote:
        st.caption("⚠️ 임시 저장 중 (앱 재시작 시 사라짐). Secrets에 `GITHUB_TOKEN`·`DATA_REPO`를 넣으면 영구 보관됩니다.")
    convs = conversation_list()
    query = st.text_input("검색", placeholder="🔍 질문 내용으로 찾기", label_visibility="collapsed")
    shown = [c for c in convs if query.strip().lower() in c["title"].lower()] if query.strip() else convs
    if shown:
        labels = {c["id"]: f"{c['updated'][5:]} · {c['title']} ({c['turns']})" for c in shown}
        current = ss.conv_id if ss.conv_id in labels else None
        pick = st.selectbox("지난 대화", list(labels), format_func=labels.get, label_visibility="collapsed",
                            index=list(labels).index(current) if current else 0)
        c1, c2, c3 = st.columns(3)
        if c1.button("📂", help="열기", use_container_width=True, disabled=pick == ss.conv_id):
            open_conversation(pick)
            st.rerun()
        if c2.button("🗑", help="삭제", use_container_width=True):
            ss.confirm_delete = pick
        if c3.button("🔄", help="목록 새로고침", use_container_width=True):
            conversation_list(force=True)
            st.rerun()
        if ss.get("confirm_delete") == pick:
            st.warning("이 대화를 지울까요? 되돌릴 수 없습니다.")
            d1, d2 = st.columns(2)
            if d1.button("삭제", type="primary", use_container_width=True):
                try:
                    flush_saves()
                    store.delete(pick)
                    if pick == ss.conv_id:
                        reset_conversation()
                except Exception as e:
                    st.error(f"삭제 실패: {str(e)[:80]}")
                ss.pop("confirm_delete", None)
                ss.pop("history_cache", None)
                st.rerun()
            if d2.button("취소", use_container_width=True):
                ss.pop("confirm_delete", None)
                st.rerun()
    else:
        st.caption("검색 결과가 없습니다." if query.strip() else "아직 저장된 대화가 없습니다.")

roles = {"hub": hub, "a": a, "b": b}

st.title(f"🔮 {APP_NAME}")
st.caption("허브 AI가 질문을 정리하면 두 AI가 토론하고, 허브가 최종 결론을 판정합니다. "
           "후속 질문은 직전 최종안을 기준으로 이어집니다.")


def label(item):
    return f"{item['role']} · {item['name']}{'' if item['ok'] else ' ❌'}"


INFO_MARKS = ("🧭", "🔎", "💰")               # 일상 안내: 회색 글씨 / 그 밖(⚠️ ⌛ ✂️ 등)은 노란 경고


def show_notice(n):
    if n.startswith(INFO_MARKS):
        st.caption(n)
    else:
        st.warning(n)


def render_result(res):
    for n in res.get("notices", []):
        show_notice(n)
    final = res["final"]
    if final and final["ok"]:
        major, minor = split_report(final["text"])
        st.markdown("##### 📜 Majority Report")
        st.markdown(major)
        st.caption(f"최종 판정: {final['name']} · {final['sec']}초"
                   + (f" · 웹 검색 {res['searches']}회" if res.get("searches") else "")
                   + (" · 💰 " + cost_caption(result_costs(res)) if res.get("costs") or res.get("cost") else ""))
        if "costs" not in res:
            st.caption("이전 기록: Gemini 비용은 수집되지 않았습니다.")
        if res.get("sources"):
            with st.expander(f"🔎 허브가 확인한 출처 ({len(res['sources'])}개)"):
                for src in res["sources"]:
                    st.markdown(f"- [{src['title']}]({src['url']})")
        if minor:
            with st.container(border=True):
                st.markdown("**🔍 Minority Report** — 토론의 쟁점과 판정 근거")
                st.markdown(minor)
    else:
        st.error(f"최종 판정 단계 실패: {final['text'] if final else '알 수 없는 오류'}")
    items = [res["brief"]] + res["firsts"] + res["debates"]
    with st.expander("토론 과정 보기 (허브 정리 · 1차 판단 · 상호 토론)"):
        tabs = st.tabs([label(i) for i in items])
        for tab, item in zip(tabs, items):
            with tab:
                st.caption(f"{item['model']} · {item['sec']}초")
                st.markdown(item["text"])


def final_text(res):
    f = res["final"]
    return split_report(f["text"])[0] if f and f["ok"] else ""


def show_question(question, files, answers=None):
    """사용자 질문 말풍선 (되묻기에 답한 내용 포함)."""
    with st.chat_message("user"):
        st.markdown(question)
        for name in files:
            st.caption(f"📎 {name}")
        for a in answers or []:
            st.caption(f"❓ {a['q']} → **{a['a']}**")


def history_pairs():
    return [(with_clarification(t["q"], t.get("clarify")), final_text(t["result"]))
            for t in ss.turns if final_text(t["result"])]


def make_input(question, answers=()):
    # 이미지는 이번 질문에 보낼 것만 (질문마다 모든 AI에 다시 보내면 비용·시간이 늘고 판단 B가 Haiku로 고정됨)
    return UserInput(prompt=with_clarification(question, answers), doc_text=ss.conv_docs,
                     images=ss.turn_images, history=history_pairs())


def mark_stopped(question):
    """⏹ 중단 버튼: 버튼을 누르면 Streamlit이 화면을 다시 그리며 진행 중인 토론이 즉시 취소됨."""
    ss.stopped_q = question


def run_turn(question, files, warns, answers, extra_cost=0.0, extra_costs=None):
    """토론 실행 → 결과 표시 → 저장. extra_cost: 되묻기 단계에서 쓴 비용."""
    ui = make_input(question, answers)
    extra_costs = extra_costs if extra_costs is not None else {roles["hub"]: extra_cost}
    display_roles, display_models = dict(roles), dict(models)
    if auto_b:
        route = judge_b_plan(ui, roles, models, cooldowns)
        if route["candidates"]:
            display_roles["b"], display_models["b"] = route["candidates"][0]
    with st.chat_message("assistant", avatar="🔮"):
        for w in warns:
            st.warning(w)
        st.button("⏹ 중단", key="stop_run", on_click=mark_stopped, args=(question,),
                  help="오타나 잘못된 질문이면 눌러서 바로 멈추세요. 고친 질문을 입력창에 바로 보내도 이전 질문은 멈춥니다.")
        status = st.status("허브가 토론을 주재하는 중…", expanded=True)
        with status:
            board = ProgressBoard(display_roles, status, display_models, extra_costs)
            board.update_cost({})
        board.live = st.empty()           # 최종 판정이 실시간으로 나타날 자리
        with status:
            try:
                job = run_pipeline(ui, roles, models, board.note, cooldowns, web_search,
                                   board=board, fast=fast, judge_timeout=judge_timeout, auto_judge_b=auto_b)
                result = asyncio.run(run_with_ticker(job, board.render))
                board.render()
                board.live.empty()            # 실시간 초안을 지우고 완성본으로 다시 그림
                status.update(label=f"판정 완료 · {time.time() - board.t0:.0f}초", state="complete",
                              expanded=False)
            except Exception as e:
                result = None
                status.update(label="실패", state="error")
                st.error(str(e))
        if result:
            result["cost"] = round(result.get("cost", 0) + extra_cost, 6)
            for n, value in extra_costs.items():
                result["costs"][n] = round(result["costs"].get(n, 0) + value, 6)
            render_result(result)
            ss.turn_images = []               # 다음 질문에는 새로 첨부한 이미지만
            ss.turns.append({"q": question, "clarify": list(answers), "files": files, "result": result,
                             "at": now_text()})
            try:
                save_conversation()               # 뒤에서 저장하므로 바로 넘어감 (실패는 다음 화면 사이드바에 표시)
            except Exception as e:
                st.warning(f"대화 기록 저장을 시작하지 못했습니다 ({str(e)[:80]}). 답변은 이 화면에 그대로 남습니다.")
                time.sleep(3)
            st.rerun()


# 지난 대화 다시 그리기
if ss.conv_id and ss.img_count and not ss.conv_images:
    st.info(f"이 대화에서 쓴 이미지 {ss.img_count}장은 기록에 저장되지 않습니다. "
            "이어서 이미지에 대해 물어보려면 다시 첨부해 주세요. 문서 내용은 그대로 이어집니다.")
def rate_turn(i, key):
    """👍/👎 평가를 그 질문 기록에 남기고 저장 (나중에 설정 변경의 효과를 비교하는 근거)."""
    v = ss.get(key)
    ss.turns[i]["rating"] = {1: "up", 0: "down"}.get(v)
    try:
        save_conversation()
    except Exception as e:
        st.toast(f"평가 저장 실패: {str(e)[:60]}")


for i, t in enumerate(ss.turns):
    show_question(t["q"], t["files"], t.get("clarify"))
    with st.chat_message("assistant", avatar="🔮"):
        render_result(t["result"])
        if t["result"].get("final", {}) and t["result"]["final"].get("ok"):
            key = f"rate_{ss.conv_id}_{i}"
            st.feedback("thumbs", key=key, default={"up": 1, "down": 0}.get(t.get("rating")),
                        on_change=rate_turn, args=(i, key))

stopped = ss.pop("stopped_q", None)
if stopped:
    st.info(f"⏹ 질문을 중단했습니다. 고쳐서 다시 보내 주세요.\n\n> {stopped}")

# 명시적 하단 배치: 루트 st.chat_input의 페이지 자동 스크롤과
# 긴 답변·expander 높이 변화가 충돌하지 않게, 일반 페이지 스크롤을 유지.
msg = st.bottom.chat_input(
    "질문을 입력하세요. 직전 결론에 대한 후속 질문도 됩니다. (📎로 PDF·TXT·MD·이미지 첨부)",
    accept_file="multiple", file_type=["pdf", "txt", "md", *IMAGE_MIME.keys()],
)

if msg and (msg.text.strip() or msg.files):
    ss.pop("pending", None)               # 되묻는 중에 새 질문을 보내면 이전 확인은 버림
    question = msg.text.strip() or "첨부한 파일의 내용을 정리해 주세요."
    new_names = [f.name for f in msg.files]
    show_question(question, new_names)

    # 새 첨부를 대화 전체 첨부에 누적
    doc_text, images, warns = process_uploads(msg.files)
    if doc_text:
        merged = (ss.conv_docs + "\n\n" + doc_text).strip()
        if len(merged) > MAX_DOC_CHARS:
            merged = merged[-MAX_DOC_CHARS:]
            warns.append(f"대화에 첨부된 문서가 길어 최근 {MAX_DOC_CHARS:,}자만 사용합니다.")
        ss.conv_docs = merged
    earlier = list(ss.conv_images) if ss.get("keep_images") else []
    ss.turn_images = earlier + images      # 기본은 이번에 첨부한 이미지만, 켜 두면 이전 이미지도
    ss.conv_images += images
    ss.conv_files += [n for n in new_names if n not in ss.conv_files]

    if None in models.values():
        st.error("모델이 선택되지 않은 역할이 있습니다. 사이드바의 모델 선택을 확인하세요.")
        st.stop()

    questions, clarify_cost, clarify_costs, clarify_notes = [], 0.0, {}, []
    if clarify_on:
        holder = st.empty()               # 확인이 끝나면 말풍선째 지워서 빈 칸이 남지 않게 함
        with holder.container():
            with st.chat_message("assistant", avatar="🔮"):
                with st.spinner("허브가 질문을 확인하는 중…"):
                    questions, clarify_cost = asyncio.run(
                        hub_clarify(make_input(question), hub, models["hub"], cooldowns,
                                    log=clarify_notes.append, cost_details=clarify_costs))
        holder.empty()
    warns.extend(clarify_notes)
    if questions:                         # 화면을 다시 그려 선택형 질문을 보여줌
        ss.pending = {"id": pysecrets.token_hex(3), "q": question, "files": new_names,
                      "warns": warns, "questions": questions, "cost": clarify_cost, "costs": clarify_costs}
        st.rerun()
    run_turn(question, new_names, warns, [], clarify_cost, clarify_costs)

elif ss.get("pending"):
    pend = ss.pending
    show_question(pend["q"], pend["files"])
    slot = st.empty()
    with slot.container():
        with st.chat_message("assistant", avatar="🔮"):
            st.markdown("**❓ 토론 전에 확인할 점이 있습니다**")
            st.caption("고른 내용에 맞춰 두 AI가 토론합니다. 고르지 않은 질문은 허브가 합리적으로 가정합니다.")
            with st.form(f"clarify_{pend['id']}", border=True):
                picks = []
                for i, q in enumerate(pend["questions"]):
                    key = f"cq_{pend['id']}_{i}"
                    if q["multi"]:
                        picks.append(st.multiselect(q["q"], q["options"], key=key, placeholder="여러 개 선택 가능"))
                    else:
                        picks.append(st.radio(q["q"], q["options"], index=None, key=key))
                extra = st.text_input("기타·추가 설명 (선택)", key=f"cq_{pend['id']}_extra",
                                      placeholder="선택지에 없는 내용이나 덧붙일 조건을 적어 주세요")
                c1, c2 = st.columns(2)
                go = c1.form_submit_button("답하고 토론 시작", type="primary", use_container_width=True)
                skip = c2.form_submit_button("건너뛰고 그대로 진행", use_container_width=True)
    if go or skip:
        answers = []
        if go:
            for q, v in zip(pend["questions"], picks):
                v = ", ".join(v) if isinstance(v, list) else v
                if v:
                    answers.append({"q": q["q"], "a": v})
            if extra.strip():
                answers.append({"q": "추가 설명", "a": extra.strip()})
        ss.pop("pending", None)
        # 양식 자리는 비워 두기만 함. 같은 자리에 다른 말풍선을 바로 쓰면 Streamlit이 '비우기'를 생략해
        # 제출한 양식이 토론 내내 흐리게 남고, 그 버튼을 다시 누르면 진행 중인 토론이 끊김.
        slot.empty()
        if answers:                       # 고른 답은 그 아래 새 자리에 표시
            with st.chat_message("user"):
                for a in answers:
                    st.caption(f"❓ {a['q']} → **{a['a']}**")
        run_turn(pend["q"], pend["files"], pend["warns"], answers, pend.get("cost", 0.0), pend.get("costs"))
