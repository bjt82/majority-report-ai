"""모델 목록 조회, 한도 초과 시 내려갈 낮은 모델, 판단자 기본 모델, 모델 이름 표시."""
import time
import streamlit as st
from anthropic import Anthropic, AsyncAnthropic
from google import genai
from openai import AsyncOpenAI, OpenAI
from majority.settings import *  # noqa: F401,F403


@st.cache_data(ttl=3600, show_spinner=False)
def list_models(provider: str, api_key: str):
    p = PROVIDERS[provider]
    if p["kind"] == "anthropic":
        return [(m.id, m.display_name) for m in Anthropic(api_key=api_key).models.list(limit=100)]
    if p["kind"] == "gemini":
        out = []
        for m in genai.Client(api_key=api_key).models.list():
            mid = (m.name or "").removeprefix("models/")
            if (mid.startswith("gemini") and "generateContent" in (m.supported_actions or [])
                    and not any(k in mid for k in GEMINI_EXCLUDE)):
                out.append((mid, m.display_name or mid))
        return sorted(out, key=lambda x: x[0], reverse=True)
    # OpenAI 호환
    out = []
    for m in OpenAI(api_key=api_key, base_url=p["base_url"]).models.list():
        mid = m.id
        if any(k in mid.lower() for k in p.get("exclude", ())):
            continue
        if p.get("only_suffix") and not mid.endswith(p["only_suffix"]):
            continue
        out.append((mid, getattr(m, "name", None) or mid))
    return sorted(out, key=lambda x: x[0])


MODEL_LIST_RETRY_SEC = 60                    # 모델 목록 조회 실패 후 다시 시도하기까지


@st.cache_resource
def model_list_failures() -> dict:
    """{AI 이름: (목록 조회 실패 시각, 이유)}. 실패를 잠깐만 기억해서 매번 느려지지 않게 하되,
    일시적인 서버 과부하로 실패한 목록이 오래 비어 있지 않게 함. 이유는 설정 화면에 표시."""
    return {}


MISSING_MODEL_SEC = 24 * 3600                # '모델 없음(404)'으로 확인된 모델을 다시 시도하지 않는 시간


@st.cache_resource
def missing_models() -> dict:
    """{'AI/모델': 확인 시각}. 없는 모델을 매 질문마다 다시 호출하지 않도록 기억."""
    return {}


def is_missing(provider: str, model: str) -> bool:
    return time.time() - missing_models().get(f"{provider}/{model}", 0) < MISSING_MODEL_SEC


def model_ids(provider: str):
    """그 AI에서 쓸 수 있는 모델 id 목록. 조회 실패면 None (이때는 설정한 목록을 그대로 믿음).
    성공한 목록은 list_models가 1시간 기억하고, 실패는 1분만 기억함."""
    fails = model_list_failures()
    if time.time() - fails.get(provider, (0, ""))[0] < MODEL_LIST_RETRY_SEC:
        return None
    try:
        ids = [i for i, _ in list_models(provider, secret(PROVIDERS[provider]["key"]))]
    except Exception as e:
        fails[provider] = (time.time(), f"{type(e).__name__}: {str(e)[:120]}")
        return None
    if not ids:                                   # 받았지만 조건에 맞는 모델이 없음 → 원인 확인용으로 기록
        fails[provider] = (time.time(), "목록은 받았지만 대화용 모델이 0개로 걸러짐")
        list_models.clear()
        return None
    fails.pop(provider, None)
    return ids


def guess_lower(provider: str, primary: str) -> list:
    """모델 목록을 못 불러왔을 때의 낮은 모델: 이름으로 추정한 것(예: gemini-3.8-flash → gemini-3.8-flash-lite)
    + 그 AI의 고정 별칭. 없는 모델이면 '모델 없음'으로 기억되고 다음 후보로 넘어가므로 안전함."""
    out = []
    if "flash" in primary and "lite" not in primary:
        out.append(primary.replace("flash", "flash-lite", 1))
    elif "pro" in primary:
        out.append(primary.replace("pro", "flash", 1))
    return out + list(PROVIDERS[provider].get("aliases", ()))


def fallbacks(provider: str, primary: str) -> list:
    """primary가 한도 초과일 때 차례로 쓸 같은 AI의 낮은 모델들."""
    p = PROVIDERS[provider]
    conf = secret(p["key"].replace("_API_KEY", "_FALLBACK"), p.get("fallback", ()))
    if isinstance(conf, str):
        conf = "auto" if conf.strip() == "auto" else [m.strip() for m in conf.split(",") if m.strip()]
    ids = model_ids(provider)
    if conf == "auto":
        if not ids:                                   # 목록 조회 실패: 이름 추정 + 고정 별칭
            conf = guess_lower(provider, primary or "")
        else:
            lite = sorted((i for i in ids if "flash-lite" in i), reverse=True)
            flash = sorted((i for i in ids if "flash" in i and "lite" not in i), reverse=True)
            conf = lite + flash
    out = [m for m in conf if m and m != primary and (ids is None or m in ids) and not is_missing(provider, m)]
    return list(dict.fromkeys(out))[:FALLBACK_MAX]


def chain(provider: str, primary) -> list:
    """[기본 모델, 낮은 모델1, 낮은 모델2, ...]"""
    return ([primary] if primary else []) + fallbacks(provider, primary or "")


def resolve_model(name: str):
    """대타로 투입할 AI의 모델: Secrets 기본값, 없으면 목록의 첫 모델."""
    default = PROVIDERS[name]["model"]
    if default:
        return default
    try:
        return list_models(name, secret(PROVIDERS[name]["key"]))[0][0]
    except Exception:
        return None


def judge_model(name: str, hub: str, hub_model):
    """판단자의 기본 모델. 허브와 같은 AI면 허브와 다른 낮은 모델(예: Sonnet 허브 → Haiku 판단자)."""
    if name == hub:
        low = fallbacks(name, hub_model or "")
        if low:
            return low[0]
    return resolve_model(name)


def short(model: str) -> str:
    return (model or "").rsplit("/", 1)[-1]


def pretty(model: str) -> str:
    """사람이 읽기 쉬운 모델 이름. claude-sonnet-5-5 → Sonnet 5.5, gemini-3.8-flash-lite → Gemini 3.8 Flash-Lite."""
    m = short(model)
    if m.startswith("claude-"):
        parts = m.removeprefix("claude-").split("-")
        words = [x for x in parts if not x.isdigit()]
        nums = [x for x in parts if x.isdigit()]
        return " ".join(w.capitalize() for w in words) + (f" {'.'.join(nums)}" if nums else "")
    if m.startswith("gemini-"):
        rest = m.removeprefix("gemini-").replace("flash-lite", "Flash-Lite").replace("-latest", " (최신)")
        return "Gemini " + " ".join(w if not w.isalpha() or w[0].isupper() else w.capitalize()
                                    for w in rest.split("-"))
    return m
