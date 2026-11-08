"""설정값, Secrets 읽기, AI 공급자 등록부."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import streamlit as st
from openai import AsyncOpenAI, OpenAI


APP_NAME = "Majority Report AI"


MINORITY_MARKER = "<<<MINORITY_REPORT>>>"   # 최종 결론과 소수 의견을 나누는 구분자


MAX_DOC_CHARS = 60_000                       # 문서 텍스트 최대 길이(토큰 비용 보호)


MAX_FILE_MB = 10


CALL_TIMEOUT_SEC = 180                       # 허브 정리·최종 판정 한 번의 최대 대기


JUDGE_TIMEOUT_DEFAULT = 90                   # 판단자 한 단계의 기본 시간 제한 (화면에서 변경)

# 판단 B 자동 선택: 짧은 텍스트는 Groq → Mistral → Haiku, 긴 자료·이미지는 Haiku.
AUTO_B_INPUT_CHARS = 5000
AUTO_B_FREE_BUDGET_SEC = 15.0                # 질문 전체(1차·반론 합산)의 무료 후보 대기 상한
AUTO_B_FREE_CALL_SEC = 7.5                   # 한 무료 후보에 쓰는 최대 시간


RUNTIME = {"gemini_thinking": "low"}         # 화면 설정을 호출부에 전달 (Gemini 생각 수준)


RETRY_WAIT_SEC = 20                          # 요청 한도(429)·과부하 시 한 번 재시도 전 대기 (허브)


QUICK_RETRY_SEC = 10                         # 판단자는 이보다 오래 기다려야 하면 재시도 대신 다른 모델로 교체


FALLBACK_MAX = 3                             # 한 AI에서 한도 초과 시 내려갈 낮은 모델 최대 개수


CLAUDE_MAX_TOKENS = 16_000                   # 한 번 응답의 최대 출력 (생각 토큰 포함)


CLAUDE_MAX_TOKENS_STREAM = 32_000            # 스트리밍(최종 판정)은 시간 제한 걱정이 없어 더 넉넉히


SEARCH_MAX_USES = 3                          # 허브가 질문 하나에 웹 검색할 수 있는 최대 횟수


CLAUDE_SEARCH_TOOLS = ("web_search_20260318", "web_search_20250305")   # 최신 → 예전 버전 순으로 시도

CLAUDE_FETCH_TOOLS = ("web_fetch_20260309", "web_fetch_20250910")
FETCH_MAX_USES = 3
FETCH_MAX_CONTENT_TOKENS = 12_000             # 페이지 본문이 질문당 입력을 과도하게 늘리지 않도록 제한


COOLDOWN_MIN_SEC = 10 * 60                   # 분당 한도 초과 시 그 모델을 자동 제외하는 시간 (대기 시간을 모를 때)


COOLDOWN_DAY_SEC = 60 * 60                   # 일일 한도 초과로 보일 때 그 모델을 자동 제외하는 시간


CLARIFY_TIMEOUT_SEC = 30                     # 허브가 되물을 것을 판단하는 단계의 최대 대기


CLARIFY_MAX_Q = 3                            # 한 번에 되묻는 질문 최대 개수


CLARIFY_MAX_OPTS = 5                         # 질문 하나의 선택지 최대 개수


MAX_HISTORY_TURNS = 6                        # 후속 질문 때 함께 보낼 이전 대화 수


LATEST_ANSWER_CHARS = 12_000                 # 직전 최종안은 거의 전부 전달


OLDER_ANSWER_CHARS = 3_000                   # 그 이전 최종안은 요약 분량만


# Claude API 요금 (100만 토큰당 달러: 입력, 출력). 생각(thinking) 토큰은 출력에 포함되어 청구됨.
# 표에 없는 모델은 비용을 계산하지 않음. Haiku 5.5는 입력 10만 토큰 초과 시 더 비싸지만 이 앱에선 드묾.
CLAUDE_PRICES = {
    "claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5": (5.00, 25.00), "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00), "claude-haiku-4-5": (1.00, 5.00),
}


WEB_SEARCH_PRICE = 10 / 1000                 # Claude 웹 검색 1회 (1,000회당 10달러)

# Gemini Developer API Standard 유료 단가 (USD/100만 토큰: 입력, 출력+생각, 캐시 입력).
# https://ai.google.dev/gemini-api/docs/pricing (2026-11-08 확인)
# 3.8 Flash의 출시 할인은 2026-12-31까지. 별칭은 API가 반환한 실제 모델로 계산.
GEMINI_PRICES = {
    "gemini-3.8-flash": (0.75, 3.75, 0.075),
    "gemini-3.5-flash": (1.50, 9.00, 0.15),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
}


IMAGE_MIME = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "webp": "image/webp", "gif": "image/gif",
}


def secret(key: str, default=None):
    """secrets.toml이 없어도 앱이 죽지 않도록 안전하게 읽기."""
    try:
        return st.secrets[key]
    except Exception:
        return default


# Claude 생각 수준(effort): low < medium < high. 낮을수록 빠르고 저렴, 높을수록 꼼꼼함. None이면 모델 기본값
# (Sonnet 5.5 기본 high, Haiku 5.5 기본 medium). Secrets의 CLAUDE_EFFORT_BRIEF 등으로 바꿀 수 있음.
CLAUDE_EFFORT = {
    "clarify": secret("CLAUDE_EFFORT_CLARIFY", "low"),      # 되물을 것이 있는지만 판단
    "brief": secret("CLAUDE_EFFORT_BRIEF", "medium"),       # 질문 정리·웹 검색 (가장 오래 걸리던 단계)
    "verdict": secret("CLAUDE_EFFORT_VERDICT", "high"),     # 최종 판정은 품질 우선
}


# kind: anthropic / gemini / openai(OpenAI 호환 API)
# vision: 이미지를 직접 받을 수 있는지. False면 허브가 정리한 이미지 설명으로 대신함
# fallback: 기본 모델이 한도 초과일 때 차례로 내려갈 낮은 모델. API 모델 목록에 있는 것만 사용하며,
#           Secrets의 <접두어>_FALLBACK (예: GEMINI_FALLBACK = ["모델1", "모델2"])로 바꿀 수 있음.
#           "auto"는 모델 목록에서 자동으로 고름 (Gemini: flash-lite → 다른 flash 순)
PROVIDERS = {
    "Claude": dict(kind="anthropic", key="ANTHROPIC_API_KEY", vision=True,
                   model=secret("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                   fallback=("claude-haiku-5-5",)),
    "Gemini": dict(kind="gemini", key="GEMINI_API_KEY", vision=True,
                   model=secret("GEMINI_MODEL", "gemini-3.8-flash"),
                   fallback="auto",
                   # 모델 목록을 못 불러올 때도 쓸 수 있는 Google의 고정 별칭 (항상 최신 flash-lite / flash를 가리킴)
                   aliases=("gemini-flash-lite-latest", "gemini-flash-latest")),
    "Groq": dict(kind="openai", key="GROQ_API_KEY", vision=False,
                 base_url="https://api.groq.com/openai/v1",
                 # 무료 등급은 분당 토큰이 작아(gpt-oss-120b 8K) 요청 하나가 그 안에 들어가야 함
                 max_tokens=int(secret("GROQ_MAX_TOKENS", 2048)),
                 max_input_chars=int(secret("GROQ_MAX_INPUT_CHARS", 5000)),
                 model=secret("GROQ_MODEL", "openai/gpt-oss-120b"),
                 fallback=("openai/gpt-oss-20b", "llama-3.3-70b-versatile", "qwen/qwen3-32b"),
                 exclude=("whisper", "tts", "guard", "playai", "orpheus", "distil")),
    "Mistral": dict(kind="openai", key="MISTRAL_API_KEY", vision=False,
                    base_url="https://api.mistral.ai/v1",
                    model=secret("MISTRAL_MODEL", "mistral-small-latest"),
                    fallback=("ministral-8b-latest",),
                    exclude=("embed", "ocr", "moderation", "transcribe", "voxtral")),
    "OpenRouter": dict(kind="openai", key="OPENROUTER_API_KEY", vision=False,
                       base_url="https://openrouter.ai/api/v1",
                       model=secret("OPENROUTER_MODEL", ""),
                       only_suffix=":free"),          # 무료 모델만 표시
}


# 기본 배치: 허브 Claude(Sonnet) · 판단 A Gemini(무료) · 판단 B Claude(Haiku, 한도에 걸리지 않는 고정 판단자)
DEFAULT_ROLES = {"hub": ["Claude", "Gemini"],
                 "a": ["Gemini", "Mistral", "OpenRouter", "Groq", "Claude"],
                 "b": ["Claude", "Groq", "Mistral", "OpenRouter", "Gemini"]}


GEMINI_EXCLUDE = ("embedding", "tts", "image", "audio", "live", "robotics",
                  "computer-use", "aqa", "cyber", "veo", "imagen", "lyria")


def today() -> str:
    return datetime.now(ZoneInfo(secret("APP_TIMEZONE", "Asia/Seoul"))).strftime("%Y-%m-%d")


def available_providers():
    return [n for n, p in PROVIDERS.items() if secret(p["key"])]
