# Majority Report AI — 인계 문서

> **GitHub 기록에 남지 않는 정보만** 모은 문서입니다.
> - 무엇을 왜 바꿨는지: [병합된 PR 목록](https://github.com/bjt82/majority-report-ai/pulls?q=is%3Apr+is%3Amerged) (PR 설명에 변경 이유와 시험 결과가 있음)
> - 기능·설정·파일 구조: [`README.md`](../README.md), [`secrets.toml.example`](../secrets.toml.example)
> - 작업 규칙: [`AGENTS.md`](../AGENTS.md)
>
> 사용자 환경·원칙·남은 과제가 바뀌면 이 문서를 고쳐 주세요. API 키·비밀번호·결제 계정 ID·카드 정보는 적지 않습니다.

## 1. 한 줄 요약
허브 AI가 질문을 정리하면 두 판단자가 답하고 서로 반박한 뒤, 허브가 최종 판정(Majority/Minority Report)을 내리는 **1인용 Streamlit 앱**입니다. Streamlit Cloud에서 `main` 브랜치(`app.py`)로 배포됩니다.

| 역할 | 기본 모델 |
|---|---|
| 허브 | Claude Sonnet 5.5 |
| 판단 A | Gemini 3.8 Flash |
| 판단 B | 자동 선택: 짧은 입력은 Groq → Mistral → Haiku 5.5, 긴 입력·이미지는 Haiku 5.5 |

## 2. 사용자 환경·결제 (GitHub에 없음)
| 서비스 | 상태 |
|---|---|
| Claude API | Team 계정으로 **매달 약 150달러** 사용 가능 |
| Gemini API | **유료 Tier 1.** Google AI Pro 구독 혜택으로 **매달 약 10달러 Google Cloud 크레딧** 수령. AI Studio는 **선불** 방식 |
| Groq, Mistral 등 | 무료 등급 (판단 B 후보·대타) |
| ChatGPT | Team 구독. **구독에는 API가 포함되지 않음** (API는 platform.openai.com에서 별도 결제) |
| Perplexity | Pro 구독. Pro의 월 API 크레딧 혜택은 2026년 2월 종료된 것으로 보임 (미확인) |
| 대화 기록 | 비공개 저장소 `bjt82/majority-report-data`에 저장 (`GITHUB_TOKEN`, `DATA_REPO`) |

**Gemini 결제에서 겪은 일 (재발 시 참고)**
- Google Cloud 결제 계정이 둘 있었고, 크레딧은 "내 결제 계정"에 들어옴 → API 키 프로젝트를 그쪽으로 옮김.
- 무료 체험 계정이라 무료 등급으로 내려갔다가 **결제 활성화(정식 전환)**로 Tier 1 복귀.
- 모든 Gemini 모델이 "한도 초과"로 보인 원인은 **AI Studio 선불 크레딧 잔액 ₩0**. 최소 금액을 충전해 해결 (월 GCP 크레딧이 먼저 차감됨).
- 무료 체험 크레딧(약 30만 원)은 Gemini API에 적용되지 않는 것으로 보임.
- 권장: Cloud 결제 계정에 월 ₩15,000 정도 예산 알림, AI Studio 자동 충전은 끄기.

## 3. 정한 원칙
- 사용자는 **혼자** 씁니다. 설명은 **한국어로, 단계별로**.
- 작업은 **브랜치 → PR → CI 통과 → 병합**. PR은 묻지 않고 만들고, 사용자가 허용한 세션에서는 CI 통과 후 바로 병합합니다. 강제 푸시·삭제·결제 설정 변경은 먼저 묻습니다.
- 비용보다 **상호 검토의 질과 안정성**을 우선하되, Claude 월 150달러·Gemini 월 10달러 안에서 씁니다.
- 모델 이름(`claude-sonnet-5-5`, `gemini-3.8-flash` 등)은 2026년 기준 실제 이름입니다. 학습 데이터에 없다고 바꾸지 마세요.
- `APP_PASSWORD`는 필수입니다 (없으면 앱이 열리지 않음).

## 4. 남은 과제
| 우선순위 | 과제 | 메모 |
|---|---|---|
| 확인 | 배포 후 점검 | 이미지는 첨부한 질문에만 가는지(다음 질문에서 판단 B가 무료 AI로 가는지), 👍/👎 저장, 긴 답변 스크롤, 사이드바 비용과 실제 청구액(Claude Console·AI Studio) 비교 |
| 확인 | 날짜별 정보의 본문 조회 검증 | 배정고 11월 9일 석식 질문에서 검색의 8월 발췌만 읽은 사례가 있음. Claude 허브에 본문 조회와 날짜 검증 지침을 추가. 병합·배포 후 검색 켜고 같은 질문·URL로 확인 필요. JS 화면은 여전히 직접 조회가 안 될 수 있으며, 확인 실패를 미게시·급식 없음으로 단정하면 안 됨 |
| 중 | 평가 데이터 활용 | 👍/👎가 쌓이면 판단 B 경로·생각 수준별로 비교 (데이터 저장소의 `rating`) |
| 중 | 단계별 비용 기록 | 지금은 공급자별 합계만. 정리·판단·판정 단계별로 나누면 절감 포인트가 보임 |
| 중 | 웹 검색 동적 필터링 실험 | 지금은 `allowed_callers: ["direct"]`로 꺼 둠. 켜면 토큰·정확도가 나아질 수 있으나 느려질 수 있음 |
| 중 | 되묻기 빈도 조정 | 너무 자주·드물게 묻는지 사용자 피드백 대기 |
| 낮음 | 합쳐진 작업 브랜치 정리 | 오래된 `claude/…`, `feat/…` 브랜치 |

## 5. 증상별 대응 (환경 쪽)
| 증상 | 확인할 것 |
|---|---|
| Gemini가 모든 모델에서 "사용량 한도 초과" | AI Studio 결제의 **선불 잔액**(0이면 충전), 키가 Tier 1인지, Secrets의 키, 사이드바 **제외 해제** |
| 앱이 안 열리고 비밀번호 안내가 뜸 | Secrets에 `APP_PASSWORD` 추가 |
| Secrets의 모델 설정을 바꿨는데 그대로 | Streamlit Cloud **Reboot app** (설정은 앱 시작 시 한 번 읽음) |
| 대화 기록 저장 실패 경고 | `GITHUB_TOKEN` 권한(데이터 저장소 Contents 읽기·쓰기)과 만료 |
| 짧은 질문인데 판단 B가 Haiku | 이미지가 있거나, 같은 대화의 이전 대화·문서까지 합쳐 5,000자 초과 |
| 비용이 예상보다 큼 | `CLAUDE_EFFORT_BRIEF` 낮추기, Gemini가 정상인지(대타로 Claude가 쓰이는지) |
