# 🔮 Majority Report AI

Claude와 Gemini가 각자 답하고, 서로 검토하고, 하나의 답으로 합의하는 상호 검토형 AI 허브입니다.

## 작동 방식
1. **병렬 초안**: Claude와 Gemini가 같은 질문에 독립적으로 동시에 답합니다.
2. **교차 검토**: Claude는 Gemini의 답을, Gemini는 Claude의 답을 동시에 검토해 오류·누락·장점을 찾습니다.
3. **최종 합의**: 사이드바에서 고른 심판 모델(Claude 또는 Gemini)이 초안과 검토를 종합해 **Majority Report**(최종 답변)와 **Minority Report**(이견·교정 내역)를 냅니다.

텍스트 질문과 함께 PDF, TXT, MD 문서와 이미지(PNG, JPG, WEBP, GIF)를 첨부할 수 있습니다.

## 실행
```bash
pip install -r requirements.txt
mkdir -p .streamlit && cp secrets.toml.example .streamlit/secrets.toml   # 실제 키로 수정
streamlit run app.py
```

## 배포
Streamlit Community Cloud에서 이 저장소를 연결하고, Settings → Secrets에 `secrets.toml.example` 형식으로 키를 입력합니다.

> ⚠️ API 키는 절대 커밋하지 마세요. 요청 1회당 API 호출이 5회 발생하므로 `APP_PASSWORD`를 꼭 설정하세요.
