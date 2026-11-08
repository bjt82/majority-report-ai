"""대화 기록 저장소 (GitHub 비공개 저장소 또는 로컬 data/ 폴더)."""
import base64
import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from majority.settings import *  # noqa: F401,F403


class ConversationStore:
    """대화 기록 저장소.
    - GITHUB_TOKEN + DATA_REPO가 있으면: 비공개 GitHub 저장소에 JSON으로 영구 저장 (무료)
    - 없으면: 앱 서버의 data/ 폴더 (앱이 재시작되면 사라짐)
    구조: index.json (목록) + conversations/<id>.json (대화 전체)
    """

    def __init__(self):
        self.token = secret("GITHUB_TOKEN")
        self.repo = secret("DATA_REPO")
        self.remote = bool(self.token and self.repo)
        # 기본은 저장소 루트의 data/ (이 파일은 majority/ 안에 있으므로 한 단계 위). 시험에서는 LOCAL_DATA_DIR로 바꿈
        self.local_dir = Path(secret("LOCAL_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")

    # GitHub Contents API
    def _url(self, path):
        return f"https://api.github.com/repos/{self.repo}/contents/{path}"

    def _headers(self, raw=False):
        return {"Authorization": f"Bearer {self.token}", "X-GitHub-Api-Version": "2022-11-28",
                "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json"}

    def _read(self, path):
        if not self.remote:
            f = self.local_dir / path
            return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        r = requests.get(self._url(path), headers=self._headers(raw=True), timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return json.loads(r.text)

    def _sha(self, path):
        r = requests.get(self._url(path), headers=self._headers(), timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json().get("sha")

    def _write(self, path, obj, message):
        text = json.dumps(obj, ensure_ascii=False, indent=1)
        if not self.remote:
            f = self.local_dir / path
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.name + ".tmp")      # 다 쓴 뒤 한 번에 바꿔서, 읽는 쪽이 반쯤 쓰인 파일을 보지 않게
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(f)
            return
        for _ in range(2):                       # 동시 수정으로 sha가 어긋나면 한 번 더
            body = {"message": message, "content": base64.b64encode(text.encode()).decode()}
            sha = self._sha(path)
            if sha:
                body["sha"] = sha
            r = requests.put(self._url(path), headers=self._headers(), json=body, timeout=30)
            if r.status_code != 409:
                r.raise_for_status()
                return
        r.raise_for_status()

    def _remove(self, path, message):
        if not self.remote:
            (self.local_dir / path).unlink(missing_ok=True)
            return
        sha = self._sha(path)
        if sha:
            requests.delete(self._url(path), headers=self._headers(),
                            json={"message": message, "sha": sha}, timeout=30).raise_for_status()

    # 공개 메서드
    def list(self):
        idx = self._read("index.json") or {}
        return sorted(({"id": k, **v} for k, v in idx.items()), key=lambda x: x["updated"], reverse=True)

    def load(self, cid):
        return self._read(f"conversations/{cid}.json")

    def save(self, conv):
        self._write(f"conversations/{conv['id']}.json", conv, f"대화 저장: {conv['title']}")
        idx = self._read("index.json") or {}
        idx[conv["id"]] = {"title": conv["title"], "created": conv["created"],
                           "updated": conv["updated"], "turns": len(conv["turns"]),
                           "cost": conv.get("cost", {}),
                           "cost_by_provider": conv.get("cost_by_provider", {
                               "Claude": conv.get("cost", {}), "Gemini": {}})}
        self._write("index.json", idx, f"목록 갱신: {conv['title']}")

    def delete(self, cid):
        self._remove(f"conversations/{cid}.json", "대화 삭제")
        idx = self._read("index.json") or {}
        if idx.pop(cid, None) is not None:
            self._write("index.json", idx, "목록 갱신: 삭제")


def now_text():
    return datetime.now(ZoneInfo(secret("APP_TIMEZONE", "Asia/Seoul"))).strftime("%Y-%m-%d %H:%M")
