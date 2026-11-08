"""모델 목록 조회 실패를 잠깐만 기억하고, 실패 이유를 기록하는지."""
import time

import conftest


def fresh_core():
    """실제 model_ids를 쓰는 이름공간 (공통 준비물은 model_ids를 가짜로 바꾸므로 따로 만듦)."""
    ns = conftest.load_core(fake_model_list=False)
    ns["secret"] = lambda k, d=None: "x" if k.endswith("_API_KEY") else d
    return ns


def stub(fn):
    fn.clear = lambda: None                   # 실제 list_models(st.cache_data)처럼 clear()를 가짐
    return fn


def test_failure_is_remembered_briefly_and_reason_recorded():
    ns = fresh_core()
    state = {"calls": 0, "fail": True}

    @stub
    def list_models(provider, key):
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("PermissionDenied 403 bad key")
        return [("gemini-3.8-flash", ""), ("gemini-3.8-flash-lite", "")]
    ns["list_models"] = list_models

    assert ns.model_ids("Gemini") is None and ns.model_ids("Gemini") is None
    assert state["calls"] == 1                                    # 실패는 1분 동안 다시 묻지 않음
    assert "403" in ns.model_list_failures()["Gemini"][1]

    ns.model_list_failures()["Gemini"] = (time.time() - 61, "old")
    state["fail"] = False
    assert ns.model_ids("Gemini") == ["gemini-3.8-flash", "gemini-3.8-flash-lite"]
    assert "Gemini" not in ns.model_list_failures()               # 성공하면 실패 기록 삭제


def test_empty_filtered_list_is_recorded():
    ns = fresh_core()
    ns["list_models"] = stub(lambda p, k: [])
    assert ns.model_ids("Gemini") is None
    assert "0개" in ns.model_list_failures()["Gemini"][1]
