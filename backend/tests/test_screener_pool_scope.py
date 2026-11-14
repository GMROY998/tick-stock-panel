"""选股范围 (股票池) 与作用域缓存隔离。

契约:
  * run_all 传 pool 时, engine 收到同一份池; 结果写 strategy_cache_scoped.json,
    全市场 strategy_cache.json 不被改动。
  * 不传 pool = 全市场, 写全市场缓存, 响应 scope=None。
  * 空池 (如空自选分组) = 0 命中, 不能退化成全市场。
  * cached / cached-summary / cached-result 传 scope 时只读对应作用域,
    缺省仍读全市场。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from types import SimpleNamespace

from app.api import screener as screener_api
from app.services import strategy_cache

AS_OF = "2026-11-13"


@dataclass
class _FakeResult:
    total: int = 1
    rows: list = field(default_factory=lambda: [{"symbol": "600000.SH", "close": 10.0}])
    as_of: str = AS_OF


class _FakeEngine:
    def __init__(self) -> None:
        self.received_pools: list = []

    def has(self, sid: str) -> bool:
        return sid == "s1"

    def get(self, sid: str):
        return SimpleNamespace(meta={})

    def run_all(
        self,
        context,
        params_map=None,
        overrides_map=None,
        *,
        strategy_ids=None,
        parallel=True,
        pool=None,
    ):
        self.received_pools.append(pool)
        return {sid: _FakeResult() for sid in (strategy_ids or [])}


class _FakeService:
    def __init__(self, repo, asset_type="stock"):
        pass

    def latest_date(self):
        return date.fromisoformat(AS_OF)

    def build_strategy_context(self, *args, **kwargs):
        return SimpleNamespace()


def _request(tmp_path, engine=None):
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    state = SimpleNamespace(
        repo=repo,
        strategy_engine=engine or _FakeEngine(),
        monitor_engine=SimpleNamespace(latest_strategy_results=lambda: {}),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _body(**overrides):
    body = {
        "as_of": AS_OF,
        "strategy_ids": ["s1"],
        "asset_type": "stock",
        "timeframe": "1d",
    }
    body.update(overrides)
    return body


def _patch_service(monkeypatch):
    monkeypatch.setattr(screener_api, "ScreenerService", _FakeService)


def _scope(pool: list[str]) -> str:
    joined = ",".join(sorted(pool))
    return "pool:" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def test_pool_run_passes_pool_and_writes_scope_cache(tmp_path, monkeypatch):
    _patch_service(monkeypatch)
    engine = _FakeEngine()

    resp = screener_api.run_all(_request(tmp_path, engine), _body(pool=["600000.SH"]))

    assert engine.received_pools == [["600000.SH"]]
    assert resp["scope"] == _scope(["600000.SH"])
    assert resp["results"]["s1"]["total"] == 1

    # 全市场缓存不被污染
    assert (strategy_cache.read_cache(tmp_path) or {}).get("results") in (None, {})
    # 作用域缓存写入
    bucket = strategy_cache.read_scoped(tmp_path, resp["scope"])
    assert bucket is not None
    assert bucket["as_of"] == AS_OF
    assert bucket["results"]["s1"]["total"] == 1


def test_full_market_run_keeps_default_cache_and_null_scope(tmp_path, monkeypatch):
    _patch_service(monkeypatch)
    engine = _FakeEngine()

    resp = screener_api.run_all(_request(tmp_path, engine), _body())

    assert engine.received_pools == [None]
    assert resp["scope"] is None
    assert (strategy_cache.read_cache(tmp_path) or {}).get("as_of") == AS_OF
    assert strategy_cache.read_cache(tmp_path)["results"]["s1"]["total"] == 1


def test_empty_pool_returns_zero_hits_without_touching_caches(tmp_path, monkeypatch):
    _patch_service(monkeypatch)
    engine = _FakeEngine()

    resp = screener_api.run_all(_request(tmp_path, engine), _body(pool=[]))

    assert resp["results"] == {}
    assert resp["complete"] is True
    assert engine.received_pools == []
    assert strategy_cache.read_cache(tmp_path) is None


def test_scoped_cache_endpoints_do_not_leak_full_market(tmp_path, monkeypatch):
    scope = _scope(["600000.SH"])
    strategy_cache.write_scoped(
        tmp_path,
        scope,
        AS_OF,
        {"s1": {"total": 1, "as_of": AS_OF, "rows": [{"symbol": "600000.SH"}]}},
    )
    request = _request(tmp_path)

    summary = screener_api.get_cached_summary(request, scope=scope)
    assert summary["results"]["s1"]["total"] == 1

    cached = screener_api.get_cached(request, ext_columns=None, scope=scope)
    assert cached["results"]["s1"]["rows"][0]["symbol"] == "600000.SH"

    single = screener_api.get_cached_result("s1", request, ext_columns=None, scope=scope)
    assert single["result"]["rows"][0]["symbol"] == "600000.SH"

    # 不传 scope → 读全市场 (此时为空), 不会串到池化结果
    full = screener_api.get_cached(request, ext_columns=None)
    assert full["results"] == {}
