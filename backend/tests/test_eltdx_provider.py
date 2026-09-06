"""eltdx-gateway 数据源插件离线单测: 代码转换、日K/分钟映射、快照 schema、
五档 depth 映射、设置页试拉与 depth_service 偏好路由。"""

from __future__ import annotations

import threading
from datetime import date, datetime

import pytest

from app.plugins.eltdx import client as eltdx_client_mod
from app.plugins.eltdx import provider as eltdx_mod
from app.plugins.eltdx.client import to_gw_code, to_symbol
from app.plugins.eltdx.provider import EltdxProvider

DAILY_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
MINUTE_COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
SNAPSHOT_KEYS = eltdx_mod._SNAPSHOT_RECORD_KEYS


class FakeClient:
    """按 method 分发 canned 数据; 记录调用 (线程安全, provider 内部并发)。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def rpc(self, method, params=None, timeout=None):
        with self._lock:
            self.calls.append((method, dict(params or {})))
        return self.handler(method, params or {})

    def close(self) -> None:
        pass


def make_provider(handler) -> tuple[EltdxProvider, FakeClient]:
    prov = EltdxProvider(base_url="http://test", max_workers=4)
    client = FakeClient(handler)
    prov._client = client  # type: ignore[assignment]
    return prov, client


# ---------------------------------------------------------------- 代码转换
def test_symbol_conversion():
    assert to_gw_code("600519.SH") == "sh600519"
    assert to_gw_code("000001.SZ") == "sz000001"
    assert to_gw_code("300750.sz") == "sz300750"
    assert to_gw_code("600519") == "sh600519"      # 纯代码按位推断
    assert to_gw_code("000001") == "sz000001"
    assert to_gw_code("sh600519") == "sh600519"    # 已是网关格式
    assert to_gw_code("830799.BJ") is None         # 北交所不入网关链路
    assert to_gw_code("") is None
    assert to_symbol("sh600519") == "600519.SH"
    assert to_symbol("sz000001") == "000001.SZ"


def test_resolve_base_url_env(monkeypatch):
    monkeypatch.setenv(eltdx_client_mod.ENV_GATEWAY_URL, "http://10.0.0.1:9999/")
    assert eltdx_client_mod.resolve_base_url() == "http://10.0.0.1:9999"
    monkeypatch.delenv(eltdx_client_mod.ENV_GATEWAY_URL)
    assert eltdx_client_mod.resolve_base_url() == eltdx_client_mod.DEFAULT_BASE_URL


# ---------------------------------------------------------------- daily
def _day_bar(d: str, close: float) -> dict:
    return {"time": f"{d}T15:00:00+08:00", "open": close - 0.1, "high": close + 0.2,
            "low": close - 0.2, "close": close,
            "volume_lots": 1234.5, "amount": close * 123456}


def test_daily_parses_filters_and_maps():
    bars = [
        _day_bar("2026-08-17", 100.5),
        _day_bar("2026-08-18", 102.0),
        _day_bar("2026-09-10", 1.0),   # 窗口外应被过滤
    ]

    def handler(method, params):
        if method == "bars.get":
            assert params["period"] == "day"
            assert params["adjust"] == "none"
            assert params["code"] == "sh600519"
            return {"code": "600519", "bars": bars}
        return None

    prov, _ = make_provider(handler)
    df = prov.get_daily(["600519.SH"], datetime(2026, 8, 17), datetime(2026, 9, 1))
    assert df.columns == DAILY_COLS
    assert df.height == 2
    assert df["symbol"].to_list() == ["600519.SH", "600519.SH"]
    assert df["date"].to_list() == [date(2026, 8, 17), date(2026, 8, 18)]
    assert df["close"].to_list() == [100.5, 102.0]
    assert df["volume"].to_list() == [1234.5, 1234.5]     # 手
    assert df["amount"].to_list() == [pytest.approx(100.5 * 123456), pytest.approx(102.0 * 123456)]


def test_daily_non_stock_and_unsupported_codes():
    def handler(method, params):  # pragma: no cover - 不应被调用
        raise AssertionError("should not request")

    prov, client = make_provider(handler)
    assert prov.get_daily(["510300.SH"], asset_type="etf").height == 0
    assert not client.calls

    prov2, client2 = make_provider(lambda m, p: {"bars": [_day_bar("2026-08-17", 10.0)]})
    df = prov2.get_daily(["830799.BJ", "600519.SH"], datetime(2026, 8, 17), datetime(2026, 8, 17))
    assert df["symbol"].to_list() == ["600519.SH"]   # 北交所跳过
    assert len(client2.calls) == 1


def test_daily_per_code_failure_skipped():
    def handler(method, params):
        if params.get("code") == "sz000001":
            raise eltdx_mod.EltdxError("boom")
        return {"bars": [_day_bar("2026-08-17", 10.0)]}

    prov, _ = make_provider(handler)
    df = prov.get_daily(["600519.SH", "000001.SZ"], datetime(2026, 8, 17), datetime(2026, 8, 17))
    assert df["symbol"].to_list() == ["600519.SH"]


def test_daily_large_window_uses_all_pages():
    captured: dict = {}

    def handler(method, params):
        if method == "bars.get":
            captured.update(params)
            return {"bars": [_day_bar("2020-01-02", 10.0)]}
        return None

    prov, _ = make_provider(handler)
    df = prov.get_daily(["600519.SH"], datetime(2019, 1, 1), datetime(2026, 9, 1))
    assert captured["all_pages"] is True                  # >800 根走 all_pages 分页
    assert captured["page_size"] == 800
    assert "count" not in captured
    assert df.height == 1 and df["date"].to_list() == [date(2020, 1, 2)]


# ---------------------------------------------------------------- minute
def _mh_point(d: str, hm: str, price: float, volume: float = 100.0) -> dict:
    return {"time": f"{d}T{hm}:00+08:00", "time_label": hm,
            "price": price, "volume": volume}


def test_minute_builds_bars_with_naive_beijing_datetime():
    def handler(method, params):
        if method == "minutes.history":
            assert params["code"] == "sz000001"
            if params["trading_date"] != "2026-09-04":     # 其余日期按非交易日失败
                raise eltdx_mod.EltdxError("not a trading day")
            return {"points": [_mh_point("2026-09-04", "09:31", 11.9, 27126),
                               _mh_point("2026-09-04", "09:32", 11.89, 14408)]}
        return None

    prov, client = make_provider(handler)
    df = prov.get_minute(["000001.SZ"], datetime(2026, 9, 1), datetime(2026, 9, 5))
    assert df.columns == MINUTE_COLS
    assert df.height == 2
    dts = df["datetime"].to_list()
    assert dts == [datetime(2026, 9, 4, 9, 31), datetime(2026, 9, 4, 9, 32)]
    assert all(dt.tzinfo is None for dt in dts)           # naive 北京时间
    assert df["volume"].to_list() == [27126.0, 14408.0]   # 手
    row = df.row(0, named=True)
    assert row["open"] == row["high"] == row["low"] == row["close"] == 11.9  # 分时价
    assert row["amount"] == pytest.approx(11.9 * 27126 * 100)                # 手→股 估算
    # 窗口内每个自然日都发起了请求(非交易日失败被跳过, 不影响其余日)
    hist_days = {p["trading_date"] for m, p in client.calls if m == "minutes.history"}
    assert hist_days == {"2026-09-01", "2026-09-02", "2026-09-03",
                         "2026-09-04", "2026-09-05"}


def test_minute_window_capped_and_failures_skipped():
    def handler(method, params):
        if method == "minutes.history":
            if params["code"] == "sz300750":
                raise eltdx_mod.EltdxError("boom")         # 整票失败跳过
            return {"points": [_mh_point(params["trading_date"], "09:31", 10.0)]}
        return None

    prov, client = make_provider(handler)
    df = prov.get_minute(["600519.SH", "sz300750"], datetime(2026, 9, 3), datetime(2026, 9, 5))
    assert df["symbol"].to_list() == ["600519.SH"] * 3     # 整票失败跳过, 其余 3 天齐
    assert len(client.calls) == 6                          # 2 票 × 3 自然日(9/3,9/4,9/5)


def test_minute_non_stock_and_wrong_freq_raise():
    prov, client = make_provider(lambda *_: None)
    with pytest.raises(eltdx_mod.EltdxError):
        prov.get_minute(["510300.SH"], asset_type="etf")
    with pytest.raises(eltdx_mod.EltdxError):
        prov.get_minute(["600519.SH"], freq="5m")
    assert not client.calls


# ---------------------------------------------------------------- realtime
def _snap_row(code, exchange, last, pre):
    return {"exchange": exchange, "code": code, "last_price": last,
            "pre_close_price": pre, "open_price": pre, "high_price": last,
            "low_price": pre, "total_hand": 814372, "amount": 969948416.0}


def test_realtime_full_market_snapshot_mapping():
    all_codes = ["sh600519", "sz000001", "sz300750"]

    def handler(method, params):
        if method == "codes.all_a_shares":
            return all_codes
        if method == "quotes.get_snapshots":
            return [_snap_row(c[2:], c[:2], 10.0, 9.8) for c in params["codes"]]
        return None

    prov, client = make_provider(handler)
    recs = prov.get_realtime()
    assert len(recs) == 3
    assert set(recs[0].keys()) == set(SNAPSHOT_KEYS)
    by_sym = {r["symbol"]: r for r in recs}
    assert set(by_sym) == {"600519.SH", "000001.SZ", "300750.SZ"}
    r = by_sym["600519.SH"]
    assert r["last_price"] == 10.0
    assert r["prev_close"] == 9.8
    assert r["change_pct"] == pytest.approx(0.2 / 9.8)    # 小数制
    assert r["change_amount"] == pytest.approx(0.2)
    assert r["volume"] == 814372.0                         # 手
    assert r["amount"] == pytest.approx(969948416.0)       # 元
    assert r["name"] is None and r["amplitude"] is None and r["turnover_rate"] is None
    assert r["timestamp"] > 0
    # 分批: 3 只一批 → 1 次快照请求
    snap_calls = [p for m, p in client.calls if m == "quotes.get_snapshots"]
    assert len(snap_calls) == 1 and snap_calls[0]["codes"] == all_codes


def test_realtime_invalid_rows_and_chunk_failure_tolerated(monkeypatch):
    monkeypatch.setattr(eltdx_mod, "_SNAPSHOT_CHUNK", 1)   # 每码一批, 失败互不影响

    def handler(method, params):
        if method == "codes.all_a_shares":
            return ["sh600519", "sz000001"]
        if method == "quotes.get_snapshots":
            codes = params["codes"]
            if "sz000001" in codes:            # 该批失败
                raise eltdx_mod.EltdxError("gateway boom")
            rows = [_snap_row("600519", "sh", 10.0, 9.8)]
            rows.append({"exchange": "sh", "code": "bad", "last_price": 1})  # 无效行
            return rows
        return None

    prov, _ = make_provider(handler)
    recs = prov.get_realtime()
    assert [r["symbol"] for r in recs] == ["600519.SH"]


def test_realtime_zero_price_filtered_and_retry_once(monkeypatch):
    monkeypatch.setattr(eltdx_mod, "_SNAPSHOT_CHUNK", 1)
    attempts: dict[tuple, int] = {}

    def handler(method, params):
        if method == "codes.all_a_shares":
            return ["sh600519", "sz000001"]
        if method == "quotes.get_snapshots":
            codes = tuple(params["codes"])
            attempts[codes] = attempts.get(codes, 0) + 1
            if codes == ("sz000001",) and attempts[codes] == 1:
                raise eltdx_mod.EltdxError("timeout")      # 首次失败 → 串行重试成功
            rows = []
            if "sh600519" in codes:
                rows.append(_snap_row("600519", "sh", 10.0, 9.8))
            if "sz000001" in codes:
                rows.append(_snap_row("000001", "sz", 0.0, 0.0))   # 零价行应剔除
            return rows
        return None

    prov, _ = make_provider(handler)
    recs = prov.get_realtime()
    assert [r["symbol"] for r in recs] == ["600519.SH"]    # 零价行被过滤
    assert attempts[("sz000001",)] == 2                    # 失败批重试了一次


def test_realtime_indices_suffix_mapping():
    def handler(method, params):
        if method == "quotes.get_snapshots":
            assert params["codes"] == ["sh000001", "sz399001"]
            return [_snap_row("000001", "sh", 3220.5, 3205.25),
                    _snap_row("399001", "sz", 10000.0, 9950.0)]
        return None

    prov, _ = make_provider(handler)
    recs = prov.get_realtime_indices(["000001.SH", "399001.SZ"])
    assert sorted(r["symbol"] for r in recs) == ["000001.SH", "399001.SZ"]
    assert recs[0]["prev_close"] == pytest.approx(3205.25)


# ---------------------------------------------------------------- depth5
def _depth_record(code, exchange, bid1, ask1):
    return {"exchange": exchange, "code": code,
            "buy_levels": [{"price": 10.0 - i * 0.01, "volume": bid1 + i} for i in range(5)],
            "sell_levels": [{"price": 10.0 + i * 0.01, "volume": ask1 + i} for i in range(5)]}


def test_depth_batch_maps_five_levels():
    def handler(method, params):
        if method == "quotes.get_depth":
            assert params["codes"] == ["sz000001", "sh600519"]
            return {"requested_codes": params["codes"],
                    "records": [_depth_record("000001", "sz", 908, 0),
                                _depth_record("600519", "sh", 100, 200)]}
        return None

    prov, _ = make_provider(handler)
    out = prov.batch(["000001.SZ", "600519.SH"])
    assert set(out) == {"000001.SZ", "600519.SH"}
    d = out["000001.SZ"]
    assert d["bid_volumes"] == [908, 909, 910, 911, 912]
    assert d["ask_volumes"] == [0, 1, 2, 3, 4]            # 涨停真封: 卖一=0
    assert d["timestamp"] > 0


def test_depth_batch_chunking_and_unsupported(monkeypatch):
    monkeypatch.setattr(eltdx_mod, "_DEPTH_CHUNK", 2)
    seen = []

    def handler(method, params):
        if method == "quotes.get_depth":
            seen.append(params["codes"])
            return {"records": [_depth_record(c[2:], c[:2], 1, 2) for c in params["codes"]]}
        return None

    prov, _ = make_provider(handler)
    out = prov.batch(["000001.SZ", "600519.SH", "830799.BJ", "300750.SZ"])
    assert len(seen) == 2                                  # ceil(3/2)
    assert set(out) == {"000001.SZ", "600519.SH", "300750.SZ"}


# ---------------------------------------------------------------- surface
def test_config_datasets_membership():
    prov = EltdxProvider()
    for ds in ("daily", "minute", "realtime", "depth5"):
        assert ds in prov.config.datasets
    assert prov.name == "eltdx"


def test_test_dataset_shapes():
    def handler(method, params):
        if method == "bars.get" and params["period"] == "day":
            return {"bars": [_day_bar("2026-08-17", 10.0)]}
        if method == "quotes.get_depth":
            return {"records": [_depth_record("600519", "sh", 1, 2)]}
        return None

    prov, _ = make_provider(handler)
    out = prov.test_dataset("daily", ["600519.SH"])
    assert out["provider"] == "eltdx" and out["rows"] == 1

    out = prov.test_dataset("depth5", ["600519.SH"])
    assert out["rows"] == 1
    assert out["preview"][0] == {"symbol": "600519.SH", "bid1_vol": 1, "ask1_vol": 2}

    with pytest.raises(ValueError):
        prov.test_dataset("adj_factor")


def test_availability_delegates_to_health(monkeypatch):
    class FakeHealthClient:
        def __init__(self, *a, **k):
            pass

        def health(self):
            return False, "网关不可达"

        def close(self):
            pass

    monkeypatch.setattr(eltdx_mod, "EltdxClient", FakeHealthClient)
    ok, msg = eltdx_mod.availability()
    assert not ok and "网关不可达" in msg


# ---------------------------------------------------------------- depth_service 路由
class _FakePluginDepth:
    def __init__(self, result=None, exc=None):
        self.result = result if result is not None else {
            "000001.SZ": {"ask_volumes": [0], "bid_volumes": [1234], "timestamp": 1}}
        self.exc = exc
        self.calls: list[list[str]] = []

    def batch(self, symbols):
        self.calls.append(list(symbols))
        if self.exc:
            raise self.exc
        return self.result


class _FakeTickFlowDepth:
    def __init__(self):
        self.calls: list[list[str]] = []

    def batch(self, symbols):
        self.calls.append(list(symbols))
        return {"000001.SZ": {"ask_volumes": [9], "bid_volumes": [9], "timestamp": 2}}


def _route(monkeypatch, plugin_name):
    """把 DepthService 的偏好/注册表路由劫持到 fake。"""
    from app.services.depth_service import DepthService

    svc = DepthService()
    tf = _FakeTickFlowDepth()
    svc._tf_depth = tf
    plugin = _FakePluginDepth()
    monkeypatch.setattr("app.services.preferences.get_depth5_data_provider",
                        lambda: plugin_name)
    monkeypatch.setattr("app.data_providers.custom.provider_has_dataset",
                        lambda name, ds: True)
    monkeypatch.setattr("app.data_providers.custom.get_provider", lambda name: plugin)
    return svc, plugin, tf


def test_depth_routes_to_plugin_by_preference(monkeypatch):
    svc, plugin, tf = _route(monkeypatch, "eltdx")
    out = svc._call_depth_batch(["000001.SZ"])
    assert out["000001.SZ"]["ask_volumes"] == [0]
    assert plugin.calls == [["000001.SZ"]]
    assert tf.calls == []                                  # 不应回退


def test_depth_falls_back_to_tickflow_on_plugin_failure(monkeypatch):
    svc, plugin, tf = _route(monkeypatch, "eltdx")
    plugin.exc = RuntimeError("gateway down")
    out = svc._call_depth_batch(["000001.SZ"])
    assert out["000001.SZ"]["ask_volumes"] == [9]          # TickFlow 兜底
    assert tf.calls == [["000001.SZ"]]


def test_depth_falls_back_on_empty_plugin_result(monkeypatch):
    svc, plugin, tf = _route(monkeypatch, "eltdx")
    plugin.result = {}
    out = svc._call_depth_batch(["000001.SZ"])
    assert out["000001.SZ"]["ask_volumes"] == [9]
    assert tf.calls == [["000001.SZ"]]


def test_depth_tickflow_preference_never_touches_plugin(monkeypatch):
    svc, plugin, tf = _route(monkeypatch, "tickflow")
    out = svc._call_depth_batch(["000001.SZ"])
    assert out["000001.SZ"]["ask_volumes"] == [9]
    assert plugin.calls == [] and tf.calls == [["000001.SZ"]]


# ---------------------------------------------------------------- TickFlowDepthProvider
def test_tickflow_depth_provider_slices_and_throttles(monkeypatch):
    from app.data_providers.depth_provider import TickFlowDepthProvider

    from types import SimpleNamespace
    import app.tickflow.client as tf_client_mod
    import app.tickflow.rate_limits as rl_mod

    seen_chunks: list[list[str]] = []

    class FakeTF:
        class depth:
            @staticmethod
            def batch(chunk):
                seen_chunks.append(list(chunk))
                return {s: {"ask_volumes": [0], "bid_volumes": [1], "timestamp": 3}
                        for s in chunk}

    monkeypatch.setattr(tf_client_mod, "get_client", lambda: FakeTF)
    monkeypatch.setattr(rl_mod, "resolve_limit",
                        lambda *a, **k: SimpleNamespace(batch=2, rpm=0))
    monkeypatch.setattr(rl_mod, "sleep_between_batches", lambda *a, **k: None)

    out = TickFlowDepthProvider().batch(["a", "b", "c"])
    assert seen_chunks == [["a", "b"], ["c"]]
    assert set(out) == {"a", "b", "c"}


def test_tickflow_depth_provider_single_chunk_failure_tolerated(monkeypatch):
    from app.data_providers.depth_provider import TickFlowDepthProvider

    from types import SimpleNamespace
    import app.tickflow.client as tf_client_mod
    import app.tickflow.rate_limits as rl_mod

    class FakeDepth:
        @staticmethod
        def batch(chunk):
            if "b" in chunk:
                raise RuntimeError("boom")
            return {s: {"ask_volumes": [1], "bid_volumes": [1], "timestamp": 3} for s in chunk}

    class FakeTF:
        depth = FakeDepth

    monkeypatch.setattr(tf_client_mod, "get_client", lambda: FakeTF)
    monkeypatch.setattr(rl_mod, "resolve_limit",
                        lambda *a, **k: SimpleNamespace(batch=2, rpm=0))
    monkeypatch.setattr(rl_mod, "sleep_between_batches", lambda *a, **k: None)

    out = TickFlowDepthProvider(capset_provider=lambda: None).batch(["a", "b", "c"])
    assert set(out) == {"c"}                               # 失败批(a,b)整体跳过
