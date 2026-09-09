"""MeoZ 数据源插件离线单测: 解析、窗口过滤、除权事件推导、分钟组装与分支。"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from app.plugins.meoz import provider as meoz_mod
from app.plugins.meoz.provider import MeoZProvider

DAILY_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
ADJ_COLS = ["symbol", "trade_date", "ex_factor"]
MINUTE_COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]


class FakeClient:
    """按 apiname 分发 canned 数据; 记录调用。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, dict]] = []

    def rows(self, apiname, fields, params):
        self.calls.append((apiname, dict(params)))
        return self.handler(apiname, params) or []


def make_provider(handler) -> tuple[MeoZProvider, FakeClient]:
    prov = MeoZProvider(api_key="test-key")
    client = FakeClient(handler)
    prov._client = client  # type: ignore[assignment]
    return prov, client


# ---------------------------------------------------------------- daily
def test_daily_parses_remaps_and_filters():
    rows = [
        {"tradedate": "20260817", "symbol": "600519", "open": "100.0", "high": "101",
         "low": "99.5", "close": "100.5", "vol": "1234", "amount": "12500000"},
        {"tradedate": "20260818", "symbol": "600519", "open": "100.5", "high": "103",
         "low": "100.0", "close": "102.0", "vol": "2000", "amount": "20400000"},
        # 窗口外的行应被丢弃
        {"tradedate": "20260910", "symbol": "600519", "open": "1", "high": "1",
         "low": "1", "close": "1", "vol": "1", "amount": "1"},
    ]

    def handler(apiname, params):
        if apiname == "daily":
            return rows

    prov, _ = make_provider(handler)
    df = prov.get_daily(["600519.SH"], datetime(2026, 8, 17), datetime(2026, 9, 1))
    assert df.columns == DAILY_COLS
    assert df.height == 2
    assert df["symbol"].to_list() == ["600519.SH", "600519.SH"]  # 纯代码回标为带后缀
    assert df["date"].to_list() == [date(2026, 8, 17), date(2026, 8, 18)]
    assert df["volume"].to_list() == [1234.0, 2000.0]
    assert df["amount"].to_list() == [12500000.0, 20400000.0]
    assert df["close"].to_list() == [100.5, 102.0]


def test_daily_asset_not_stock_returns_empty():
    prov, client = make_provider(lambda *_: None)
    df = prov.get_daily(["510300.SH"], asset_type="etf")
    assert df.height == 0
    assert not client.calls  # 不应发出请求


# ---------------------------------------------------------------- adj
def test_adj_factors_emit_only_real_ex_events():
    # 茅台 2024: 06-19 真实除息(因子明显变化); 06-20 有一处 ~1e-5 舍入噪声应被滤除;
    # 05-20 事件日期在窗口外(start=06-01)不输出。
    rows = [
        {"tradedate": "20240520", "symbol": "600519", "adj_factor": "2.95"},
        {"tradedate": "20240531", "symbol": "600519", "adj_factor": "2.9999"},
        {"tradedate": "20240618", "symbol": "600519", "adj_factor": "3.00"},
        {"tradedate": "20240619", "symbol": "600519", "adj_factor": "3.1130"},
        {"tradedate": "20240620", "symbol": "600519", "adj_factor": "3.113037"},
        {"tradedate": "20240701", "symbol": "600519", "adj_factor": "3.113037"},
    ]

    def handler(apiname, params):
        if apiname == "adj_factor":
            return rows

    prov, _ = make_provider(handler)
    df = prov.get_adj_factors(["600519.SH"], datetime(2024, 6, 1), datetime(2024, 7, 31))
    assert df.columns == ADJ_COLS
    assert df.height == 1
    row = df.row(0)
    assert row[0] == "600519.SH"
    assert row[1] == date(2024, 6, 19)
    assert abs(row[2] - 3.1130 / 3.00) < 1e-9


def test_adj_uses_buffer_before_window_start():
    """start 当日发生除权也应能被检测(需要窗口前基准因子)。"""
    rows = [
        {"tradedate": "20240610", "symbol": "600519", "adj_factor": "3.0"},
        {"tradedate": "20240611", "symbol": "600519", "adj_factor": "3.1"},
    ]

    def handler(apiname, params):
        if apiname == "adj_factor":
            return rows

    prov, client = make_provider(handler)
    df = prov.get_adj_factors(["600519.SH"], datetime(2024, 6, 11), datetime(2024, 6, 11))
    # buffer 前移, adj_factor 请求窗口起点应早于 start(2024-06-11 - 15 天)
    adj_call = next(p for a, p in client.calls if a == "adj_factor")
    assert adj_call["startdate"] == "20240527"
    assert df.height == 1 and df.row(0)[1] == date(2024, 6, 11)


# ---------------------------------------------------------------- minute
def _minute_row(day, sym, hm, price):
    return {"tradedate": day, "symbol": sym, "trademin": hm,
            "open": price, "high": price, "low": price, "close": price,
            "vol": "10", "amount": str(float(price) * 1000)}


def test_minute_builds_bars_per_trading_day():
    cal = [
        {"cal_date": "20260901", "is_open": "1"},
        {"cal_date": "20260902", "is_open": "1"},
        {"cal_date": "20260903", "is_open": "0"},
    ]

    def handler(apiname, params):
        if apiname == "tradecal":
            return cal
        if apiname == "minute":
            day = params["tradedate"]
            return [
                _minute_row(day, "600519", "0930", "100.0"),
                _minute_row(day, "600519", "1500", "101.0"),
            ]

    prov, client = make_provider(handler)
    df = prov.get_minute(["600519.SH"], datetime(2026, 9, 1), datetime(2026, 9, 3))
    assert df.columns == MINUTE_COLS
    assert df.height == 4  # 2 交易日 x 2 根
    assert df["symbol"].to_list() == ["600519.SH"] * 4
    dts = df["datetime"].to_list()
    assert dts[0] == datetime(2026, 9, 1, 9, 30)
    assert dts[1] == datetime(2026, 9, 1, 15, 0)
    assert dts[2] == datetime(2026, 9, 2, 9, 30)
    # 只请求 is_open=1 的交易日
    minute_days = [p["tradedate"] for a, p in client.calls if a == "minute"]
    assert minute_days == ["20260901", "20260902"]
    # 其余日期无 minute 请求
    assert all(dt.tzinfo is None for dt in dts)  # naive 北京时间


def test_minute_non_stock_and_wrong_freq_raise():
    prov, _ = make_provider(lambda *_: None)
    with pytest.raises(meoz_mod.MeoZError):
        prov.get_minute(["510300.SH"], asset_type="etf")
    with pytest.raises(meoz_mod.MeoZError):
        prov.get_minute(["600519.SH"], freq="5m")


# ---------------------------------------------------------------- helpers / surface
def test_config_datasets_membership():
    prov = MeoZProvider(api_key="test-key")
    for ds in ("daily", "adj_factor", "minute", "realtime", "full_minute", "depth5"):
        assert ds in prov.config.datasets


def test_strip_exchange_and_symbol_map():
    assert meoz_mod._strip_exchange("600519.SH") == "600519"
    assert meoz_mod._strip_exchange("000001.sz") == "000001"
    assert meoz_mod._symbol_map(["600519.SH", "000001.SZ"]) == {
        "600519": "600519.SH", "000001": "000001.SZ"}


def test_parse_helpers():
    assert meoz_mod._parse_date("20260902") == date(2026, 9, 2)
    assert meoz_mod._parse_date(20260902) == date(2026, 9, 2)
    assert meoz_mod._parse_hm("0930") == datetime(2000, 1, 1, 9, 30).time()
    assert meoz_mod._parse_hm(None, "15:00:00") == datetime(2000, 1, 1, 15, 0).time()


def test_test_dataset_shape():
    def handler(apiname, params):
        if apiname == "daily":
            return [{"tradedate": "20260901", "symbol": "600519", "open": "1",
                     "high": "1", "low": "1", "close": "1", "vol": "1", "amount": "1"}]
        return []

    prov, _ = make_provider(handler)
    out = prov.test_dataset("daily")
    assert out["provider"] == "meoz"
    assert out["dataset"] == "daily"
    assert out["rows"] == 1
    assert set(out["columns"]) >= {"symbol", "date", "close"}


# ---------------------------------------------------------------- realtime
def test_tick_snapshot_row_mapping():
    """tick 行 → record: pct 百分数转小数制、prev 反推、volume 透传、全键 schema。"""
    rows = [
        {"symbol": "600519.SH", "open": "9.90", "high": "10.05", "low": "9.85",
         "close": "10.00", "vol": "1234", "amount": "12500000.5", "pct_chg": "2.0"},
        {"symbol": "000001.SZ", "open": "11.4", "high": "11.6", "low": "11.3",
         "close": "11.5", "vol": "100", "amount": "1150000", "pct_chg": "0.0"},
        # 无效行(空/无收盘价)应被剔除
        {"symbol": "", "close": "1", "vol": "1", "amount": "1"},
        {"symbol": "300750.SZ", "close": None, "vol": "1", "amount": "1"},
    ]

    def handler(apiname, params):
        if apiname == "tick":
            return rows
        return []

    prov, client = make_provider(handler)
    recs = prov.get_realtime()
    assert len(recs) == 2
    assert set(recs[0].keys()) == set(meoz_mod._SNAPSHOT_RECORD_KEYS)
    assert recs[0]["symbol"] == "600519.SH"
    assert recs[0]["last_price"] == 10.0
    assert recs[0]["prev_close"] == pytest.approx(10.0 / 1.02)
    assert recs[0]["change_pct"] == pytest.approx(0.02)      # 2.0% 百分数 → 小数制
    assert recs[0]["change_amount"] == pytest.approx(10.0 - 10.0 / 1.02)
    assert recs[0]["volume"] == 1234.0                        # 手, 直接透传
    assert recs[0]["amount"] == pytest.approx(12500000.5)
    assert recs[0]["name"] is None and recs[0]["amplitude"] is None
    assert recs[1]["prev_close"] == pytest.approx(11.5)       # pct=0 → prev=close
    assert recs[1]["change_pct"] == 0.0
    # 单次全市场 tick 请求, 无 symbols 入参
    assert [a for a, _ in client.calls] == ["tick"]
    assert client.calls[0][1] == {"type": 1}


def test_realtime_tick_failure_returns_empty():
    def handler(apiname, params):
        raise meoz_mod.MeoZError("boom")

    prov, _ = make_provider(handler)
    assert prov.get_realtime() == []


def test_realtime_indices_pure_code_param():
    def handler(apiname, params):
        if apiname == "index_snapshot":
            return [{"symbol": "000001.SH", "name": "上证指数",
                     "open": "3200", "high": "3230", "low": "3185",
                     "close": "3220.5", "pre_close": "3205.25", "change": "15.25",
                     "pct_chg": "0.4758", "vol": "516472775", "amount": "835300000000"}]
        return []

    prov, client = make_provider(handler)
    recs = prov.get_realtime_indices(["000001.SH", "399001.SZ"])
    assert client.calls[0][0] == "index_snapshot"
    assert client.calls[0][1]["symbols"] == "000001,399001"   # 纯代码入参
    assert len(recs) == 1
    assert recs[0]["symbol"] == "000001.SH"
    assert recs[0]["name"] == "上证指数"
    assert recs[0]["prev_close"] == pytest.approx(3205.25)
    assert recs[0]["change_amount"] == pytest.approx(15.25)
    assert recs[0]["change_pct"] == pytest.approx(0.004758)   # 0.4758% → 小数
    assert recs[0]["volume"] == pytest.approx(516472775.0)


# ---------------------------------------------------------------- full_minute
DAY2 = "20260902"


def _intraday_rows(params, syms=("600519.SH", "000001.SZ", "300750.SZ")):
    out = []
    day = params.get("tradedate") or DAY2
    bucket = params.get("trademin")
    for sym in syms:
        if bucket is not None:
            rows = [{"tradedate": day, "symbol": sym, "trademin": bucket,
                     "open": "1", "high": "1", "low": "1",
                     "close": str(int(bucket) * 10 + 0.5),
                     "vol": "10", "amount": "10000"}]
        else:
            rows = [{"tradedate": day, "symbol": sym, "trademin": "0930",
                     "open": "1", "high": "1", "low": "1", "close": "10",
                     "vol": "10", "amount": "10000"},
                    {"tradedate": day, "symbol": sym, "trademin": "1500",
                     "open": "1", "high": "1", "low": "1", "close": "11",
                     "vol": "10", "amount": "10000"}]
        out.extend(rows)
    return out


def test_intraday_batch_whole_day_single_chunk(monkeypatch):
    monkeypatch.setattr(meoz_mod, "_cn_now", lambda: datetime(2026, 9, 2, 10, 5))

    def handler(apiname, params):
        if apiname == "minute":
            return _intraday_rows(params)
        return []

    prov, client = make_provider(handler)
    df = prov.get_intraday_batch(["600519.SH", "000001.SZ", "300750.SZ"])
    assert df.columns == MINUTE_COLS
    assert df.height == 6
    exp_symbols = sorted(["600519.SH", "000001.SZ", "300750.SZ"])
    assert df["symbol"].to_list() == [s for s in exp_symbols for _ in (0, 1)]
    assert df["datetime"].to_list() == [
        datetime(2026, 9, 2, 9, 30), datetime(2026, 9, 2, 15, 0)] * 3
    assert df["volume"].to_list() == [10.0] * 6
    # 修复轮: 整日窗口单请求(无 trademin), 入参为纯代码
    call = next(p for a, p in client.calls if a == "minute")
    assert call == {"symbols": "600519,000001,300750", "tradedate": DAY2}


def test_intraday_batch_empty_and_wrong_asset(monkeypatch):
    monkeypatch.setattr(meoz_mod, "_cn_now", lambda: datetime(2026, 9, 2, 10, 5))
    prov, client = make_provider(lambda *_: None)
    df = prov.get_intraday_batch([])
    assert df.height == 0 and not client.calls
    with pytest.raises(meoz_mod.MeoZError):
        prov.get_intraday_batch(["510300.SH"], asset_type="etf")


def test_intraday_latest_buckets_full_market(monkeypatch):
    monkeypatch.setattr(meoz_mod, "_cn_now", lambda: datetime(2026, 9, 2, 10, 5))

    def handler(apiname, params):
        if apiname == "minute":
            return _intraday_rows(params, syms=("600519.SH", "000001.SZ"))
        return []

    prov, client = make_provider(handler)
    df = prov.get_intraday_latest()   # count=3 → 桶 1003/1004/1005
    buckets = [p["trademin"] for a, p in client.calls if a == "minute"]
    assert buckets == ["1003", "1004", "1005"]
    assert all(p.get("symbols") is None for a, p in client.calls if a == "minute")
    assert df.height == 6
    assert sorted(df["symbol"].unique().to_list()) == ["000001.SZ", "600519.SH"]
    assert df["datetime"].to_list() == [
        datetime(2026, 9, 2, 10, 3), datetime(2026, 9, 2, 10, 4),
        datetime(2026, 9, 2, 10, 5)] * 2


def test_intraday_latest_symbols_filter_and_offhours(monkeypatch):
    monkeypatch.setattr(meoz_mod, "_cn_now", lambda: datetime(2026, 9, 2, 10, 5))

    def handler(apiname, params):
        if apiname == "minute":
            return _intraday_rows(params)
        return []

    prov, client = make_provider(handler)
    df = prov.get_intraday_latest(symbols=["000001.SZ"])
    assert df.height == 3
    assert df["symbol"].unique().to_list() == ["000001.SZ"]

    # 盘前无有效桶 → 空表且不请求
    monkeypatch.setattr(meoz_mod, "_cn_now", lambda: datetime(2026, 9, 2, 9, 0))
    client.calls.clear()
    df = prov.get_intraday_latest()
    assert df.height == 0 and not client.calls


# ---------------------------------------------------------------- 分钟桶 label helpers
def test_session_label_helpers():
    assert meoz_mod._minute_label_valid(930)
    assert meoz_mod._minute_label_valid(931)
    assert meoz_mod._minute_label_valid(1130)
    assert meoz_mod._minute_label_valid(1301)
    assert meoz_mod._minute_label_valid(1500)
    assert not meoz_mod._minute_label_valid(900)
    assert not meoz_mod._minute_label_valid(1300)  # 午休首分钟(实测标签自 1301 起)
    assert meoz_mod._recent_minute_labels(datetime(2026, 9, 2, 10, 5), 3) == [1003, 1004, 1005]
    assert meoz_mod._recent_minute_labels(datetime(2026, 9, 2, 13, 0), 3) == [1128, 1129, 1130]
    assert meoz_mod._session_label_count(datetime(2026, 9, 2, 9, 35)) == 6   # 0930~0935
    assert meoz_mod._session_label_count(datetime(2026, 9, 2, 9, 0)) == 0
    assert meoz_mod._session_label_count(datetime(2026, 9, 2, 15, 30)) == 241


# ---------------------------------------------------------------- depth5(五档盘口)
def _l2_row(sym="600000", t="145959000", bid_vol=505700, ask_vol=14300,
            bid=9.23, ask=9.24):
    return {"symbol": sym, "time": t, "bid1": bid, "ask1": ask,
            "bid_vol1": bid_vol, "ask_vol1": ask_vol}


def test_depth_batch_maps_latest_valid_snapshot_to_lots():
    rows = [
        _l2_row(t="093500000", bid_vol=10000, ask_vol=0),      # 涨停形态: 卖侧空
        _l2_row(t="145959000", bid_vol=505700, ask_vol=14300),  # 最新有效快照
        # 盘后定价簿残留(15:00+): 双零空簿, 应被跳过
        _l2_row(t="151500000", bid_vol=0, ask_vol=0),
    ]

    def handler(apiname, params):
        if apiname == "level2_quote_history":
            assert params["symbol"] == "600000"
            assert params["end_time"] == meoz_mod._DEPTH_SESSION_END
            return rows

    prov, client = make_provider(handler)
    out = prov.get_depth_batch(["600000.SH"])
    assert list(out) == ["600000.SH"]
    d = out["600000.SH"]
    assert d["ask_volumes"] == [143.0, None, None, None, None]  # 股→手
    assert d["bid_volumes"] == [5057.0, None, None, None, None]
    assert isinstance(d["timestamp"], int)


def test_depth_batch_keeps_limit_board_single_side_zero():
    """最新快照为涨停形态(卖一无挂单)时保留 0, 不得回退到更早的有效行。"""

    def handler(apiname, params):
        if apiname == "level2_quote_history":
            return [
                _l2_row(t="100000000", bid_vol=9000, ask_vol=8000),
                _l2_row(t="143000000", bid_vol=999900, ask_vol=0),
            ]

    prov, _ = make_provider(handler)
    out = prov.get_depth_batch(["000001.SZ"])
    d = out["000001.SZ"]
    assert d["ask_volumes"][0] == 0.0
    assert d["bid_volumes"][0] == 9999.0


def test_depth_batch_empty_book_and_unsupported_codes():
    def handler(apiname, params):
        if apiname == "level2_quote_history":
            if params["symbol"] == "600000":
                return [_l2_row(bid_vol=0, ask_vol=0)]  # 全天空簿
            return []  # 无数据

    prov, _ = make_provider(handler)
    out = prov.get_depth_batch(["600000.SH", "830001.BJ", "600000.SH"])
    assert out == {}


def test_depth_batch_request_failures_are_skipped():
    calls = {"n": 0}

    def handler(apiname, params):
        if apiname == "level2_quote_history":
            calls["n"] += 1
            if calls["n"] == 1:
                raise meoz_mod.MeoZError("boom")
            return [_l2_row()]

    prov, _ = make_provider(handler)
    out = prov.get_depth_batch(["600000.SH", "000001.SZ"])
    assert set(out) == {"000001.SZ"}


def test_depth_batch_dedup_and_exchange_suffix():
    seen: list[str] = []

    def handler(apiname, params):
        if apiname == "level2_quote_history":
            seen.append(params["symbol"])
            return [_l2_row()]

    prov, _ = make_provider(handler)
    out = prov.get_depth_batch(["600000.SH", "600000"])
    assert list(out) == ["600000.SH"]
    assert seen == ["600000"]  # 去重后单次请求, 纯代码入参
