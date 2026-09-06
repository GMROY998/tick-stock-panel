"""MeoZ(猫爪数据) provider: A 股股票 日K / 除权因子 / 1 分钟K / 实时快照 / 全量分钟。

数据集契约(与 kline_sync/quote_service 消费端一致):
- daily      不复权日K: symbol/date/open/high/low/close/volume(手)/amount(元)
- adj_factor 除权事件行: symbol/trade_date/ex_factor, ex_factor=当日因子/前一日因子比值,
             仅因子变化日产出(同 fuyao 语义)
- minute     1分钟K: symbol/datetime(naive 北京)/open/high/low/close/volume(手)/amount(元)
- realtime   全市场 A 股实时快照(tick 单请求), 记录 schema 同 fuyao realtime;
             volume 透传=手(实测 cat tick/minute/daily vol 同单位且与日终手数一致);
             change_pct 已转小数制(0.0174=+1.74%), prev_close 由 close 反推;
             name/amplitude/turnover_rate 置空(交下游维表/enriched)。
- full_minute 当日分钟全量+增量: get_intraday_batch(修复轮, 整日窗口) 与
             get_intraday_latest(稳态增量, 最近 count 个分钟桶的全市场快照);
             桶标签 0930~1130 / 1301~1500(实测 241 根/票/日, volume=手)。

仅覆盖 A 股股票(asset_type=="stock"); ETF/指数由上层回退 TickFlow,
核心指数实时行情由 get_realtime_indices(index_snapshot) 单独补拉。
异常抛 MeoZError, 由上层按自定义源异常规则处理。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from typing import Any

import polars as pl

from app import secrets_store

from .client import MeoZClient, MeoZError

logger = logging.getLogger(__name__)

_DATASETS = ("daily", "adj_factor", "minute", "realtime", "full_minute")

API_KEY_ENV = "MEOZ_API_KEY"
SECRETS_FIELD = "meoz_api_key"  # UI 配置的 Key 存 secrets.json, 优先级高于 .env
CONFIG_FILE = os.path.expanduser("~/.meoz/config.json")  # 兜底: skill 客户端本机 Key

_ROW_BUDGET = 6000            # cat 单次查询行数上限(分钟接口明确; 日K保守沿用)
_MINUTE_ROWS_PER_SYM_DAY = 241
_MINUTE_CHUNK = max(1, _ROW_BUDGET // _MINUTE_ROWS_PER_SYM_DAY - 2)
_DAILY_BUFFER_DAYS = 15       # 除权事件基准需窗口前若干自然日(覆盖长假)
_ADJ_TOL = 1e-4               # 除权事件判定阈值(过滤因子序列舍入噪声)
_DAILY_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
_ADJ_COLS = ["symbol", "trade_date", "ex_factor"]
_MINUTE_COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]

# realtime(实时快照)记录 schema —— 与 fuyao realtime / quote_service 消费端一致
_SNAPSHOT_RECORD_KEYS = (
    "symbol", "name", "last_price", "prev_close", "open", "high", "low",
    "volume", "amount", "change_pct", "change_amount", "amplitude",
    "turnover_rate", "timestamp", "session",
)
# full_minute 当日分钟桶标签范围(实测 2026-09): 早盘含 0930 开盘/竞价首根,
# 午盘自 1301 起至 1500, 全天 241 根/票(与 _MINUTE_ROWS_PER_SYM_DAY 一致)
_SESSION_LABEL_RANGES = ((930, 1130), (1301, 1500))
_CN_TZ = timezone(timedelta(hours=8))

_EX_SUFFIXES = (".SH", ".SZ", ".BJ")


def _strip_exchange(symbol: str) -> str:
    """去掉交易所后缀, 保留 6 位证券代码(cat 查询入参用纯代码)。"""
    s = str(symbol).strip()
    for suf in _EX_SUFFIXES:
        if s.upper().endswith(suf):
            return s[:-len(suf)]
    return s


def _symbol_map(symbols: list[str]) -> dict[str, str]:
    """纯代码 -> 原始代码(带后缀), 回标 cat 返回的纯代码。"""
    out: dict[str, str] = {}
    for s in symbols or []:
        out.setdefault(_strip_exchange(s), s)
    return out


def _remap_symbol(raw: Any, symbol_map: dict[str, str]) -> str:
    raw_s = str(raw or "").strip()
    if not raw_s:
        return ""
    return symbol_map.get(_strip_exchange(raw_s), raw_s)


def _parse_date(value: Any) -> date | None:
    """YYYYMMDD / YYYY-MM-DD / date -> date。"""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    s = str(value).strip().replace("-", "")
    if len(s) < 8:
        return None
    try:
        return datetime.strptime(s[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _parse_hm(value: Any, fallback: Any = None) -> dtime | None:
    """把 trademin(HHMM/四位字符串) 或 time(HH:MM:SS) 解析成 time。"""
    for v in (value, fallback):
        if v is None:
            continue
        s = str(v).strip()
        if ":" in s:
            try:
                return datetime.strptime(s, "%H:%M:%S").time() if len(s) >= 8 \
                    else datetime.strptime(s, "%H:%M").time()
            except ValueError:
                continue
        digits = s.replace(":", "")
        digits = digits.zfill(4)[:4]
        if len(digits) == 4:
            try:
                return dtime(int(digits[:2]), int(digits[2:]))
            except ValueError:
                continue
    return None


def _yyyymmdd(d: date) -> str:
    return f"{d:%Y%m%d}"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dt(value: datetime | date | None) -> datetime:
    if value is None:
        return datetime.now()
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, dtime.min)


def _trading_days_estimate(start: date, end: date) -> int:
    """不依赖网络的交易日估计(仅分批预算)。"""
    days = (end - start).days + 1
    if days <= 0:
        return 0
    return max(1, round(days * 250 / 365) + 2)


def _cn_now() -> datetime:
    """北京墙钟(naive), 与分钟契约的 naive 北京时间一致, 不依赖部署机时区。"""
    return datetime.now(_CN_TZ).replace(tzinfo=None)


def _minute_label_valid(label: int) -> bool:
    """label 是否为当日有效分钟桶(整数 HHMM, 如 931 / 1301)。"""
    return any(lo <= label <= hi for lo, hi in _SESSION_LABEL_RANGES)


def _recent_minute_labels(now: datetime, count: int) -> list[int]:
    """盘中已出现(含当前形成中)的近 count 个有效分钟桶标签, 时间升序。"""
    out: list[int] = []
    cur = now.hour * 60 + now.minute
    while cur >= 0 and len(out) < count:
        label = (cur // 60) * 100 + cur % 60
        if _minute_label_valid(label):
            out.append(label)
        cur -= 1
    out.reverse()
    return out


def _session_label_count(now: datetime) -> int:
    """当日截止 now 已产生的分钟桶数(≤241; 未开盘/早盘前为 0)。"""
    n = 0
    for t in range(now.hour * 60 + now.minute + 1):
        if _minute_label_valid((t // 60) * 100 + t % 60):
            n += 1
    return n


def _snapshot_record(symbol: str, name: Any, last: float, prev: float | None,
                     open_: float | None, high: float | None, low: float | None,
                     volume: float | None, amount: float | None,
                     change_pct: float | None, change_amount: float | None,
                     fetched_ms: int) -> dict[str, Any]:
    """realtime 记录骨架: 15 键全量 schema, 空缺字段置 None。"""
    rec = {k: None for k in _SNAPSHOT_RECORD_KEYS}
    rec.update({
        "symbol": symbol,
        "name": name,
        "last_price": last,
        "prev_close": prev,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": amount,
        "change_pct": change_pct,
        "change_amount": change_amount,
        "timestamp": fetched_ms,
    })
    return rec


def _tick_snapshot_row(row: dict[str, Any], fetched_ms: int) -> dict[str, Any] | None:
    """cat tick(盘中/收盘快照)行 -> realtime record。

    实测(2026-09): vol/amount 与 minute/daily 同单位(手/元), 直接透传;
    pct_chg 为百分数数值(-0.1585 即 -0.1585%), 需转小数制; tick 无 pre_close
    字段, prev_close 由 last/(1+pct/100) 反推。name/amplitude/turnover 留空。
    """
    symbol = str(row.get("symbol") or row.get("code") or "").strip()
    if not symbol:
        return None
    last = _to_float(row.get("close"))
    if last is None:
        return None
    pct_raw = _to_float(row.get("pct_chg"))
    prev = None
    if pct_raw is not None:
        denom = 1.0 + pct_raw / 100.0
        if abs(denom) > 1e-9:
            prev = last / denom
    change_amount = last - prev if prev is not None else None
    change_pct = pct_raw / 100.0 if pct_raw is not None else None
    if change_pct is None and change_amount is not None and prev not in (None, 0):
        change_pct = change_amount / prev
    return _snapshot_record(
        symbol, None, last, prev,
        _to_float(row.get("open")), _to_float(row.get("high")), _to_float(row.get("low")),
        _to_float(row.get("vol")), _to_float(row.get("amount")),
        change_pct, change_amount, fetched_ms,
    )


def _index_snapshot_row(row: dict[str, Any], fetched_ms: int) -> dict[str, Any] | None:
    """cat index_snapshot 行 -> realtime record(核心指数)。

    入参为纯 6 位代码(000001/399001), 返回 symbol 自带 .SH/.SZ; pre_close/name/
    change 字段齐全, pct_chg 同为百分数数值。
    """
    symbol = str(row.get("symbol") or row.get("code") or "").strip()
    if not symbol:
        return None
    last = _to_float(row.get("close"))
    if last is None:
        return None
    prev = _to_float(row.get("pre_close"))
    pct_raw = _to_float(row.get("pct_chg"))
    change_amount = _to_float(row.get("change"))
    if change_amount is None and last is not None and prev is not None:
        change_amount = last - prev
    change_pct = pct_raw / 100.0 if pct_raw is not None else None
    if change_pct is None and change_amount is not None and prev not in (None, 0):
        change_pct = change_amount / prev
    return _snapshot_record(
        symbol, row.get("name"), last, prev,
        _to_float(row.get("open")), _to_float(row.get("high")), _to_float(row.get("low")),
        _to_float(row.get("vol")), _to_float(row.get("amount")),
        change_pct, change_amount, fetched_ms,
    )


@dataclass
class _MeoZConfig:
    """轻量 config shim: 让 custom loader 的 provider_has_dataset 识别数据集。"""

    name: str = "meoz"
    display_name: str = "猫爪 MeoZ"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def get_api_key() -> str | None:
    """Key 优先级: secrets.json(UI) > 环境变量 > ~/.meoz/config.json。"""
    key = secrets_store.get_env_backed_secret(SECRETS_FIELD, API_KEY_ENV)
    if key:
        return key
    try:
        import json
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return (json.load(f).get("api_key") or "").strip() or None
    except Exception:
        return None


def availability() -> tuple[bool, str]:
    """插件可用性: 以 Key 是否可取得为准(不发起网络请求)。"""
    if not get_api_key():
        return False, f"缺少 API Key(可配置环境变量 {API_KEY_ENV} 或在设置页填写)"
    return True, "ok"


def probe_api_key(api_key: str) -> tuple[bool, str]:
    """设置页“测试”: 真实请求一次日K验证 Key 有效性。"""
    client = MeoZClient(api_key=api_key)
    try:
        rows = client.rows(
            "daily", ["tradedate", "symbol", "close"],
            {"symbols": "600519", "recentdays": 1},
        )
        return True, f"Key 有效, 返回 {len(rows)} 行"
    except MeoZError as exc:
        return False, str(exc)
    finally:
        client.close()


class MeoZProvider:
    name = "meoz"
    builtin = True

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.config = _MeoZConfig()
        self._api_key = api_key
        self._base_url = base_url
        self._client: MeoZClient | None = None

    def _get_client(self) -> MeoZClient:
        if self._client is None:
            self._client = MeoZClient(api_key=self._api_key or get_api_key(),
                                      base_url=self._base_url)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _trading_days(self, start: date, end: date) -> list[date]:
        """区间内交易日列表(升序)。tradecal 只返回 is_open=1 的行。"""
        rows = self._get_client().rows(
            "tradecal", ["cal_date", "is_open"],
            {"startdate": _yyyymmdd(start), "enddate": _yyyymmdd(end)},
        )
        days = []
        for r in rows:
            if str(r.get("is_open")) != "1":
                continue
            d = _parse_date(r.get("cal_date"))
            if d is not None:
                days.append(d)
        days.sort()
        return days

    # ==================== 日 K ====================
    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | date | None = None,
        end_time: datetime | date | None = None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if asset_type != "stock" or not symbols:
            return pl.DataFrame()
        end = _as_dt(end_time).date()
        start = _as_dt(start_time).date()
        if end < start:
            return pl.DataFrame()
        if start_time is None:
            start = end - timedelta(days=365)

        code_map = _symbol_map(symbols)
        codes = list(code_map.keys())
        try:
            n_days = max(1, len(self._trading_days(start, end)))
        except MeoZError:
            n_days = max(1, _trading_days_estimate(start, end))
        chunk_size = max(1, min(len(codes), _ROW_BUDGET // n_days))

        frames: list[pl.DataFrame] = []
        processed = 0
        params = {"startdate": _yyyymmdd(start), "enddate": _yyyymmdd(end)}
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i + chunk_size]
            rows = self._get_client().rows(
                "daily",
                ["tradedate", "symbol", "open", "high", "low", "close", "vol", "amount"],
                {"symbols": ",".join(chunk), **params},
            )
            mapped = []
            for r in rows:
                d = _parse_date(r.get("tradedate"))
                if d is None or not (start <= d <= end):
                    continue
                close = _to_float(r.get("close"))
                if close is None:
                    continue
                mapped.append({
                    "symbol": _remap_symbol(r.get("symbol"), code_map),
                    "date": d,
                    "open": _to_float(r.get("open")),
                    "high": _to_float(r.get("high")),
                    "low": _to_float(r.get("low")),
                    "close": close,
                    "volume": _to_float(r.get("vol")),
                    "amount": _to_float(r.get("amount")),
                })
            if mapped:
                frames.append(pl.DataFrame(mapped))
            processed += len(chunk)
            if on_chunk_done:
                on_chunk_done(processed, len(codes))
        return self._finalize_daily(frames)

    @staticmethod
    def _finalize_daily(frames: list[pl.DataFrame]) -> pl.DataFrame:
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames, how="diagonal_relaxed")
        df = df.unique(subset=["symbol", "date"], keep="last")
        df = df.sort(["symbol", "date"])
        return df.select([c for c in _DAILY_COLS if c in df.columns])

    # ==================== 除权因子事件 ====================
    def get_adj_factors(
        self,
        symbols: list[str],
        start_time: datetime | date | None = None,
        end_time: datetime | date | None = None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if asset_type != "stock" or not symbols:
            return pl.DataFrame()
        end = _as_dt(end_time).date()
        start = _as_dt(start_time).date()
        if end < start:
            return pl.DataFrame()
        if start_time is None:
            start = end - timedelta(days=365)
        # 推导 start 当日事件需要 start 之前交易日的因子做基准
        fetch_start = start - timedelta(days=_DAILY_BUFFER_DAYS)

        code_map = _symbol_map(symbols)
        codes = list(code_map.keys())
        try:
            n_days = max(1, len(self._trading_days(fetch_start, end)))
        except MeoZError:
            n_days = max(1, _trading_days_estimate(fetch_start, end))
        chunk_size = max(1, min(len(codes), _ROW_BUDGET // n_days))

        frames: list[pl.DataFrame] = []
        processed = 0
        params = {"startdate": _yyyymmdd(fetch_start), "enddate": _yyyymmdd(end)}
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i + chunk_size]
            rows = self._get_client().rows(
                "adj_factor",
                ["tradedate", "symbol", "adj_factor"],
                {"symbols": ",".join(chunk), **params},
            )
            events = self._adj_events_from_rows(rows, code_map, start, end)
            if events:
                frames.append(pl.DataFrame(events))
            processed += len(chunk)
            if on_chunk_done:
                on_chunk_done(processed, len(codes))
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames, how="diagonal_relaxed")
        df = df.unique(subset=["symbol", "trade_date"], keep="last")
        df = df.sort(["symbol", "trade_date"])
        return df.select([c for c in _ADJ_COLS if c in df.columns])

    @staticmethod
    def _adj_events_from_rows(
        rows: list[dict[str, Any]],
        code_map: dict[str, str],
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        """逐交易日复权因子序列 -> 除权事件行(仅因子变化日)。

        cat 的 adj_factor 为“当日复权因子”(非交易日不产出行, 变化日=除权日)。
        ex_factor = 当日因子 / 前一个交易日因子, 同 fuyao 单事件比值语义。
        """
        per_symbol: dict[str, list[tuple[date, float]]] = {}
        for r in rows:
            sym = _remap_symbol(r.get("symbol"), code_map)
            if not sym:
                continue
            d = _parse_date(r.get("tradedate"))
            factor = _to_float(r.get("adj_factor"))
            if d is None or factor is None or factor <= 0:
                continue
            per_symbol.setdefault(sym, []).append((d, factor))
        events: list[dict[str, Any]] = []
        for sym, series in per_symbol.items():
            series.sort()
            prev_factor: float | None = None
            for d, factor in series:
                if prev_factor is not None:
                    ratio = factor / prev_factor
                    # 容差: cat 因子序列存在 ~1e-5 的舍入噪声(实测产生 1.2e-5 伪变化),
                    # 而真实 A 股除权日因子变化通常 >1e-4。基准仍随序列推进, 滤掉的
                    # 微噪声对后续真实事件比值的误差 ~1e-5, 可忽略。
                    if abs(ratio - 1.0) > _ADJ_TOL and start <= d <= end:
                        events.append({"symbol": sym, "trade_date": d, "ex_factor": ratio})
                prev_factor = factor
        return events

    # ==================== 1 分钟 K ====================
    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | date | None = None,
        end_time: datetime | date | None = None,
        asset_type: str = "stock",
        freq: str = "1m",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if asset_type != "stock":
            # 抛异常让上层回退 TickFlow(provider 异常契约)
            raise MeoZError("MeoZ 分钟数据仅支持 A 股股票(ETF/指数请走 TickFlow)")
        if not symbols:
            return pl.DataFrame()
        if freq not in (None, "", "1m", "1min"):
            raise MeoZError(f"MeoZ 分钟数据仅支持 1m, 不支持 {freq!r}")
        end = _as_dt(end_time)
        start = _as_dt(start_time)
        if end.date() < start.date():
            return pl.DataFrame()
        if start_time is None:
            start = end - timedelta(days=8)

        # 分钟接口按交易日查询, 需精确枚举区间内交易日
        days = self._trading_days(start.date(), end.date())
        if not days:
            return pl.DataFrame()

        code_map = _symbol_map(symbols)
        codes = list(code_map.keys())
        chunks = [codes[i:i + _MINUTE_CHUNK] for i in range(0, len(codes), _MINUTE_CHUNK)]

        frames: list[pl.DataFrame] = []
        total = len(chunks) * len(days)
        step = 0
        for day in days:
            for chunk in chunks:
                step += 1
                rows = self._get_client().rows(
                    "minute",
                    ["tradedate", "symbol", "trademin", "open",
                     "high", "low", "close", "vol", "amount"],
                    {"symbols": ",".join(chunk), "tradedate": _yyyymmdd(day)},
                )
                mapped = self._minute_rows(rows, code_map, day)
                if mapped:
                    frames.append(pl.DataFrame(mapped))
                if on_chunk_done:
                    on_chunk_done(step, total)
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames, how="diagonal_relaxed")
        df = df.unique(subset=["symbol", "datetime"], keep="last")
        df = df.sort(["symbol", "datetime"])
        df = df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
        return df.select([c for c in _MINUTE_COLS if c in df.columns])

    @staticmethod
    def _minute_rows(
        rows: list[dict[str, Any]],
        code_map: dict[str, str],
        default_day: date,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in rows:
            sym = _remap_symbol(r.get("symbol"), code_map)
            if not sym:
                continue
            day = _parse_date(r.get("tradedate")) or default_day
            hm = _parse_hm(r.get("trademin"), r.get("time"))
            if hm is None:
                continue
            close = _to_float(r.get("close"))
            if close is None:
                continue
            out.append({
                "symbol": sym,
                "datetime": datetime.combine(day, hm),
                "open": _to_float(r.get("open")),
                "high": _to_float(r.get("high")),
                "low": _to_float(r.get("low")),
                "close": close,
                "volume": _to_float(r.get("vol")),
                "amount": _to_float(r.get("amount")),
            })
        return out

    # ==================== 实时行情(全市场快照) ====================
    def get_realtime(self) -> list[dict[str, Any]]:
        """全市场 A 股实时快照(tick, 单请求约 5.5k 行), 记录 schema 同 fuyao realtime。

        volume/amount 直接透传(实测为手/元); change_pct 已转小数制;
        prev_close 由 close 与 pct_chg 反推; name/amplitude/turnover_rate
        置空(交下游维表/enriched)。请求失败(如休市)返回 []。
        """
        fetched_ms = int(time.time() * 1000)
        try:
            rows = self._get_client().rows(
                "tick",
                ["symbol", "open", "high", "low", "close", "vol", "amount", "pct_chg"],
                {"type": 1},
            )
        except MeoZError as exc:
            logger.warning("MeoZ realtime tick 拉取失败: %s", exc)
            return []
        records: list[dict[str, Any]] = []
        for r in rows:
            rec = _tick_snapshot_row(r, fetched_ms)
            if rec is not None:
                records.append(rec)
        return records

    def get_realtime_indices(self, symbols: list[str]) -> list[dict[str, Any]]:
        """核心指数实时快照(index_snapshot, 纯代码入参), schema 同 get_realtime。"""
        codes = [_strip_exchange(s) for s in (symbols or []) if str(s or "").strip()]
        if not codes:
            return []
        fetched_ms = int(time.time() * 1000)
        try:
            rows = self._get_client().rows(
                "index_snapshot",
                ["symbol", "name", "open", "high", "low", "close",
                 "pre_close", "change", "pct_chg", "vol", "amount"],
                {"symbols": ",".join(codes)},
            )
        except MeoZError as exc:
            logger.warning("MeoZ index_snapshot 拉取失败: %s", exc)
            return []
        records: list[dict[str, Any]] = []
        for r in rows:
            rec = _index_snapshot_row(r, fetched_ms)
            if rec is not None:
                records.append(rec)
        return records

    # ==================== 全量分钟(当日) ====================
    def get_intraday_batch(
        self,
        symbols: list[str],
        count: int = 300,
        asset_type: str = "stock",
    ) -> pl.DataFrame:
        """修复轮: 拉当日(北京时间)9:30 竞价首根起至今的整日 1 分钟K。

        交易日盘后调用返回当日完整 241 根(可作收盘补录); 桶标签与 minute 契约
        一致(0930~1130 / 1301~1500, volume=手)。按“当日已产生桶数”自适应分批,
        单请求不超 6000 行。count 仅为接口签名兼容(修复口径无窗口回溯)。
        """
        if asset_type != "stock":
            raise MeoZError("MeoZ 全量分钟仅支持 A 股股票(ETF/指数请走 TickFlow)")
        if not symbols:
            return pl.DataFrame()
        now = _cn_now()
        day = now.date()
        n_labels = max(1, _session_label_count(now))
        chunk_size = max(1, min(len(symbols), _ROW_BUDGET // n_labels))

        code_map = _symbol_map(symbols)
        codes = list(code_map.keys())
        frames: list[pl.DataFrame] = []
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i + chunk_size]
            rows = self._get_client().rows(
                "minute",
                ["tradedate", "symbol", "trademin", "open",
                 "high", "low", "close", "vol", "amount"],
                {"symbols": ",".join(chunk), "tradedate": _yyyymmdd(day)},
            )
            mapped = self._minute_rows(rows, code_map, day)
            if mapped:
                frames.append(pl.DataFrame(mapped))
        return self._finalize_minute_frames(frames)

    def get_intraday_latest(
        self,
        symbols: list[str] | None = None,
        count: int = 3,
    ) -> pl.DataFrame:
        """稳态增量: 最近 count 个有效分钟桶的全市场快照(每桶一次请求)。

        对应猫爪分钟行情“不传 symbols 时返回指定 trademin 的全市场快照”能力,
        单请求约 5.5k 行(≤3×6000 行)。形成中(桶形)的分钟行会被后续轮次
        重复返回, 由上层按 (symbol, datetime) 唯一合并。非交易时段/当日尚无
        有效桶时返回空表。
        """
        now = _cn_now()
        day = now.date()
        labels = _recent_minute_labels(now, count)
        if not labels:
            return pl.DataFrame()
        day_str = _yyyymmdd(day)
        frames: list[pl.DataFrame] = []
        for label in labels:
            rows = self._get_client().rows(
                "minute",
                ["tradedate", "symbol", "trademin", "open",
                 "high", "low", "close", "vol", "amount"],
                {"tradedate": day_str, "trademin": f"{label:04d}"},
            )
            mapped = self._minute_rows(rows, {}, day)
            if mapped:
                frames.append(pl.DataFrame(mapped))
        df = self._finalize_minute_frames(frames)
        if symbols and not df.is_empty():
            df = df.filter(pl.col("symbol").is_in(set(symbols)))
        return df

    @staticmethod
    def _finalize_minute_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
        """拼接(升序去重)分钟帧 -> 契约列 [_MINUTE_COLS]。"""
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames, how="diagonal_relaxed")
        df = df.unique(subset=["symbol", "datetime"], keep="last")
        df = df.sort(["symbol", "datetime"])
        df = df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
        return df.select([c for c in _MINUTE_COLS if c in df.columns])

    # ==================== 设置页“试拉” ====================
    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        symbols = (symbols or ["600519.SH"])[:3]
        if dataset == "daily":
            df = self.get_daily(symbols, None, None)
        elif dataset == "adj_factor":
            df = self.get_adj_factors(symbols, None, None)
        elif dataset == "minute":
            df = self.get_minute(symbols, None, None)
        elif dataset == "realtime":
            records = self.get_realtime()
            df = pl.DataFrame(records) if records else pl.DataFrame()
            return _preview(self.name, dataset, df)
        elif dataset == "full_minute":
            df = self.get_intraday_batch(symbols)
        else:
            raise ValueError(f"MeoZ 不支持数据集: {dataset}")
        return _preview(self.name, dataset, df)


def _preview(provider: str, dataset: str, df: pl.DataFrame) -> dict:
    return {
        "provider": provider,
        "dataset": dataset,
        "rows": df.height,
        "columns": df.columns,
        "preview": df.head(5).to_dicts() if not df.is_empty() else [],
    }
