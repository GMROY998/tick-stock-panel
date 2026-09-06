"""eltdx-gateway provider: A 股 日K / 1分钟K / 实时快照 / 实时五档盘口。

数据集契约 (与 kline_sync / quote_service / depth_service 消费端一致):
- daily    不复权日K: bars.get(period=day, adjust=none) 逐票并发,
           volume=手(volume_lots) / amount=元, symbol 带交易所后缀回标;
           窗口 >800 根时 all_pages 分页(网关单页上限 800)
- minute   1分钟K: minutes.history(trading_date=...) 逐票逐日并发。
           注意: TDX 主站 1m K 线保留约 99 个交易日且**最新若干天缺失**
           (实测 2026-09 bars.get 1m 最新只到 3 个交易日前), 不能用于
           分钟同步; minutes.history 数据实时但只有分钟价与量, 故
           open/high/low 取分钟价(无日内高低), amount 按价×量(手→股)估算。
- realtime 全市场 A 股快照: codes.all_a_shares + quotes.get_snapshots
           分批(100/批, 4线程), 失败批串行重试一次; 记录 schema 同 fuyao
           realtime; change_pct 小数制, 零价行(长期停牌/无行情)剔除,
           name/amplitude/turnover_rate 置空(交下游维表)。
           实测约 1100 只代码无快照行(退市/长期停牌), 属数据源正常行为。
- depth5   实时五档: quotes.get_depth, sell/buy_levels 五档量(手) →
           ask_volumes/bid_volumes, timestamp 取本地毫秒。
           实现 MarketDataProvider.get_depth_batch 契约(上游 #246 自定义
           depth 源): depth_service 统一按 capset 切片节流, 失败跳过本轮
           不跨源回退; 本方法内部再按 50 只/批并发, 单批失败跳过。

核心指数实时行情由 get_realtime_indices(quotes.get_snapshots 按码) 补充。
北交所 (4/8 开头) 不入网关链路, 逐票跳过并记 warning。
异常抛 EltdxError, 由上层按自定义源异常规则处理 (降级 TickFlow)。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from .client import EltdxClient, EltdxError, to_gw_code, to_symbol

logger = logging.getLogger(__name__)

_DATASETS = ("daily", "minute", "realtime", "depth5")

_MAX_WORKERS = 4          # 网关批量并发(单核限额容器, 并发过高会超时)
_REALTIME_WORKERS = 2     # 全市场快照专用: 56 批串行 ~60ms/批, 2 并发最少排队不丢批
_SNAPSHOT_CHUNK = 100     # quotes.get_snapshots 单批代码数(实测 100 只 ~60ms)
_DEPTH_CHUNK = 50         # quotes.get_depth 单批代码数(五档包更大, 取保守值)
_DAILY_SINGLE_MAX = 800   # bars.get 单请求条数上限(网关 page size 限制)
_MINUTE_PER_DAY = 241     # 每交易日 1 分钟桶数(含竞价首根)
_MINUTE_MAX_DAYS = 120    # get_minute 单次窗口自然日上限(防请求爆炸)
_DEFAULT_WINDOW_DAYS = 365   # get_daily 未给 start_time 时的回看窗口
_MINUTE_DEFAULT_WINDOW_DAYS = 8  # get_minute 未给 start_time 时的回看窗口
_SNAPSHOT_TIMEOUT = 15.0  # 快照单批超时(并发下网关排队, 需宽松)

_DAILY_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
_MINUTE_COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]

# realtime(实时快照)记录 schema —— 与 fuyao realtime / quote_service 消费端一致
_SNAPSHOT_RECORD_KEYS = (
    "symbol", "name", "last_price", "prev_close", "open", "high", "low",
    "volume", "amount", "change_pct", "change_amount", "amplitude",
    "turnover_rate", "timestamp", "session",
)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_date(value: datetime | date | None) -> date | None:
    if value is None:
        return None
    return value.date() if isinstance(value, datetime) else value


def _trading_days_estimate(start: date, end: date) -> int:
    """不依赖网络的交易日估计(仅用于 bars.get 的 count 预算, 误差由 buffer 吸收)。"""
    days = (end - start).days + 1
    if days <= 0:
        return 0
    return max(1, round(days * 250 / 365) + 2)


def _parse_bar_day(value: Any) -> date | None:
    """bars.get 的 time ("2026-09-02T15:00:00+08:00") → date。"""
    s = str(value or "")
    if len(s) >= 10:
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            return None
    return None


def _parse_point_minute(value: Any, day: date) -> datetime | None:
    """minutes.history 点的 time (ISO 带时区) → naive 北京 datetime。"""
    s = str(value or "")
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _gw_codes_of(symbols: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """symbols → [(系统symbol, 网关code)], 不支持的代码进 skipped。"""
    pairs: list[tuple[str, str]] = []
    skipped: list[str] = []
    for s in symbols or []:
        gw = to_gw_code(s)
        if gw:
            pairs.append((s, gw))
        else:
            skipped.append(str(s))
    return pairs, skipped


@dataclass
class _EltdxConfig:
    """轻量 config shim: 让 custom loader 的 provider_has_dataset 识别数据集。"""

    name: str = "eltdx"
    display_name: str = "eltdx 行情网关"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def availability() -> tuple[bool, str]:
    """插件可用性: 探测网关 /health (短超时, 不打数据接口)。"""
    client = EltdxClient()
    try:
        return client.health()
    finally:
        client.close()


class EltdxProvider:
    name = "eltdx"
    builtin = True

    def __init__(self, base_url: str | None = None, max_workers: int = _MAX_WORKERS) -> None:
        self.config = _EltdxConfig()
        self._base_url = base_url
        self._max_workers = max(1, max_workers)
        self._client: EltdxClient | None = None

    def _get_client(self) -> EltdxClient:
        if self._client is None:
            self._client = EltdxClient(base_url=self._base_url)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ==================== 内部: 并发逐票 RPC ====================
    def _run_per_code(
        self,
        pairs: list[tuple[str, str]],
        fn: Callable[[str], list[dict[str, Any]]],
        on_chunk_done: Callable[[int, int], None] | None,
    ) -> list[dict[str, Any]]:
        """并发执行 fn(网关code) → 行列表; 单票失败跳过(记 warning)。"""
        out: list[dict[str, Any]] = []
        done = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(fn, gw): sym for sym, gw in pairs}
            for fut in as_completed(futures):
                sym = futures[fut]
                done += 1
                try:
                    out.extend(fut.result())
                except EltdxError as e:
                    logger.warning("eltdx %s 拉取失败: %s", sym, e)
                if on_chunk_done:
                    on_chunk_done(done, len(pairs))
        return out

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
        end = _as_date(end_time) or date.today()
        start = _as_date(start_time) or (end - timedelta(days=_DEFAULT_WINDOW_DAYS))
        if end < start:
            return pl.DataFrame()

        pairs, skipped = _gw_codes_of(symbols)
        if skipped:
            logger.warning("eltdx daily 跳过不支持代码(北交所等): %s", skipped[:10])
        count = _trading_days_estimate(start, end) + 10

        # 网关单请求 page size ≤800: 小窗口单请求; 大窗口 all_pages 全量后本地过滤
        params: dict[str, Any] = {"period": "day", "adjust": "none"}
        if count <= _DAILY_SINGLE_MAX:
            params["count"] = count
        else:
            params["all_pages"] = True
            params["page_size"] = _DAILY_SINGLE_MAX

        rows_all = self._run_per_code(
            pairs,
            lambda gw: self._daily_rows(
                to_symbol(gw),
                (self._get_client().rpc("bars.get", {"code": gw, **params}) or {}).get("bars") or [],
                start, end),
            on_chunk_done,
        )
        frames: list[pl.DataFrame] = [pl.DataFrame(rows_all)] if rows_all else []
        return self._finalize(frames, _DAILY_COLS, ["symbol", "date"])

    @staticmethod
    def _daily_rows(symbol: str, bars: list[dict[str, Any]],
                    start: date, end: date) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for b in bars:
            d = _parse_bar_day(b.get("time"))
            close = _to_float(b.get("close"))
            if d is None or close is None or not (start <= d <= end):
                continue
            out.append({
                "symbol": symbol,
                "date": d,
                "open": _to_float(b.get("open")),
                "high": _to_float(b.get("high")),
                "low": _to_float(b.get("low")),
                "close": close,
                # volume_lots 单位=手(与日K total_hand 同量纲, 实测 2026-09)
                "volume": _to_float(b.get("volume_lots")),
                "amount": _to_float(b.get("amount")),
            })
        return out

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
            raise EltdxError("eltdx 分钟数据仅支持 A 股股票(ETF/指数请走 TickFlow)")
        if not symbols:
            return pl.DataFrame()
        if freq not in (None, "", "1m", "1min"):
            raise EltdxError(f"eltdx 分钟数据仅支持 1m, 不支持 {freq!r}")
        end = _as_date(end_time) or date.today()
        start = _as_date(start_time) or (end - timedelta(days=_MINUTE_DEFAULT_WINDOW_DAYS))
        if end < start:
            return pl.DataFrame()
        if (end - start).days > _MINUTE_MAX_DAYS:
            start = end - timedelta(days=_MINUTE_MAX_DAYS)

        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        pairs, skipped = _gw_codes_of(symbols)
        if skipped:
            logger.warning("eltdx minute 跳过不支持代码(北交所等): %s", skipped[:10])

        client = self._get_client()

        def fetch_code(gw: str) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for day in days:
                try:
                    res = client.rpc("minutes.history",
                                     {"code": gw, "trading_date": day.isoformat()})
                except EltdxError:
                    # 非交易日/该日无数据 → 跳过该日(其余日不受影响)
                    continue
                rows.extend(self._minute_rows(gw, (res or {}).get("points") or [], day))
            return rows

        rows_all = self._run_per_code(pairs, fetch_code, on_chunk_done)
        frames: list[pl.DataFrame] = [pl.DataFrame(rows_all)] if rows_all else []
        return self._finalize(frames, _MINUTE_COLS, ["symbol", "datetime"],
                              datetime_col="datetime")

    @staticmethod
    def _minute_rows(gw_code: str, points: list[dict[str, Any]],
                     day: date) -> list[dict[str, Any]]:
        """minutes.history 点 → 分钟行。TDX 分时只有分钟价与量:
        open/high/low 取分钟价(无日内高低), volume=手(实测累计与日K一致),
        amount 按价×量(手→股)估算(接口不提供)。"""
        symbol = to_symbol(gw_code)
        out: list[dict[str, Any]] = []
        for p in points:
            price = _to_float(p.get("price"))
            dt = _parse_point_minute(p.get("time"), day)
            if price is None or dt is None:
                continue
            volume = _to_float(p.get("volume")) or 0.0
            out.append({
                "symbol": symbol,
                "datetime": dt,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "amount": price * volume * 100.0,   # 手→股 估算
            })
        return out

    # ==================== 实时行情(全市场快照) ====================
    def get_realtime(self) -> list[dict[str, Any]]:
        """全市场 A 股实时快照(codes.all_a_shares + 分批 get_snapshots)。

        记录 schema 同 fuyao realtime: last_price/pre_close/open/high/low
        直取, volume=手(total_hand), amount=元, change_pct 小数制;
        零价行剔除; name/amplitude/turnover_rate 置空(交下游维表/enriched)。
        单批失败串行重试一次, 仍失败则跳过(其余批不受影响); 全部失败返回 []。
        """
        try:
            codes = self._get_client().rpc("codes.all_a_shares") or []
        except EltdxError as exc:
            logger.warning("eltdx realtime 代码表拉取失败: %s", exc)
            return []
        chunks = [codes[i:i + _SNAPSHOT_CHUNK] for i in range(0, len(codes), _SNAPSHOT_CHUNK)]

        rows: list[dict[str, Any]] = []
        failed: list[list[str]] = []
        with ThreadPoolExecutor(max_workers=min(self._max_workers, _REALTIME_WORKERS)) as pool:
            futures = {
                pool.submit(self._get_client().rpc, "quotes.get_snapshots",
                            {"codes": chunk}, _SNAPSHOT_TIMEOUT): chunk
                for chunk in chunks
            }
            for fut in as_completed(futures):
                try:
                    rows.extend(fut.result() or [])
                except EltdxError:
                    failed.append(futures[fut])
        # 失败批串行重试一次(网关瞬时排队超时的兜底)
        for chunk in failed:
            try:
                rows.extend(self._get_client().rpc("quotes.get_snapshots",
                                                   {"codes": chunk}, _SNAPSHOT_TIMEOUT) or [])
            except EltdxError as e:
                logger.warning("eltdx 快照批重试仍失败(%d 只): %s", len(chunk), e)
        if failed:
            logger.warning("eltdx realtime: %d/%d 批曾失败(已重试)", len(failed), len(chunks))

        fetched_ms = int(time.time() * 1000)
        records: list[dict[str, Any]] = []
        for q in rows:
            rec = self._snapshot_row(q, fetched_ms)
            if rec is not None:
                records.append(rec)
        return records

    @staticmethod
    def _snapshot_row(q: dict[str, Any], fetched_ms: int) -> dict[str, Any] | None:
        """get_snapshots 行 → realtime record (15 键全量 schema, 空缺置 None)。

        零价行(长期停牌/无行情, 实测约 16+ 只)剔除, 避免污染当日 OHLCV。
        """
        code = str(q.get("code") or "").strip()
        exchange = str(q.get("exchange") or "").strip().lower()
        if len(code) != 6 or not code.isdigit() or exchange not in ("sh", "sz"):
            return None
        last = _to_float(q.get("last_price"))
        if not last:  # None 或 0
            return None
        symbol = f"{code}.{exchange.upper()}"
        prev = _to_float(q.get("pre_close_price"))
        change_amount = (last - prev) if prev is not None else None
        change_pct = (change_amount / prev) if (prev not in (None, 0) and change_amount is not None) else None
        rec: dict[str, Any] = {k: None for k in _SNAPSHOT_RECORD_KEYS}
        rec.update({
            "symbol": symbol,
            "last_price": last,
            "prev_close": prev,
            "open": _to_float(q.get("open_price")),
            "high": _to_float(q.get("high_price")),
            "low": _to_float(q.get("low_price")),
            "volume": _to_float(q.get("total_hand")),   # 手
            "amount": _to_float(q.get("amount")),        # 元
            "change_pct": change_pct,
            "change_amount": change_amount,
            "timestamp": fetched_ms,
        })
        return rec

    # ==================== 核心指数实时行情 ====================
    def get_realtime_indices(self, symbols: list[str]) -> list[dict[str, Any]]:
        """核心指数实时快照(quotes.get_snapshots 按码, TDX 快照含指数)。

        指数代码带后缀直接用(.SH→sh/.SZ→sz); 纯 6 位按 39 开头→sz、
        其余→sh 推断(上证 000xxx/中证 880/999 均在 sh)。
        """
        codes: list[str] = []
        for s in symbols or []:
            s = str(s or "").strip()
            if not s:
                continue
            low = s.lower()
            if low.endswith(".sh") or low.endswith(".sz"):
                code, _, ex = low.partition(".")
                codes.append(f"{ex}{code}")
            elif len(low) == 6 and low.isdigit():
                codes.append(("sz" if low.startswith("39") else "sh") + low)
        if not codes:
            return []
        fetched_ms = int(time.time() * 1000)
        try:
            res = self._get_client().rpc("quotes.get_snapshots", {"codes": codes}) or []
        except EltdxError as exc:
            logger.warning("eltdx 指数快照拉取失败: %s", exc)
            return []
        records: list[dict[str, Any]] = []
        for q in res:
            rec = self._snapshot_row(q, fetched_ms)
            if rec is not None:
                records.append(rec)
        return records

    # ============ 实时五档盘口 (depth5 / get_depth_batch 契约, 上游 #246) ============
    def get_depth_batch(self, symbols: list[str]) -> dict[str, dict]:
        """批量取实时五档(MarketDataProvider.get_depth_batch 契约)。

        返回 {symbol: MarketDepth}:
        MarketDepth = {
            "ask_volumes": [卖一..卖五量(手)],
            "bid_volumes": [买一..买五量(手)],
            "timestamp": 本地毫秒 epoch,
        }
        与 depth_service._call_depth_batch 的解析结构一致(sealed 只取首档)。
        切片与节流由 depth_service 统一负责; 本方法内部再按 50 只/批并发,
        单批失败跳过(不跨源回退)。
        """
        pairs, skipped = _gw_codes_of(list(symbols or []))
        if skipped:
            logger.warning("eltdx depth 跳过不支持代码(北交所等): %s", skipped[:10])
        if not pairs:
            return {}
        out: dict[str, dict] = {}
        chunks = [pairs[i:i + _DEPTH_CHUNK] for i in range(0, len(pairs), _DEPTH_CHUNK)]
        fetched_ms = int(time.time() * 1000)
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._get_client().rpc, "quotes.get_depth",
                            {"codes": [gw for _, gw in chunk]}, 8.0): chunk
                for chunk in chunks
            }
            for fut in as_completed(futures):
                try:
                    res = fut.result() or {}
                except EltdxError as e:
                    logger.warning("eltdx depth 批(%d 只)失败: %s", len(futures[fut]), e)
                    continue
                for rec in (res.get("records") or []):
                    entry = self._depth_row(rec, fetched_ms)
                    if entry is not None:
                        out[entry[0]] = entry[1]
        return out

    @staticmethod
    def _depth_row(rec: dict[str, Any], fetched_ms: int) -> tuple[str, dict] | None:
        """get_depth record → (symbol, MarketDepth); 五档量(手)直接映射。"""
        code = str(rec.get("code") or "").strip()
        exchange = str(rec.get("exchange") or "").strip().lower()
        if len(code) != 6 or not code.isdigit() or exchange not in ("sh", "sz"):
            return None
        ask_vols = [lv.get("volume") for lv in (rec.get("sell_levels") or [])][:5]
        bid_vols = [lv.get("volume") for lv in (rec.get("buy_levels") or [])][:5]
        return f"{code}.{exchange.upper()}", {
            "ask_volumes": ask_vols,
            "bid_volumes": bid_vols,
            "timestamp": fetched_ms,
        }

    # ==================== 拼装与设置页"试拉" ====================
    @staticmethod
    def _finalize(
        frames: list[pl.DataFrame],
        cols: list[str],
        unique_keys: list[str],
        datetime_col: str | None = None,
    ) -> pl.DataFrame:
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames, how="diagonal_relaxed")
        df = df.unique(subset=unique_keys, keep="last")
        df = df.sort(unique_keys)
        if datetime_col and datetime_col in df.columns:
            df = df.with_columns(pl.col(datetime_col).cast(pl.Datetime("us")))
        return df.select([c for c in cols if c in df.columns])

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        symbols = (symbols or ["600519.SH", "000001.SZ"])[:3]
        if dataset == "daily":
            df = self.get_daily(symbols, None, None)
            return _preview(self.name, dataset, df)
        if dataset == "minute":
            df = self.get_minute(symbols, None, None)
            return _preview(self.name, dataset, df)
        if dataset == "realtime":
            records = self.get_realtime()
            df = pl.DataFrame(records) if records else pl.DataFrame()
            return _preview(self.name, dataset, df)
        if dataset == "depth5":
            depth = self.get_depth_batch(symbols)
            rows = [
                {"symbol": sym,
                 "bid1_vol": (d["bid_volumes"] or [None])[0],
                 "ask1_vol": (d["ask_volumes"] or [None])[0]}
                for sym, d in list(depth.items())[:5]
            ]
            return {
                "provider": self.name,
                "dataset": dataset,
                "rows": len(depth),
                "columns": ["symbol", "bid1_vol", "ask1_vol"],
                "preview": rows,
            }
        raise ValueError(f"eltdx 不支持数据集: {dataset}")


def _preview(provider: str, dataset: str, df: pl.DataFrame) -> dict:
    return {
        "provider": provider,
        "dataset": dataset,
        "rows": df.height,
        "columns": df.columns,
        "preview": df.head(5).to_dicts() if not df.is_empty() else [],
    }
