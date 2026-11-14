"""顺序策略 (sequential) — 先满足前置策略 A, 再在 N 个交易日内满足触发策略 B。

与叠加策略 (composite) 的区别:
- composite 是**同一交易日**的多策略合并 (union / intersect);
- sequential 是**跨交易日**的时序模式: A 命中后, 最多等 N 个交易日, B 命中才入选。

合并语义 (策略选股与回测共用, 口径必须一致 — CONTRIBUTING §5.1):
- entry:  trigger[t] AND (exists k in [0, N]: pre[t-k])
           即 B 当日命中, 且 A 在当日或之前 N 个交易日内命中过。
           N=0 表示只认同日命中。
- exit:   来源投影, 只用 trigger 的 exit, 且仅在 trigger 自己的持仓窗口内生效,
           不会用 pre 的卖点平掉仓位。
- score:  直接取 trigger 当日的 score (不再做跨子策略的排名融合)。
- 多个 A 命中: 取最近一次 A (gap 最小)。
- A→B 之间出现 A 的卖出信号: 不作废 (按需求约定)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# 允许的最大等待窗口 (交易日)。过大没有业务意义, 只增加历史窗口与计算量。
MAX_GAP_DAYS = 60


@dataclass(frozen=True)
class SequentialSpec:
    """顺序策略的声明式配置 (无业务代码)。"""

    pre_strategy_id: str
    trigger_strategy_id: str
    max_gap_days: int


def parse_spec(raw: Any) -> SequentialSpec:
    """解析 META["sequential"]。

    形如: {"pre": "strategy_a", "trigger": "strategy_b", "max_gap_days": 3}
    兼容 pre_strategy_id / trigger_strategy_id 长字段名。
    """
    if not isinstance(raw, dict):
        raise ValueError("sequential strategy META['sequential'] must be a dict")
    pre = raw.get("pre") or raw.get("pre_strategy_id")
    trigger = raw.get("trigger") or raw.get("trigger_strategy_id")
    if not isinstance(pre, str) or not pre:
        raise ValueError("sequential META['sequential']['pre'] must be a non-empty strategy id")
    if not isinstance(trigger, str) or not trigger:
        raise ValueError(
            "sequential META['sequential']['trigger'] must be a non-empty strategy id"
        )
    if pre == trigger:
        raise ValueError("sequential pre and trigger must be different strategies")
    try:
        gap = int(raw.get("max_gap_days", 3))
    except (TypeError, ValueError) as exc:
        raise ValueError("sequential max_gap_days must be an integer") from exc
    if gap < 0 or gap > MAX_GAP_DAYS:
        raise ValueError(f"sequential max_gap_days must be within [0, {MAX_GAP_DAYS}]")
    return SequentialSpec(
        pre_strategy_id=pre,
        trigger_strategy_id=trigger,
        max_gap_days=gap,
    )


def result_symbols(result: Any) -> set[str]:
    """从 StrategyResult 取命中标的集合 (scores 优先, rows 兜底)。"""
    symbols: set[str] = {str(sym) for sym in (getattr(result, "scores", None) or {})}
    for row in getattr(result, "rows", None) or []:
        if isinstance(row, dict) and row.get("symbol"):
            symbols.add(str(row["symbol"]))
    return symbols


def match_events(
    dates: list[str],
    pre_hits: dict[str, set[str]],
    trigger_hits: dict[str, set[str]],
    max_gap_days: int,
) -> dict[str, dict[str, int]]:
    """按交易日推进状态机, 返回 {触发日: {symbol: gap_days}}。

    dates 必须升序。gap 为触发日与最近一次前置命中的交易日距离 (0 = 同日)。
    A→B 之间是否出现卖点不影响 (按需求约定不作废)。
    """
    last_pre: dict[str, int] = {}
    events: dict[str, dict[str, int]] = {}
    for idx, day in enumerate(dates):
        # 同日命中算 gap=0, 所以先记录当日前置命中再判定触发。
        for symbol in pre_hits.get(day, ()):  # type: ignore[union-attr]
            last_pre[symbol] = idx
        matched: dict[str, int] = {}
        for symbol in trigger_hits.get(day, ()):  # type: ignore[union-attr]
            pre_idx = last_pre.get(symbol)
            if pre_idx is None:
                continue
            gap = idx - pre_idx
            if 0 <= gap <= max_gap_days:
                matched[symbol] = gap
        if matched:
            events[day] = matched
    return events


def merge_signal_matrices(
    shape: tuple[int, int],
    sigs: list,
    max_gap_days: int,
    max_hold: int,
):
    """回测合并: (pre, trigger) 两个 SignalMatrix → 顺序策略 SignalMatrix。

    entry[t] = trigger.entry[t] & any(pre.entry[t-k] for k in 0..N)
    exit     = trigger.exit & trigger 持仓窗口 (来源投影)
    score    = trigger.score (仅保留入选点)
    """
    from app.backtest.matrix import make_signal_matrix
    from app.strategy.composite import _hold_masks_from_entries

    n_times, n_assets = shape
    if len(sigs) < 2:
        return make_signal_matrix(shape)

    pre, trigger = sigs[0], sigs[1]
    pre_entry = np.asarray(pre.entry, dtype=bool)
    trig_entry = np.asarray(trigger.entry, dtype=bool)

    # 最近 N 个交易日内 A 是否命中过 (含当日)。用切片位移, 不环绕防未来泄漏。
    pre_recent = np.zeros(shape, dtype=bool)
    for gap in range(max_gap_days + 1):
        if gap == 0:
            pre_recent |= pre_entry
        elif gap < n_times:
            pre_recent[gap:, :] |= pre_entry[:-gap, :]

    merged_entry = trig_entry & pre_recent

    # 退出: 只用 trigger 自己的卖点, 且限定在它的持仓窗口内。
    hold_mask = _hold_masks_from_entries([trig_entry], max_hold)[0]
    merged_exit = np.asarray(trigger.exit, dtype=bool) & hold_mask

    trigger_score = np.asarray(trigger.score, dtype=np.float64)
    merged_score = np.where(merged_entry, trigger_score, 0.0)
    merged_score = np.nan_to_num(merged_score, nan=0.0, posinf=0.0, neginf=0.0)

    entry_codes = np.where(merged_entry, 1, -1).astype(np.int16)
    exit_codes = np.full(shape, -1, dtype=np.int16)

    return make_signal_matrix(
        shape,
        entry=merged_entry.astype(np.uint8),
        exit=merged_exit.astype(np.uint8),
        score=merged_score.astype(np.float32),
        entry_signal_code=entry_codes,
        exit_signal_code=exit_codes,
        entry_signal_ids=("sequential:pre", "sequential:trigger"),
    )
