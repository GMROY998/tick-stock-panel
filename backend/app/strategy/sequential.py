"""顺序策略 (sequential) — 先满足 S1, 再在窗口内满足 S2 … 最多 4 段。

与叠加策略 (composite) 的区别:
- composite 是**同一交易日**的多策略合并 (union / intersect);
- sequential 是**跨交易日**的时序链: 每段命中后, 下一段必须落在
  [最小间隔, 最大间隔] 个交易日内, 最后一段命中日即为入选日。

合并语义 (策略选股与回测共用, 口径必须一致 — CONTRIBUTING §5.1):
- entry:  最后一段命中, 且存在一条完整的链 S1→S2→…→Sk,
          每段间隔都在各自的 [min_gap, max_gap] 内。
- exit:   来源投影, 只用最后一段的 exit, 且仅在其持仓窗口内生效。
- score:  直接取最后一段当日的 score。
- 多个候选链: 每一段取最近一次满足窗口的命中。
- 中间段出现卖点: 不作废 (按需求约定)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# 单段最大等待窗口 (交易日)。过大没有业务意义, 只增加历史窗口与计算量。
MAX_GAP_DAYS = 60
# 顺序策略最大段数
MAX_STAGES = 4


@dataclass(frozen=True)
class SequentialSpec:
    """顺序策略的声明式配置 (无业务代码)。"""

    stages: tuple[str, ...]  # 2..4 个策略 id, 按顺序
    min_gaps: tuple[int, ...]  # 长度 = len(stages)-1
    max_gaps: tuple[int, ...]  # 长度 = len(stages)-1

    @property
    def transitions(self) -> int:
        return len(self.stages) - 1

    @property
    def total_max_gap(self) -> int:
        return sum(self.max_gaps)


def _coerce_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sequential {field} must be an integer") from exc


def parse_spec(raw: Any) -> SequentialSpec:
    """解析 META["sequential"]。

    新格式:
      {"stages": ["a","b","c"], "min_gaps": [0,1], "max_gaps": [3,5]}
    兼容旧格式 (两段):
      {"pre": "a", "trigger": "b", "max_gap_days": 3}
    """
    if not isinstance(raw, dict):
        raise ValueError("sequential strategy META['sequential'] must be a dict")

    stages_raw = raw.get("stages")
    if stages_raw is None:
        # 兼容旧两段格式
        pre = raw.get("pre") or raw.get("pre_strategy_id")
        trigger = raw.get("trigger") or raw.get("trigger_strategy_id")
        if not isinstance(pre, str) or not pre:
            raise ValueError(
                "sequential META['sequential']['stages'] or ['pre'] must be provided"
            )
        if not isinstance(trigger, str) or not trigger:
            raise ValueError(
                "sequential META['sequential']['trigger'] must be a non-empty strategy id"
            )
        stages = [pre, trigger]
        max_gap = _coerce_int(raw.get("max_gap_days", 3), "max_gap_days")
        min_gaps = [0]
        max_gaps = [max_gap]
    else:
        if not isinstance(stages_raw, list):
            raise ValueError("sequential META['sequential']['stages'] must be a list")
        stages = [str(item) for item in stages_raw if str(item or "").strip()]
        min_gaps = [_coerce_int(v, "min_gaps") for v in (raw.get("min_gaps") or [])]
        max_gaps = [_coerce_int(v, "max_gaps") for v in (raw.get("max_gaps") or [])]
        # 未显式给窗口时: 默认 min=0, max=3
        if not min_gaps:
            min_gaps = [0] * (len(stages) - 1)
        if not max_gaps:
            max_gaps = [3] * (len(stages) - 1)

    if not (2 <= len(stages) <= MAX_STAGES):
        raise ValueError(
            f"sequential stages count must be within [2, {MAX_STAGES}], got {len(stages)}"
        )
    if len(set(stages)) != len(stages):
        raise ValueError("sequential stages must be distinct strategies")
    if len(min_gaps) != len(stages) - 1 or len(max_gaps) != len(stages) - 1:
        raise ValueError(
            "sequential min_gaps/max_gaps length must equal len(stages)-1"
        )
    for i, (lo, hi) in enumerate(zip(min_gaps, max_gaps)):
        if lo < 0 or hi < 0:
            raise ValueError(f"sequential gap #{i + 1} must be >= 0")
        if lo > hi:
            raise ValueError(f"sequential gap #{i + 1}: min must be <= max")
        if hi > MAX_GAP_DAYS:
            raise ValueError(
                f"sequential gap #{i + 1}: max must be <= {MAX_GAP_DAYS}"
            )
    return SequentialSpec(
        stages=tuple(stages), min_gaps=tuple(min_gaps), max_gaps=tuple(max_gaps)
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
    hits_by_stage: list[dict[str, set[str]]],
    min_gaps: list[int] | tuple[int, ...],
    max_gaps: list[int] | tuple[int, ...],
) -> dict[str, dict[str, list[int]]]:
    """按交易日推进链式状态机。

    返回 {最后一段命中日: {symbol: [每段实际间隔]}}。
    dates 必须升序; 同一天多段命中时按顺序推进 (允许 min_gap=0 的同日链)。
    """
    n_stages = len(hits_by_stage)
    if n_stages < 2:
        return {}
    # stage_hits[k][symbol] = [(day_index, gaps_tuple), ...] (按时间升序)
    stage_hits: list[dict[str, list[tuple[int, tuple[int, ...]]]]] = [
        {} for _ in range(n_stages)
    ]

    for idx, day in enumerate(dates):
        # 第一段: 直接记录
        for symbol in hits_by_stage[0].get(day, ()):  # type: ignore[union-attr]
            stage_hits[0].setdefault(symbol, []).append((idx, ()))

        # 后续段: 逐段推进 (允许同日连续, 因此按 stage 递增顺序处理)
        for k in range(1, n_stages):
            lo = int(min_gaps[k - 1])
            hi = int(max_gaps[k - 1])
            prev = stage_hits[k - 1]
            for symbol in hits_by_stage[k].get(day, ()):  # type: ignore[union-attr]
                candidates = prev.get(symbol)
                if not candidates:
                    continue
                hit = None
                # 从最近往前找, 优先最近一次满足窗口的链
                for prev_idx, gaps in reversed(candidates):
                    delta = idx - prev_idx
                    if delta < lo:
                        continue
                    if delta > hi:
                        break  # 更早的只会更大
                    hit = (prev_idx, gaps + (delta,))
                    break
                if hit is not None:
                    stage_hits[k].setdefault(symbol, []).append(
                        (idx, hit[1])
                    )

        # 裁剪: 只为后续段保留窗口内可能用到的记录
        for k in range(n_stages - 1):
            hi = int(max_gaps[k])
            for symbol, items in list(stage_hits[k].items()):
                kept = [item for item in items if idx - item[0] <= hi]
                if kept:
                    stage_hits[k][symbol] = kept
                else:
                    stage_hits[k].pop(symbol, None)

    last = n_stages - 1
    events: dict[str, dict[str, list[int]]] = {}
    for symbol, items in stage_hits[last].items():
        for day_idx, gaps in items:
            events.setdefault(dates[day_idx], {})[symbol] = list(gaps)
    return events


def _window_any(matrix: np.ndarray, min_gap: int, max_gap: int) -> np.ndarray:
    """recent[t] = 是否存在 d∈[min_gap, max_gap] 使 matrix[t-d] 为真。"""
    n_times = matrix.shape[0]
    recent = np.zeros_like(matrix, dtype=bool)
    if max_gap < 0 or min_gap > max_gap:
        return recent
    cumulative = np.cumsum(matrix.astype(np.int32), axis=0)
    idx = np.arange(n_times)
    end = idx - min_gap  # 窗口右端 (含)
    start = idx - max_gap  # 窗口左端 (含)
    valid = end >= 0
    if not valid.any():
        return recent
    end_c = np.clip(end, 0, n_times - 1)
    totals = cumulative[end_c]
    # start-1 < 0 时前缀和为 0; 用列向量掩码避免 (n,) 与 (n,assets) 广播成 (n,n)
    before = cumulative[np.clip(start - 1, 0, n_times - 1)] * (start > 0)[:, None]
    window_sum = totals - before
    recent[valid] = window_sum[valid] > 0
    return recent


def merge_signal_matrices(
    shape: tuple[int, int],
    sigs: list,
    min_gaps: list[int] | tuple[int, ...],
    max_gaps: list[int] | tuple[int, ...],
    max_hold: int,
):
    """回测合并: 2~4 个 SignalMatrix → 顺序策略 SignalMatrix。

    entry[k] = stage_k.entry & any(entry[k-1] 在窗口内命中)
    exit     = 最后一段 exit & 合并 entry 的持仓窗口
    score    = 最后一段 score (仅保留入选点)
    """
    from app.backtest.matrix import make_signal_matrix
    from app.strategy.composite import _hold_masks_from_entries

    n_stages = len(sigs)
    n_times, _ = shape
    if n_stages < 2 or len(min_gaps) != n_stages - 1 or len(max_gaps) != n_stages - 1:
        return make_signal_matrix(shape)

    valid = np.asarray(sigs[0].entry, dtype=bool)
    for k in range(1, n_stages):
        recent = _window_any(valid, int(min_gaps[k - 1]), int(max_gaps[k - 1]))
        valid = np.asarray(sigs[k].entry, dtype=bool) & recent

    last = sigs[-1]
    hold_mask = _hold_masks_from_entries([valid], max_hold)[0]
    merged_exit = np.asarray(last.exit, dtype=bool) & hold_mask
    last_score = np.asarray(last.score, dtype=np.float64)
    merged_score = np.where(valid, last_score, 0.0)
    merged_score = np.nan_to_num(merged_score, nan=0.0, posinf=0.0, neginf=0.0)

    entry_codes = np.where(valid, 1, -1).astype(np.int16)
    exit_codes = np.full(shape, -1, dtype=np.int16)
    signal_ids = tuple(f"sequential:stage{i + 1}" for i in range(n_stages))

    return make_signal_matrix(
        shape,
        entry=valid.astype(np.uint8),
        exit=merged_exit.astype(np.uint8),
        score=merged_score.astype(np.float32),
        entry_signal_code=entry_codes,
        exit_signal_code=exit_codes,
        entry_signal_ids=signal_ids,
    )
