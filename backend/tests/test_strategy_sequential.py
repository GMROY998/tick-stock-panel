"""顺序策略 (sequential): 声明解析、交易日状态机、回测矩阵合并。"""
from __future__ import annotations

import numpy as np
import pytest

from app.backtest.matrix import make_signal_matrix
from app.strategy import sequential as seq


# ---------------------------------------------------------------- 声明解析
def test_parse_spec_accepts_short_and_long_fields():
    spec = seq.parse_spec({"pre": "a", "trigger": "b", "max_gap_days": 3})
    assert (spec.pre_strategy_id, spec.trigger_strategy_id, spec.max_gap_days) == ("a", "b", 3)

    spec2 = seq.parse_spec(
        {"pre_strategy_id": "a", "trigger_strategy_id": "b", "max_gap_days": 0}
    )
    assert spec2.max_gap_days == 0


def test_parse_spec_rejects_invalid():
    with pytest.raises(ValueError):
        seq.parse_spec(None)
    with pytest.raises(ValueError):
        seq.parse_spec({"pre": "a", "trigger": "a"})
    with pytest.raises(ValueError):
        seq.parse_spec({"pre": "a", "trigger": "b", "max_gap_days": -1})
    with pytest.raises(ValueError):
        seq.parse_spec({"pre": "a", "trigger": "b", "max_gap_days": seq.MAX_GAP_DAYS + 1})


# ---------------------------------------------------------------- 状态机
def test_match_events_same_day_and_gap_window():
    dates = ["d0", "d1", "d2", "d3"]
    pre = {"d0": {"A"}, "d2": {"B"}}
    trigger = {"d0": {"A"}, "d1": {"A"}, "d2": {"B"}, "d3": {"A", "B"}}

    events = seq.match_events(dates, pre, trigger, max_gap_days=1)

    # A 同日命中 (gap=0)
    assert events["d0"] == {"A": 0}
    # A 在 d0 命中, d1 触发 (gap=1)
    assert events["d1"] == {"A": 1}
    # d2: B 的前置也在 d2 命中 → 同日命中 (gap=0)
    assert events["d2"] == {"B": 0}
    # d3: B 最近一次前置是 d2 (gap=1) → 入选; A 的前置是 d0 (gap=3) → 超窗
    assert events["d3"] == {"B": 1}


def test_match_events_respects_zero_window():
    dates = ["d0", "d1"]
    events = seq.match_events(
        dates, {"d0": {"A"}}, {"d1": {"A"}}, max_gap_days=0
    )
    assert events == {}


def test_match_events_uses_latest_pre_hit():
    dates = ["d0", "d1", "d2", "d3"]
    events = seq.match_events(
        dates,
        {"d0": {"A"}, "d2": {"A"}},
        {"d3": {"A"}},
        max_gap_days=2,
    )
    # 最近一次 A 在 d2, gap=1 (而不是 d0 的 3 超窗)
    assert events["d3"] == {"A": 1}


# ---------------------------------------------------------------- 回测合并
def _matrix(shape, entry_idx=(), score=None):
    entry = np.zeros(shape, dtype=np.uint8)
    for t, a in entry_idx:
        entry[t, a] = 1
    if score is None:
        score = np.zeros(shape, dtype=np.float32)
    return make_signal_matrix(shape, entry=entry, score=np.asarray(score, dtype=np.float32))


def test_merge_signal_matrices_entry_semantics():
    shape = (6, 1)
    pre = _matrix(shape, [(0, 0)])
    trigger = _matrix(shape, [(2, 0)], score=np.full(shape, 88.0, dtype=np.float32))

    merged = seq.merge_signal_matrices(shape, [pre, trigger], max_gap_days=2, max_hold=2)
    assert merged.entry[2, 0] == 1
    assert merged.entry.sum() == 1
    assert float(merged.score[2, 0]) == pytest.approx(88.0)


def test_merge_signal_matrices_window_too_short():
    shape = (6, 1)
    pre = _matrix(shape, [(0, 0)])
    trigger = _matrix(shape, [(2, 0)])

    merged = seq.merge_signal_matrices(shape, [pre, trigger], max_gap_days=1, max_hold=2)
    assert merged.entry.sum() == 0


def test_merge_signal_matrices_same_day():
    shape = (4, 1)
    pre = _matrix(shape, [(1, 0)])
    trigger = _matrix(shape, [(1, 0)])

    merged = seq.merge_signal_matrices(shape, [pre, trigger], max_gap_days=0, max_hold=2)
    assert merged.entry[1, 0] == 1
    assert merged.entry.sum() == 1


def test_merge_signal_matrices_exit_is_projected_on_trigger():
    shape = (6, 1)
    pre = _matrix(shape, [(0, 0)])
    trigger = make_signal_matrix(
        shape,
        entry=np.array([[0], [0], [1], [0], [0], [0]], dtype=np.uint8),
        exit=np.array([[0], [0], [0], [1], [1], [0]], dtype=np.uint8),
    )

    merged = seq.merge_signal_matrices(shape, [pre, trigger], max_gap_days=2, max_hold=2)
    assert merged.entry[2, 0] == 1
    # 持仓窗口内的 trigger 卖点保留
    assert merged.exit[3, 0] == 1
    # 窗口外的卖点不生效 (不会延后平仓)
    assert merged.exit[4, 0] == 0
