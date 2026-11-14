"""顺序策略 (sequential): 声明解析、链式状态机、回测矩阵合并。"""
from __future__ import annotations

import numpy as np
import pytest

from app.backtest.matrix import make_signal_matrix
from app.strategy import sequential as seq


# ---------------------------------------------------------------- 声明解析
def test_parse_spec_legacy_two_stage():
    spec = seq.parse_spec({"pre": "a", "trigger": "b", "max_gap_days": 3})
    assert spec.stages == ("a", "b")
    assert spec.min_gaps == (0,)
    assert spec.max_gaps == (3,)


def test_parse_spec_multi_stage():
    spec = seq.parse_spec(
        {
            "stages": ["a", "b", "c"],
            "min_gaps": [0, 1],
            "max_gaps": [3, 5],
        }
    )
    assert spec.stages == ("a", "b", "c")
    assert spec.min_gaps == (0, 1)
    assert spec.max_gaps == (3, 5)
    assert spec.transitions == 2


def test_parse_spec_rejects_invalid():
    with pytest.raises(ValueError):
        seq.parse_spec(None)
    with pytest.raises(ValueError):
        seq.parse_spec({"stages": ["a"]})  # 段数不足
    with pytest.raises(ValueError):
        seq.parse_spec({"stages": ["a", "b", "c", "d", "e"]})  # 超过 4 段
    with pytest.raises(ValueError):
        seq.parse_spec({"stages": ["a", "a"], "max_gaps": [3]})  # 重复
    with pytest.raises(ValueError):
        seq.parse_spec(
            {"stages": ["a", "b"], "min_gaps": [5], "max_gaps": [3]}
        )  # min > max


# ---------------------------------------------------------------- 状态机
def test_match_events_two_stage_window():
    dates = ["d0", "d1", "d2", "d3"]
    hits = [
        {"d0": {"A"}, "d2": {"B"}},
        {"d0": {"A"}, "d1": {"A"}, "d2": {"B"}, "d3": {"A", "B"}},
    ]
    events = seq.match_events(dates, hits, [0], [1])

    assert events["d0"] == {"A": [0]}
    assert events["d1"] == {"A": [1]}
    assert events["d2"] == {"B": [0]}
    assert events["d3"] == {"B": [1]}


def test_match_events_min_gap():
    dates = ["d0", "d1", "d2"]
    hits = [{"d0": {"A"}}, {"d1": {"A"}, "d2": {"A"}}]
    # 最小间隔 2: d1 太近, d2 恰好 2
    events = seq.match_events(dates, hits, [2], [5])
    assert events == {"d2": {"A": [2]}}


def test_match_events_three_stage_chain():
    dates = ["d0", "d1", "d2", "d3"]
    hits = [
        {"d0": {"X"}},
        {"d1": {"X"}},
        {"d3": {"X"}},
    ]
    # X 在 d0 → d1 (gap 1) → d3 (gap 2), 两段窗口 [1,2] 与 [1,2]
    events = seq.match_events(dates, hits, [1, 1], [2, 2])
    assert events == {"d3": {"X": [1, 2]}}


def test_match_events_rejects_broken_chain():
    dates = ["d0", "d1", "d2", "d3"]
    hits = [{"d0": {"X"}}, {"d3": {"X"}}]
    events = seq.match_events(dates, hits, [0], [1])
    assert events == {}


# ---------------------------------------------------------------- 回测合并
def _matrix(shape, entry_idx=(), score=None):
    entry = np.zeros(shape, dtype=np.uint8)
    for t, a in entry_idx:
        entry[t, a] = 1
    if score is None:
        score = np.zeros(shape, dtype=np.float32)
    return make_signal_matrix(shape, entry=entry, score=np.asarray(score, dtype=np.float32))


def test_merge_signal_matrices_two_stage():
    shape = (6, 1)
    s1 = _matrix(shape, [(0, 0)])
    s2 = _matrix(shape, [(2, 0)], score=np.full(shape, 88.0, dtype=np.float32))

    merged = seq.merge_signal_matrices(shape, [s1, s2], [0], [2], 2)
    assert merged.entry[2, 0] == 1
    assert merged.entry.sum() == 1
    assert float(merged.score[2, 0]) == pytest.approx(88.0)


def test_merge_signal_matrices_min_gap():
    shape = (6, 1)
    s1 = _matrix(shape, [(0, 0)])
    s2 = _matrix(shape, [(1, 0), (3, 0)])

    merged = seq.merge_signal_matrices(shape, [s1, s2], [2], [3], 2)
    # t=1 太近, t=3 间隔 3 有效
    assert merged.entry[1, 0] == 0
    assert merged.entry[3, 0] == 1


def test_merge_signal_matrices_three_stage():
    shape = (8, 1)
    s1 = _matrix(shape, [(0, 0)])
    s2 = _matrix(shape, [(1, 0)])
    s3 = _matrix(shape, [(3, 0)], score=np.full(shape, 70.0, dtype=np.float32))

    merged = seq.merge_signal_matrices(shape, [s1, s2, s3], [1, 1], [2, 2], 2)
    assert merged.entry[3, 0] == 1
    assert merged.entry.sum() == 1


def test_merge_signal_matrices_three_stage_breaks_middle():
    shape = (8, 1)
    s1 = _matrix(shape, [(0, 0)])
    s2 = _matrix(shape, [(4, 0)])  # 距 S1 太远
    s3 = _matrix(shape, [(5, 0)])

    merged = seq.merge_signal_matrices(shape, [s1, s2, s3], [0, 0], [2, 2], 2)
    assert merged.entry.sum() == 0


def test_merge_signal_matrices_exit_projected_on_last_stage():
    shape = (6, 1)
    s1 = _matrix(shape, [(0, 0)])
    s2 = make_signal_matrix(
        shape,
        entry=np.array([[0], [0], [1], [0], [0], [0]], dtype=np.uint8),
        exit=np.array([[0], [0], [0], [1], [1], [0]], dtype=np.uint8),
    )

    merged = seq.merge_signal_matrices(shape, [s1, s2], [0], [2], 2)
    assert merged.entry[2, 0] == 1
    assert merged.exit[3, 0] == 1
    assert merged.exit[4, 0] == 0
