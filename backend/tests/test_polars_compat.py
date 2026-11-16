"""polars 兼容层: 历史策略里的 pl.arctan(...) 仍可用。"""
from __future__ import annotations

import polars as pl

from app.strategy import polars_compat


def test_install_restores_module_level_trig_aliases():
    polars_compat.install()
    for name in ("arctan", "arcsin", "arccos"):
        assert hasattr(pl, name), f"pl.{name} 未补回"


def test_module_level_arctan_returns_expr():
    polars_compat.install()
    expr = pl.arctan(pl.col("x"))
    df = pl.DataFrame({"x": [0.0, 1.0]}).with_columns(expr.alias("a"))
    assert df["a"][0] == 0.0
    assert abs(df["a"][1] - 0.7853981) < 1e-5


def test_expr_method_still_works():
    df = pl.DataFrame({"x": [1.0]}).with_columns(pl.col("x").arctan().alias("a"))
    assert abs(df["a"][0] - 0.7853981) < 1e-5
