"""polars 版本兼容: 补回模块级数学函数别名。

polars 1.4x 起把部分数学函数收到 Expr 上 (``pl.Expr.arctan`` 等), 模块级的
``pl.arctan`` / ``pl.arcsin`` / ``pl.arccos`` 被移除。AI 生成的历史策略仍会写
``pl.arctan(expr)``, 加载即抛 ``AttributeError`` → 策略运行 500。

这里在应用启动时补回别名: ``pl.Expr.arctan`` 是未绑定方法, 以
``pl.arctan(expr)`` 调用与旧版签名一致, 不影响新写法 ``expr.arctan()``。
"""
from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

# 旧模块级别名 → Expr 上的方法名
_ALIASES = {
    "arctan": "arctan",
    "arcsin": "arcsin",
    "arccos": "arccos",
    "atan": "arctan",
    "asin": "arcsin",
    "acos": "arccos",
}


def install() -> list[str]:
    """补回缺失的模块级函数, 返回实际补上的名字。"""
    patched: list[str] = []
    for alias, expr_name in _ALIASES.items():
        if hasattr(pl, alias):
            continue
        func = getattr(pl.Expr, expr_name, None)
        if func is None:
            continue
        setattr(pl, alias, func)
        patched.append(alias)
    if patched:
        logger.info("polars 兼容: 补回模块级函数 %s", ",".join(patched))
    return patched
