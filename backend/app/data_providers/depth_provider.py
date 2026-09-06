"""五档盘口 provider 接口与 TickFlow 默认实现。

DepthProvider 契约 (depth_service._call_depth_batch 消费端):

    batch(symbols: list[str]) -> dict[str, MarketDepth]

    MarketDepth = {
        "ask_volumes": list,  # 卖档量(手), [0]=卖一
        "bid_volumes": list,  # 买档量(手), [0]=买一
        "timestamp": int,     # 毫秒 epoch
    }

TickFlowDepthProvider 是从 depth_service._call_depth_batch 迁移出的默认实现
(resolve_limit 切片 + rpm 节流 + tf.depth.batch), TickFlow 路径行为零回归。
第三方网关 (如 plugins/eltdx) 在各自插件内实现同名 batch(), 由 depth_service
按 depth5_data_provider 偏好路由; 第三方失败时回退本实现。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DepthProvider(Protocol):
    def batch(self, symbols: list[str]) -> dict[str, dict]:
        """批量取五档盘口, 返回 {symbol: MarketDepth}。"""
        ...


class TickFlowDepthProvider:
    """TickFlow SDK 默认实现: capset 批量上限切片 + rpm 节流 + tf.depth.batch。"""

    def __init__(self, capset_provider: Callable[[], Any] | None = None) -> None:
        self._capset_provider = capset_provider

    def batch(self, symbols: list[str]) -> dict[str, dict]:
        from app.tickflow.capabilities import Cap
        from app.tickflow.client import get_client
        from app.tickflow.rate_limits import chunked, resolve_limit, sleep_between_batches

        tf = get_client()
        capset = self._capset_provider() if self._capset_provider else None
        limit = resolve_limit(capset, Cap.DEPTH5_BATCH,
                              default_batch=100, default_rpm=30)

        result: dict[str, dict] = {}
        chunks = chunked(list(symbols or []), limit.batch)
        for i, chunk in enumerate(chunks):
            sleep_between_batches(i, limit.rpm, default_interval=2.0)
            try:
                # SDK 的 batch 内部已按 batch_size 切, 这里再切一层防单请求过大
                data = tf.depth.batch(chunk)
                if isinstance(data, dict):
                    result.update(data)
            except Exception as e:  # noqa: BLE001
                logger.warning("depth.batch 第 %d 批失败(%d 只): %s", i + 1, len(chunk), e)
                # 单批失败不影响其他批
        return result
