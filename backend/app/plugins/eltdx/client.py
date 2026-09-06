"""eltdx-gateway 行情网关 HTTP 客户端 (RPC 风格)。

网关为独立容器 (eltdx[http] 官方 FastAPI), 只有 3 个 HTTP 端点:
- GET /health   健康检查
- GET /methods  可用方法清单
- POST /rpc     统一 RPC 入口: {"method": "<模块>.<方法>", "params": {...}}

响应统一为 {"id":..., "ok": true, "result": ...}; ok=false 视为业务失败
(非交易日/参数不符等), 抛 EltdxError 不重试; 连接错误/5xx 短退避重试。

代码格式: 系统内部 "600519.SH" ↔ 网关 "sh600519" (小写交易所前缀)。
北交所 (4/8 开头) 不入网关链路, to_gw_code 返回 None 由调用方跳过
(与 wencai-monitor _full_code 口径一致)。

地址: 环境变量 ELTDX_GATEWAY_URL 优先, 默认小主机局域网地址。
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://192.168.1.54:8600"
ENV_GATEWAY_URL = "ELTDX_GATEWAY_URL"

_HTTP_TIMEOUT = 10.0
_HEALTH_TIMEOUT = 1.5
_MAX_RETRIES = 2
_MIN_INTERVAL = 0.05  # 局域网网关轻节流(~20 QPS), 防打满单核限额容器

_SUFFIX_RE = re.compile(r"^(\d{6})\.(sh|sz)$")


class EltdxError(Exception):
    """eltdx-gateway 请求失败 (网络/业务错误)。"""


def resolve_base_url() -> str:
    """解析网关地址: ELTDX_GATEWAY_URL > 默认局域网地址。"""
    return (os.getenv(ENV_GATEWAY_URL, "") or DEFAULT_BASE_URL).strip().rstrip("/")


def to_gw_code(symbol: str) -> str | None:
    """系统 symbol → 网关完整代码。600519.SH → sh600519。

    纯 6 位代码按位推断: 6/9 开头 → sh, 0/3 开头 → sz;
    北交所 (4/8 开头) 与其他形态返回 None (不支持, 调用方跳过)。
    """
    s = str(symbol or "").strip().lower()
    m = _SUFFIX_RE.match(s)
    if m:
        return m.group(2) + m.group(1)
    if len(s) == 8 and s[:2] in ("sh", "sz") and s[2:].isdigit():
        return s
    if len(s) == 6 and s.isdigit():
        if s[0] in "69":
            return "sh" + s
        if s[0] in "03":
            return "sz" + s
    return None


def to_symbol(gw_code: str) -> str:
    """网关完整代码 → 系统 symbol。sh600519 → 600519.SH。"""
    s = str(gw_code or "").strip().lower()
    if len(s) == 8 and s[:2] in ("sh", "sz") and s[2:].isdigit():
        return f"{s[2:]}.{s[:2].upper()}"
    return str(gw_code)


class EltdxClient:
    """带轻节流与短重试的 eltdx-gateway RPC 客户端。"""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = _HTTP_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
        min_interval: float = _MIN_INTERVAL,
    ) -> None:
        self.base_url = (base_url or resolve_base_url()).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._http = httpx.Client(timeout=timeout)
        self._throttle_lock = threading.Lock()
        self._last_request_at = 0.0

    def rpc(self, method: str, params: dict[str, Any] | None = None,
            timeout: float | None = None) -> Any:
        """执行一次 RPC, 返回 result; ok=false 抛 EltdxError (不重试)。"""
        payload = {"method": method, "params": params or {}}
        errors: list[str] = []
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._http.post(f"{self.base_url}/rpc", json=payload,
                                       timeout=timeout or self.timeout)
            except httpx.HTTPError as exc:  # 连接失败/超时 → 退避重试
                errors.append(str(exc))
                time.sleep(min(2 ** attempt * 0.2, 1.0))
                continue
            if resp.status_code >= 500:
                errors.append(f"http {resp.status_code}")
                time.sleep(min(2 ** attempt * 0.2, 1.0))
                continue
            try:
                body = resp.json()
            except ValueError:
                errors.append(f"invalid json {resp.text[:120]!r}")
                break
            if isinstance(body, dict) and body.get("ok"):
                return body.get("result")
            # ok=false 是确定性业务结果, 重试无意义
            raise EltdxError(f"{method}: {str(body)[:200]}")
        raise EltdxError(f"{method}: " + "; ".join(errors or ["request failed"]))

    def health(self) -> tuple[bool, str]:
        """探测 /health (短超时), 返回 (是否可达, 说明)。"""
        try:
            resp = self._http.get(f"{self.base_url}/health", timeout=_HEALTH_TIMEOUT)
            resp.raise_for_status()
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            return False, f"网关不可达({self.base_url}): {e}"

    def _throttle(self) -> None:
        with self._throttle_lock:
            now = time.monotonic()
            wait = self._last_request_at + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = max(now, self._last_request_at + self.min_interval)

    def close(self) -> None:
        self._http.close()
