"""猫爪 MeoZ HTTP 客户端。

请求/响应协议(实测 2026-09):
- POST <endpoint>  JSON 载荷 {"apiname": str, "fields": str(逗号分隔), "params": dict, "apikey": str}
- 成功返回 JSON {"code": 200, "message": "...", "data": {"fields": [...], "items": [...]}}
  items 每一行与 fields 一一对应(部分接口行元素数多于 fields, 只取前 len(fields) 个)。
- 空数据也是 code==200 且 items 为空; 个别接口对“休市/非交易日”返回非 200 code + 提示 message, 视为空。
- 限频: 约 3 QPS。HTTP/业务 429 按 Retry-After 退避重试; 连接错误/5xx 指数退避重试。

线路: 公开 https://numcat.net/api 开箱可用; 机房专线(sz/sh.numcat.net:8866)仅用于部署在
猫爪机房附近的场景, 由环境变量 MEOZ_API_ROUTE=dedicated 启用。
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PUBLIC_ENDPOINT = "https://numcat.net/api"
DEDICATED_ENDPOINTS = ("http://sz.numcat.net:8866/api", "http://sh.numcat.net:8866/api")

# 业务层“无数据”提示: 这些接口(如按交易日查询)对休市/非交易日返回非 200 code。
_NO_DATA_KEYWORDS = ("非交易日", "休市", "无数据", "没有数据", "暂无", "数据为空", "没有行情")

_HTTP_TIMEOUT = 30.0
_MIN_INTERVAL = 0.35  # 约 3 QPS 下限间隔
_MAX_RETRIES = 3


class MeoZError(Exception):
    """MeoZ 请求失败(网络/业务错误、Key 无效、限频用尽等)。"""


def resolve_endpoints() -> list[str]:
    """解析可用请求端点。环境变量优先; 未配置 MEOZ_API_ROUTE 时用公开线路。"""
    route = os.getenv("MEOZ_API_ROUTE", "").strip().lower()
    if route == "dedicated":
        return list(DEDICATED_ENDPOINTS)
    return [PUBLIC_ENDPOINT]


def _no_data(message: str | None) -> bool:
    if not message:
        return False
    return any(kw in message for kw in _NO_DATA_KEYWORDS)


def _retry_after_seconds(resp: httpx.Response) -> float:
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return max(float(header), 0.5)
        except ValueError:
            pass
    text = resp.text[:400]
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s)?", text)
    if m and ("429" in text or "限频" in text or "too many" in text.lower()):
        return max(float(m.group(1)), 0.5)
    return 1.0


def _to_rows(data: Any) -> list[dict[str, Any]]:
    """把响应 data 统一成 [{field: value}, ...]。"""
    if data is None:
        return []
    if isinstance(data, list):
        return [dict(r) for r in data if isinstance(r, dict)]
    fields = data.get("fields") or []
    items = data.get("items") or []
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, (list, tuple)):
            continue
        rows.append({name: item[i] for i, name in enumerate(fields) if i < len(item)})
    return rows


class MeoZClient:
    """带全局限频与端点切换的 MeoZ JSON API 客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = _HTTP_TIMEOUT,
        min_interval: float = _MIN_INTERVAL,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        if not api_key:
            raise MeoZError("未配置猫爪 MeoZ API Key")
        self.api_key = api_key
        self.endpoints = [base_url] if base_url else resolve_endpoints()
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._http = httpx.Client(timeout=timeout)
        self._throttle_lock = threading.Lock()
        self._last_request_at = 0.0

    # ---- 对外查询接口 ----------------------------------------------------
    def rows(
        self,
        apiname: str,
        fields: list[str] | str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """执行一次查询, 返回行字典列表; 无数据返回 []。"""
        fields_str = ",".join(str(f) for f in fields) if isinstance(fields, (list, tuple)) else str(fields)
        payload = {
            "apiname": apiname,
            "fields": fields_str,
            "params": params or {},
            "apikey": self.api_key,
        }
        body = self._post_json(payload)
        if body is None:
            return []
        return _to_rows(body.get("data"))

    # ---- 内部: 带限频/重试/端点切换的 POST --------------------------------
    def _throttle(self) -> None:
        with self._throttle_lock:
            now = time.monotonic()
            wait = self._last_request_at + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = max(now, self._last_request_at + self.min_interval)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """返回响应 body dict; None 表示业务上“无数据”; 失败抛 MeoZError。"""
        errors: list[str] = []
        for endpoint in self.endpoints:
            for attempt in range(self.max_retries + 1):
                self._throttle()
                try:
                    resp = self._http.post(endpoint, json=payload)
                except httpx.HTTPError as exc:  # 连接失败/超时
                    errors.append(f"{endpoint}: {exc}")
                    time.sleep(min(2**attempt, 4.0))
                    continue

                # ---- HTTP 层频控/服务端错误: 退避后重试 ----
                if resp.status_code == 429:
                    time.sleep(_retry_after_seconds(resp))
                    continue
                if resp.status_code >= 500:
                    errors.append(f"{endpoint}: http {resp.status_code}")
                    time.sleep(min(2**attempt, 4.0))
                    continue

                # ---- 解析业务包 ----
                try:
                    body = resp.json()
                except ValueError:
                    errors.append(f"{endpoint}: invalid json {resp.text[:120]!r}")
                    continue
                if not isinstance(body, dict) or "code" not in body:
                    errors.append(f"{endpoint}: unexpected payload {str(body)[:120]!r}")
                    continue
                code = body.get("code")
                if code == 200:
                    return body
                message = body.get("message") or f"code {code}"
                if code == 429:
                    time.sleep(_retry_after_seconds(resp))
                    continue
                if _no_data(message):
                    return None
                # 4xx 业务错误(参数非法/Key 无效/权限不足): 不重试
                errors.append(f"{endpoint}: {message}")
                break  # 业务错误换端点也没用
        raise MeoZError("; ".join(errors) or "meoz request failed")

    def close(self) -> None:
        self._http.close()
