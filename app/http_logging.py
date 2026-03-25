from __future__ import annotations

import logging
import os
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

ACCESS_LOG = logging.getLogger("ml_api.access")


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if not logging.root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    for name in ("ml_api", "ml_api.access", "ml_api.api", "ml_api.ml_core"):
        logging.getLogger(name).setLevel(level)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status, duration (ms), and correlate with X-Request-ID."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            ACCESS_LOG.exception(
                "request_error method=%s path=%s duration_ms=%.3f request_id=%s",
                request.method,
                request.url.path,
                elapsed_ms,
                rid,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        status = response.status_code
        if status >= 500:
            ACCESS_LOG.error(
                "request method=%s path=%s status=%s duration_ms=%.3f request_id=%s",
                request.method,
                request.url.path,
                status,
                elapsed_ms,
                rid,
            )
        elif status >= 400:
            ACCESS_LOG.warning(
                "request method=%s path=%s status=%s duration_ms=%.3f request_id=%s",
                request.method,
                request.url.path,
                status,
                elapsed_ms,
                rid,
            )
        else:
            ACCESS_LOG.info(
                "request method=%s path=%s status=%s duration_ms=%.3f request_id=%s",
                request.method,
                request.url.path,
                status,
                elapsed_ms,
                rid,
            )
        response.headers["X-Request-ID"] = rid
        return response
