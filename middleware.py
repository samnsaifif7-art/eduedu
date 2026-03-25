"""
Request logging middleware + global error handlers for EduMind AI.
"""
import time
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("edumind")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and elapsed time."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "%s %s → %s  (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
            )
            return response
        except Exception as exc:  # pragma: no cover
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "%s %s → 500 UNHANDLED: %s  (%.1f ms)",
                request.method,
                request.url.path,
                exc,
                elapsed,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "error": str(exc)},
            )


def add_error_handlers(app):
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"Route '{request.url.path}' not found.",
                "hint": "Check /docs for available endpoints.",
            },
        )

    @app.exception_handler(422)
    async def validation_error_handler(request: Request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error — check your request body.",
                "errors": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Unexpected server error.", "error": str(exc)},
        )
