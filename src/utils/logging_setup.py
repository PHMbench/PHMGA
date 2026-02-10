from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


_CURRENT_LOGGER: contextvars.ContextVar["LoggerBundle | None"] = contextvars.ContextVar(
    "PHMGA_CURRENT_LOGGER", default=None
)

_SECRET_KEY_PATTERNS = (
    "api_key",
    "apikey",
    "authorization",
    "token",
    "secret",
    "password",
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+")
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9._\-]{32,}\b")


def _is_true(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _level_from_env() -> int:
    level_name = (os.getenv("PHM_LOG_LEVEL") or "INFO").upper().strip()
    return getattr(logging, level_name, logging.INFO)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mask_string(value: str) -> str:
    masked = _BEARER_RE.sub("Bearer ***", value)
    return _LONG_TOKEN_RE.sub("***", masked)


def _sanitize(payload: Any) -> Any:
    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for key, value in payload.items():
            lower_key = str(key).lower()
            if any(pat in lower_key for pat in _SECRET_KEY_PATTERNS):
                out[str(key)] = "***"
            else:
                out[str(key)] = _sanitize(value)
        return out
    if isinstance(payload, list):
        return [_sanitize(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_sanitize(v) for v in payload)
    if isinstance(payload, str):
        return _mask_string(payload)
    return payload


def _summarize_for_console(event: Dict[str, Any]) -> str:
    parts = [
        event.get("event", "event"),
        f"phase={event.get('phase', '-')}",
        f"node={event.get('node', '-')}",
    ]
    elapsed = event.get("elapsed_ms")
    if elapsed is not None:
        parts.append(f"elapsed_ms={elapsed}")
    msg = str(event.get("message", "") or "").strip()
    if msg:
        parts.append(f"msg={msg}")
    return " ".join(parts)


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        event = getattr(record, "event_data", None)
        if isinstance(event, dict):
            return f"{event.get('ts')} {event.get('level')} {_summarize_for_console(event)}"
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        event = getattr(record, "event_data", None)
        if isinstance(event, dict):
            return json.dumps(event, ensure_ascii=False)
        return json.dumps({"ts": _utc_now_iso(), "level": record.levelname, "message": record.getMessage()})


@dataclass(frozen=True)
class LoggerBundle:
    logger: logging.Logger
    run_id: str
    case_name: str
    log_dir: Path
    full_llm: bool


def init_run_logger(case_name: str, save_dir: str | Path, run_id: str) -> LoggerBundle:
    save_root = Path(str(os.getenv("PHM_LOG_DIR") or save_dir)).expanduser().resolve()
    log_dir = save_root / case_name / run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"phmga.{case_name}.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(_level_from_env())

    if _is_true(os.getenv("PHM_LOG_CONSOLE"), default=True):
        ch = logging.StreamHandler()
        ch.setLevel(_level_from_env())
        ch.setFormatter(_ConsoleFormatter())
        logger.addHandler(ch)

    run_log = log_dir / "run.log"
    fh = logging.FileHandler(run_log, encoding="utf-8")
    fh.setLevel(_level_from_env())
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(fh)

    if _is_true(os.getenv("PHM_LOG_JSON"), default=True):
        jh = logging.FileHandler(log_dir / "events.jsonl", encoding="utf-8")
        jh.setLevel(_level_from_env())
        jh.setFormatter(_JsonFormatter())
        logger.addHandler(jh)

    full_llm = _is_true(os.getenv("PHM_LOG_FULL_LLM"), default=True)
    bundle = LoggerBundle(
        logger=logger,
        run_id=run_id,
        case_name=case_name,
        log_dir=log_dir,
        full_llm=full_llm,
    )
    return bundle


def set_current_logger(bundle: LoggerBundle) -> None:
    _CURRENT_LOGGER.set(bundle)


def clear_current_logger() -> None:
    _CURRENT_LOGGER.set(None)


def get_current_logger() -> LoggerBundle | None:
    return _CURRENT_LOGGER.get()


def _normalize_payload(payload: Any, *, full_llm: bool) -> Any:
    if payload is None:
        return {}
    clean = _sanitize(payload)
    if full_llm:
        return clean
    if isinstance(clean, dict):
        out = dict(clean)
        for key in ("prompt", "response"):
            if key in out and isinstance(out[key], str):
                text = out[key]
                out[key] = text[:512] + (" ...<truncated>" if len(text) > 512 else "")
        return out
    return clean


def _to_exc_payload(exc: BaseException) -> Dict[str, Any]:
    return {
        "exc_type": type(exc).__name__,
        "exc_msg": str(exc),
        "traceback": traceback.format_exc(),
    }


def log_event(
    bundle: LoggerBundle | None = None,
    *,
    level: str = "INFO",
    event: str,
    phase: str | None = None,
    node: str | None = None,
    message: str = "",
    payload: Any = None,
    elapsed_ms: int | None = None,
    **fields: Any,
) -> None:
    b = bundle or get_current_logger()
    if b is None:
        return

    event_data: Dict[str, Any] = {
        "ts": _utc_now_iso(),
        "level": level.upper(),
        "run_id": b.run_id,
        "case_name": b.case_name,
        "phase": phase or "",
        "node": node or "",
        "event": event,
        "message": message,
        "payload": _normalize_payload(payload, full_llm=b.full_llm),
    }
    if elapsed_ms is not None:
        event_data["elapsed_ms"] = int(elapsed_ms)
    if fields:
        event_data.update(_sanitize(fields))

    msg = _summarize_for_console(event_data)
    level_no = getattr(logging, level.upper(), logging.INFO)
    b.logger.log(level_no, msg, extra={"event_data": event_data})


@contextmanager
def timed(
    bundle: LoggerBundle | None = None,
    *,
    event: str,
    phase: str | None = None,
    node: str | None = None,
    message: str = "",
    payload: Any = None,
    **fields: Any,
) -> Iterator[None]:
    b = bundle or get_current_logger()
    start = time.perf_counter()
    log_event(
        b,
        level="INFO",
        event=f"{event}.start",
        phase=phase,
        node=node,
        message=message,
        payload=payload,
        **fields,
    )
    try:
        yield
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        log_event(
            b,
            level="ERROR",
            event=f"{event}.fail",
            phase=phase,
            node=node,
            message=str(exc),
            payload=_to_exc_payload(exc),
            elapsed_ms=elapsed_ms,
            **fields,
        )
        raise
    else:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        log_event(
            b,
            level="INFO",
            event=f"{event}.success",
            phase=phase,
            node=node,
            message=message,
            payload=payload,
            elapsed_ms=elapsed_ms,
            **fields,
        )

