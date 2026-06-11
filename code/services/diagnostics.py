from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable
import json


@dataclass(frozen=True)
class DiagnosticEvent:
    level: str
    source: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class DiagnosticsService:
    """Small structured diagnostics bus for runtime events."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[DiagnosticEvent], None]] = []
        self._events: list[DiagnosticEvent] = []
        self._lock = Lock()

    def subscribe(self, callback: Callable[[DiagnosticEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def events(self) -> list[DiagnosticEvent]:
        with self._lock:
            return list(self._events)

    def log(
        self,
        level: str,
        source: str,
        message: str,
        **context: Any,
    ) -> DiagnosticEvent:
        event = DiagnosticEvent(
            level=str(level).lower(),
            source=str(source),
            message=str(message),
            context=dict(context),
        )
        with self._lock:
            self._events.append(event)
            subscribers = list(self._subscribers)
        self._print_event(event)
        for callback in subscribers:
            try:
                callback(event)
            except Exception as exc:
                print(f"[DiagnosticsService] Subscriber {callback!r} raised: {exc}")
        return event

    @staticmethod
    def _print_event(event: DiagnosticEvent) -> None:
        context = (
            " " + json.dumps(event.context, sort_keys=True, default=str)
            if event.context
            else ""
        )
        print(
            f"[{event.timestamp_utc}] [{event.level}] [{event.source}] {event.message}{context}",
            flush=True,
        )

    def debug(self, source: str, message: str, **context: Any) -> DiagnosticEvent:
        return self.log("debug", source, message, **context)

    def info(self, source: str, message: str, **context: Any) -> DiagnosticEvent:
        return self.log("info", source, message, **context)

    def warning(self, source: str, message: str, **context: Any) -> DiagnosticEvent:
        return self.log("warning", source, message, **context)

    def error(self, source: str, message: str, **context: Any) -> DiagnosticEvent:
        return self.log("error", source, message, **context)

    def exception(
        self,
        source: str,
        message: str,
        exc: Exception,
        **context: Any,
    ) -> DiagnosticEvent:
        return self.log(
            "error",
            source,
            message,
            error_type=type(exc).__name__,
            error=str(exc),
            **context,
        )
