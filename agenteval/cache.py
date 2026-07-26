from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable
from typing import Any

from agenteval.exceptions import ConfigurationError
from agenteval.types import Task


def prediction_key(system_name: str, task: Task, version: str = "") -> str:
    payload = json.dumps(
        {"system": system_name, "task": task.id, "input": task.input, "version": version},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class PredictionCache:
    def __init__(self, path: str = "", version: str = ""):
        self.path = path
        self._version = version
        self._entries: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        if path and os.path.exists(path):
            self.load()

    def get(self, system_name: str, task: Task) -> Any:
        key = prediction_key(system_name, task, self._version)
        with self._lock:
            if key in self._entries:
                self._hits += 1
                return self._entries[key]
            self._misses += 1
            return None

    def has(self, system_name: str, task: Task) -> bool:
        return prediction_key(system_name, task, self._version) in self._entries

    def put(self, system_name: str, task: Task, prediction: Any) -> None:
        key = prediction_key(system_name, task, self._version)
        with self._lock:
            self._entries[key] = prediction

    def wrap(self, system: Callable[[Task], Any], system_name: str) -> Callable[[Task], Any]:
        def cached(task: Task) -> Any:
            hit = self.get(system_name, task)
            if hit is not None:
                return hit
            prediction = system(task)
            self.put(system_name, task, prediction)
            return prediction
        return cached

    def save(self, path: str = "") -> None:
        target = path or self.path
        if not target:
            raise ConfigurationError("no cache path configured; pass path= to PredictionCache or save(path=...)")

        directory = os.path.dirname(os.path.abspath(target)) or "."
        os.makedirs(directory, exist_ok=True)

        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": self._version, "entries": self._entries},
                          f, default=str)
            os.replace(tmp, target)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def load(self, path: str = "") -> bool:
        target = path or self.path
        if not target or not os.path.exists(target):
            return False
        try:
            with open(target, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            return False

        if payload.get("version", "") != self._version:
            self._entries = {}
            return False

        self._entries = payload.get("entries", {})
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }

    @property
    def size(self) -> int:
        return len(self._entries)
