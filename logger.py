"""
ERIC — Structured Activity Logger
===================================
Three log tiers:

  1. Activity log  — rolling in-memory buffer of brief one-liners (GUI feed)
  2. AI log        — every Cosmos prompt + response → logs/ai_TIMESTAMP.jsonl
  3. Mission log   — complete per-mission transcript → logs/mission_TIMESTAMP_NAME.jsonl
                     includes start time, end time, duration, every event and AI call

Usage:
  from logger import log_ai, log_action, log_mission_event
  from logger import start_mission_log, end_mission_log
  from logger import get_activity_tail

Integration points:
  cosmos.py   → log_ai() wraps every requests.post() call
  motors.py   → log_action() wraps every _send() call
  mission.py  → log_mission_event() at every state change and target event
  avoidance.py → log_action() for every avoid manoeuvre
"""

import json
import logging
import pathlib
import threading
import datetime
import traceback

log = logging.getLogger("eric.logger")

# ─── Directories ──────────────────────────────────────────────────────────────
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

PHOTO_DIR = pathlib.Path("missions/photos")
PHOTO_DIR.mkdir(parents=True, exist_ok=True)

# ─── Session-level AI log (one file per process start) ────────────────────────
_session_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_ai_log_path = LOG_DIR / f"ai_{_session_ts}.jsonl"

# ─── Per-mission state ─────────────────────────────────────────────────────────
_mission_log_path: pathlib.Path | None = None
_mission_start:    datetime.datetime | None = None
_mission_name:     str = "unnamed"
_mission_steps_summary: list[str] = []

# ─── Rolling in-memory activity buffer ────────────────────────────────────────
_activity_log: list[dict] = []
_ACTIVITY_MAX = 500

_lock = threading.Lock()


# ─── Public API ───────────────────────────────────────────────────────────────

def start_mission_log(mission_name: str, steps: list[str] | None = None):
    """
    Open a new mission log file. Call at start_mission().
    mission_name: short human-readable name (used in filename)
    steps: optional list of step descriptions for the header
    """
    global _mission_start, _mission_name, _mission_log_path, _mission_steps_summary

    _mission_start  = datetime.datetime.now()
    _mission_name   = mission_name
    _mission_steps_summary = steps or []

    ts          = _mission_start.strftime("%Y%m%d_%H%M%S")
    safe_name   = mission_name[:40].replace(" ", "_").replace("/", "_").replace("\\", "_")
    _mission_log_path = LOG_DIR / f"mission_{ts}_{safe_name}.jsonl"

    header = {
        "type":    "mission_start",
        "mission": mission_name,
        "started": _mission_start.isoformat(),
        "steps":   _mission_steps_summary,
    }
    _write_jsonl(_mission_log_path, header)
    log.info(f"📋 Mission log: {_mission_log_path}")


def end_mission_log(completed: bool = True):
    """
    Write summary record and close mission log. Call at mission end/abort.
    """
    if not _mission_start or not _mission_log_path:
        return
    duration = (datetime.datetime.now() - _mission_start).total_seconds()
    summary = {
        "type":       "mission_summary",
        "mission":    _mission_name,
        "started":    _mission_start.isoformat(),
        "ended":      datetime.datetime.now().isoformat(),
        "duration_s": round(duration, 1),
        "completed":  completed,
    }
    _write_jsonl(_mission_log_path, summary)
    status = "COMPLETE ✅" if completed else "ABORTED ❌"
    log.info(f"Mission {status} in {duration:.0f}s → {_mission_log_path}")


def log_ai(prompt: str, response: str, label: str = "COSMOS",
           prompt_tokens: int | None = None):
    """
    Log every Cosmos prompt + response.
    Writes to session AI log always; also to mission log when mission is active.
    prompt is truncated to 800 chars in the log (full prompts get large).
    """
    entry = {
        "type":     "ai",
        "label":    label,
        "ts":       _now(),
        "prompt":   prompt[-800:],      # tail is more informative (json at end)
        "response": response[:1200],
        "prompt_tokens": prompt_tokens,
    }
    _write_jsonl(_ai_log_path, entry)
    if _mission_log_path:
        _write_jsonl(_mission_log_path, entry)
    # Brief activity entry
    _append_activity({
        "type":   "ai",
        "ts":     _now(),
        "label":  label,
        "brief":  f"🧠 {label}: {response[:80]}",
    })


def log_action(action: str, detail: str = ""):
    """
    Log a robot action (motor, avoidance, pan-tilt, lights, etc.).
    Always written to activity buffer; also mission log when active.
    """
    entry = {
        "type":   "action",
        "ts":     _now(),
        "action": action,
        "detail": detail,
    }
    _append_activity({**entry, "brief": f"⚙️  {action}: {detail}"})
    if _mission_log_path:
        _write_jsonl(_mission_log_path, entry)


def log_mission_event(event: str, detail: str = ""):
    """
    Log a mission-level event: target found, step complete, state change, etc.
    Always written to mission log and activity buffer.
    """
    entry = {
        "type":   "mission_event",
        "ts":     _now(),
        "event":  event,
        "detail": detail,
    }
    _append_activity({**entry, "brief": f"🎯 {event}: {detail}"})
    if _mission_log_path:
        _write_jsonl(_mission_log_path, entry)
    log.info(f"MISSION EVENT: {event} — {detail}")


def log_exception(context: str, exc: Exception):
    """
    Structured exception logger — context + traceback in mission log.
    Use in except blocks instead of bare log.error(f"... {e}").
    """
    tb = traceback.format_exc(limit=6)
    entry = {
        "type":      "error",
        "ts":        _now(),
        "context":   context,
        "exception": f"{type(exc).__name__}: {exc}",
        "traceback": tb,
    }
    _append_activity({**entry, "brief": f"❌ ERROR {context}: {exc}"})
    if _mission_log_path:
        _write_jsonl(_mission_log_path, entry)
    log.error(f"{context}: {type(exc).__name__}: {exc}\n{tb}")


def get_activity_tail(n: int = 30) -> list[dict]:
    """Return the last n activity entries for GUI or debug display."""
    with _lock:
        return list(_activity_log[-n:])


def get_activity_text(n: int = 30) -> str:
    """Return last n activity entries as a human-readable string for GUI."""
    entries = get_activity_tail(n)
    lines = []
    for e in entries:
        ts_raw = e.get("ts", "")
        ts_short = ts_raw[11:19] if len(ts_raw) >= 19 else ts_raw   # HH:MM:SS
        brief = e.get("brief", e.get("event", e.get("action", "?")))
        lines.append(f"[{ts_short}] {brief}")
    return "\n".join(lines)


def get_mission_log_path() -> pathlib.Path | None:
    return _mission_log_path


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def _append_activity(entry: dict):
    with _lock:
        _activity_log.append(entry)
        if len(_activity_log) > _ACTIVITY_MAX:
            _activity_log.pop(0)


def _write_jsonl(path: pathlib.Path, entry: dict):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        log.warning(f"Log write error ({path.name}): {e}")
