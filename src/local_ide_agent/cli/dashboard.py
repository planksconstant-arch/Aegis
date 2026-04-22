"""
Live terminal training dashboard.

Displays real-time RL training metrics in the terminal using ANSI escape codes.
No external dependencies — pure Python stdlib + the existing project modules.

Usage
-----
  local-ide-agent dashboard              # auto-refreshes every 2 s
  local-ide-agent dashboard --interval 5  # refresh every 5 s
  local-ide-agent dashboard --once       # print once then exit

What it shows
-------------
  - Policy weights file age + size
  - Replay buffer fill %
  - Epsilon / PER beta
  - Reward history: last 10 episode rewards (sparkline)
  - Avg reward (last 100 eps)
  - Recent events from .agent/events.jsonl (last 5 entries)
  - Action success rates from memory store
  - Curiosity predictor convergence (RND weight file age)
  - Curriculum difficulty level
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 can't encode block characters)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

# Use ASCII fallbacks if UTF-8 is still not available
_UTF8 = (sys.stdout.encoding or "").lower().replace("-", "") in ("utf8", "utf8")
BAR_FULL  = "\u2588" if _UTF8 else "#"
BAR_EMPTY = "\u2591" if _UTF8 else "."
_SPARK_BLOCKS = " _.'\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588" if _UTF8 else " _.,:;|IH#"


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
CYAN    = "\033[36m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
WHITE   = "\033[97m"


def _clear() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _color_for(value: float, low: float, high: float, invert: bool = False) -> str:
    """Return GREEN / YELLOW / RED depending on where value sits in [low, high]."""
    fraction = max(0.0, min(1.0, (value - low) / max(high - low, 1e-9)))
    if invert:
        fraction = 1.0 - fraction
    if fraction >= 0.65:
        return GREEN
    if fraction >= 0.35:
        return YELLOW
    return RED


def _bar(fraction: float, width: int = 20) -> str:
    filled = round(fraction * width)
    return CYAN + BAR_FULL * filled + DIM + BAR_EMPTY * (width - filled) + RESET


def _sparkline(values: list[float], width: int = 20) -> str:
    """ASCII spark-line for a list of floats."""
    blocks = _SPARK_BLOCKS
    if not values:
        return DIM + "-" * width + RESET
    mn, mx = min(values), max(values)
    span = mx - mn or 1.0
    result = ""
    recent = values[-width:]
    for v in recent:
        idx = int((v - mn) / span * (len(blocks) - 1))
        c = GREEN if v >= 0 else RED
        result += c + blocks[idx] + RESET
    return result


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def _read_weight_info(weight_path: str) -> dict:
    p = Path(weight_path)
    if not p.exists():
        return {"exists": False}
    age = time.time() - p.stat().st_mtime
    size_kb = p.stat().st_size / 1024
    return {"exists": True, "age_s": round(age, 1), "size_kb": round(size_kb, 1)}


def _read_events(events_path: str, n: int = 6) -> list[dict]:
    p = Path(events_path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(line) for line in lines if line.strip()]
    except Exception:
        return []


def _read_replay_transitions(db_path: str, limit: int = 200) -> list[dict]:
    """Read recent replay transitions directly from sqlite for reward history."""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT reward, td_error FROM replay_transitions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [{"reward": float(r[0]), "td_error": float(r[1])} for r in reversed(rows)]
    except Exception:
        return []


def _read_action_rates(db_path: str) -> list[dict]:
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """
            SELECT action_type, total_count, accept_count,
                   ROUND(reward_sum / total_count, 3) AS avg_reward
            FROM action_success_rates
            WHERE total_count >= 1
            ORDER BY avg_reward DESC
            LIMIT 8
            """
        ).fetchall()
        conn.close()
        return [
            {"action": r[0], "total": r[1], "accepted": r[2], "avg_reward": float(r[3])}
            for r in rows
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _render(
    weight_path: str,
    events_path: str,
    db_path: str,
    rnd_weight_path: str,
    refresh_interval: float,
) -> None:
    w_info = _read_weight_info(weight_path)
    rnd_info = _read_weight_info(rnd_weight_path)
    events = _read_events(events_path)
    transitions = _read_replay_transitions(db_path)
    action_rates = _read_action_rates(db_path)

    rewards = [t["reward"] for t in transitions]
    td_errors = [t["td_error"] for t in transitions]
    avg_reward_100 = (sum(rewards[-100:]) / len(rewards[-100:])) if rewards else 0.0
    avg_td = (sum(td_errors[-20:]) / len(td_errors[-20:])) if td_errors else 0.0

    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    try:
        width = max(80, os.get_terminal_size().columns - 2)
    except OSError:
        width = 100

    def line(text: str = "") -> None:
        print(text)

    def header(title: str) -> None:
        pad = width - len(title) - 4
        line(BOLD + CYAN + f"  {title} " + DIM + "-" * max(pad, 0) + RESET)

    def kv(key: str, val: str, unit: str = "") -> None:
        line(f"  {DIM}{key:<28}{RESET}{val}{DIM}{unit}{RESET}")

    # ---- Title bar ----
    line(BOLD + BLUE + "=" * width + RESET)
    title = f"  LOCAL IDE RL AGENT  \u2014  TRAINING DASHBOARD  \u2014  {now}"
    line(BOLD + WHITE + title + RESET)
    line(BOLD + BLUE + "=" * width + RESET)
    line()

    # ---- Policy weights ----
    header("POLICY WEIGHTS")
    if w_info["exists"]:
        age_c = _color_for(w_info["age_s"], 0, 300, invert=True)
        kv("Weight file:", WHITE + weight_path + RESET)
        kv("Last saved:", age_c + f"{w_info['age_s']}s ago" + RESET)
        kv("File size:", f"{w_info['size_kb']} KB")
    else:
        kv("Status:", RED + "NOT FOUND \u2014 run `train` first" + RESET)

    if rnd_info["exists"]:
        rnd_age_c = _color_for(rnd_info["age_s"], 0, 300, invert=True)
        kv("RND curiosity weights:", rnd_age_c + f"{rnd_info['age_s']}s ago" + RESET)
    line()

    # ---- Reward history ----
    header("REWARD HISTORY")
    kv("Avg reward (last 100):", _color_for(avg_reward_100, -1, 1) + f"{avg_reward_100:+.4f}" + RESET)
    kv("Avg TD error (last 20):", _color_for(avg_td, 0, 50, invert=True) + f"{avg_td:.3f}" + RESET)
    kv("Transitions stored:", f"{len(transitions)}")
    if rewards:
        spark = _sparkline(rewards[-40:], width=min(40, width - 35))
        kv("Recent rewards:", spark)
    line()

    # ---- Action success rates ----
    if action_rates:
        header("ACTION SUCCESS RATES (materialised view)")
        for row in action_rates:
            rate_c = _color_for(row["avg_reward"], -0.5, 1.0)
            bar = _bar(max(0.0, min(1.0, (row["avg_reward"] + 0.5) / 1.5)), width=12)
            line(
                f"  {DIM}{row['action']:<28}{RESET}"
                f"{bar}  "
                f"{rate_c}avg_r={row['avg_reward']:+.3f}{RESET}  "
                f"{DIM}n={row['total']}{RESET}"
            )
        line()

    # ---- Recent events ----
    if events:
        header("RECENT EVENTS (.agent/events.jsonl)")
        for ev in events[-5:]:
            ts = ev.get("timestamp", "")[-8:]
            etype = ev.get("event_type", "?")
            payload = ev.get("payload", {})
            desc = str(payload.get("description", payload.get("command", payload.get("path", ""))))[:50]
            etype_c = GREEN if "apply" in etype or "suggest" in etype else YELLOW
            line(f"  {DIM}{ts}{RESET}  {etype_c}{etype:<26}{RESET}  {DIM}{desc}{RESET}")
        line()

    # ---- Status bar ----
    line(DIM + f"  Auto-refresh every {refresh_interval}s  \u2014  Ctrl+C to exit" + RESET)
    line(BOLD + BLUE + "=" * width + RESET)


def run_dashboard(
    weight_path: str = ".agent/policy_weights.npz",
    events_path: str = ".agent/events.jsonl",
    db_path: str = ".agent/agent.db",
    rnd_weight_path: str = ".agent/rnd_weights.npz",
    refresh_interval: float = 2.0,
    once: bool = False,
) -> None:
    """Entry-point for the dashboard. Loops until Ctrl+C."""
    try:
        while True:
            _clear()
            _render(weight_path, events_path, db_path, rnd_weight_path, refresh_interval)
            if once:
                break
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        _clear()
        print(RESET + "Dashboard closed.")
