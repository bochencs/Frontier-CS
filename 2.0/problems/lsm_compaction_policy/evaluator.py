"""Evaluator for the Frontier-CS 2.0 LSM compaction policy task."""

from __future__ import annotations

import ast
import json
import math
import os
import pwd
import random
import secrets
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_VERSION = "lsm-compaction-policy-v2"
SEED = int(os.environ.get("FRONTIER_LSM_SEED", "20260614"))
POLICY_TIMEOUT_SECONDS = float(os.environ.get("FRONTIER_LSM_POLICY_TIMEOUT", "2.5"))
TOTAL_TIMEOUT_SECONDS = float(os.environ.get("FRONTIER_LSM_TOTAL_TIMEOUT", "55"))
MAX_SOURCE_BYTES = 250_000
MAX_PROTOCOL_LINE_BYTES = 200_000
POLICY_MEMORY_BYTES = int(os.environ.get("FRONTIER_LSM_POLICY_MEMORY_BYTES", str(512 * 1024 * 1024)))
POLICY_CPU_SECONDS = int(os.environ.get("FRONTIER_LSM_POLICY_CPU_SECONDS", "5"))
POLICY_NPROC_LIMIT = int(os.environ.get("FRONTIER_LSM_POLICY_NPROC", "16"))
POLICY_NOFILE_LIMIT = int(os.environ.get("FRONTIER_LSM_POLICY_NOFILE", "32"))
POLICY_FSIZE_BYTES = int(os.environ.get("FRONTIER_LSM_POLICY_FSIZE_BYTES", str(1024 * 1024)))

ALLOWED_IMPORTS = {
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "random",
}

BANNED_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "open",
    "input",
    "breakpoint",
    "help",
    "license",
    "credits",
}

BANNED_ATTRIBUTES = BANNED_NAMES | {
    "sys",
    "os",
    "modules",
    "builtins",
    "attrgetter",
    "methodcaller",
}


@dataclass
class Case:
    name: str
    family: str
    config: dict[str, Any]
    epochs: list[list[dict[str, Any]]]
    baseline_cost: float
    reference_cost: float


_BENCHMARKS: dict[str, list[Case]] = {}
_TRIAL_SEED: int | None = None


def _protect_evaluator_source() -> None:
    try:
        evaluator_path = Path(__file__).resolve()
        if str(evaluator_path).startswith(("/judge/", "/tests/")) and os.geteuid() == 0:
            evaluator_path.chmod(0o600)
    except Exception:
        pass


_protect_evaluator_source()


def _role() -> str:
    raw = os.environ.get("FRONTIER_SUBMISSION_ROLE", "final").strip().lower()
    return "final" if raw in {"final", "verifier"} else "feedback"


try:
    import resource
except Exception:  # pragma: no cover - non-POSIX fallback
    resource = None


def _set_limit(kind: Any, value: int) -> None:
    if resource is None or kind is None:
        return
    try:
        hard = resource.getrlimit(kind)[1]
        limit = value if hard in (-1, resource.RLIM_INFINITY) else min(value, hard)
        resource.setrlimit(kind, (limit, limit))
    except Exception:
        pass


def _solution_preexec():
    if os.name != "posix":
        return None
    nobody = None
    try:
        if os.geteuid() == 0:
            nobody = pwd.getpwnam("nobody")
    except Exception:
        nobody = None

    def configure_child() -> None:
        try:
            os.setsid()
        except Exception:
            pass
        _set_limit(getattr(resource, "RLIMIT_AS", None), POLICY_MEMORY_BYTES)
        _set_limit(getattr(resource, "RLIMIT_CPU", None), POLICY_CPU_SECONDS)
        _set_limit(getattr(resource, "RLIMIT_NPROC", None), POLICY_NPROC_LIMIT)
        _set_limit(getattr(resource, "RLIMIT_NOFILE", None), POLICY_NOFILE_LIMIT)
        _set_limit(getattr(resource, "RLIMIT_FSIZE", None), POLICY_FSIZE_BYTES)
        if nobody is not None:
            try:
                os.setgid(nobody.pw_gid)
                os.setuid(nobody.pw_uid)
            except Exception:
                pass

    return configure_child


def _candidate_source(solution_path: Path) -> str:
    if solution_path.is_dir():
        candidate = solution_path / "solution.py"
    else:
        candidate = solution_path
    if not candidate.exists():
        raise ValueError("solution.py not found")
    if candidate.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("solution.py exceeds size limit")
    return candidate.read_text(encoding="utf-8", errors="replace")


def _check_source_safety(source: str) -> None:
    if "__" in source:
        raise ValueError("dunder names are forbidden")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORTS:
                    raise ValueError(f"forbidden import: {root}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORTS:
                raise ValueError(f"forbidden import: {root}")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise ValueError(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BANNED_ATTRIBUTES):
            raise ValueError("private or introspective attribute access is forbidden")
    for needle in ("/tmp", "/proc", "/dev", "subprocess", "socket", "importlib", "inspect"):
        if needle in source:
            raise ValueError(f"disallowed source pattern: {needle}")


RUNNER_SOURCE = r'''
import builtins
import contextlib
import importlib.util
import json
import os
import sys

proto = os.fdopen(int(os.environ["FRONTIER_LSM_PROTO_FD"]), "w", buffering=1)
sys.stdout = sys.stderr
sys.__stdout__ = sys.stderr

ALLOWED_IMPORTS = {
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "random",
}

REAL_IMPORT = builtins.__import__


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if level != 0 or root not in ALLOWED_IMPORTS:
        raise ImportError(f"forbidden import: {root}")
    for item in fromlist or ():
        if str(item).startswith("_"):
            raise ImportError("private imports are forbidden")
    return REAL_IMPORT(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "__build_class__": builtins.__build_class__,
    "__import__": safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "callable": callable,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError,
    "zip": zip,
}

solution_path = os.environ["FRONTIER_LSM_SOLUTION"]
spec = importlib.util.spec_from_file_location("solution", solution_path)
module = importlib.util.module_from_spec(spec)
module.__dict__["__builtins__"] = SAFE_BUILTINS
try:
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
except Exception:
    proto.write(json.dumps({"ok": False, "error": "import failed"}) + "\n")
    proto.flush()
    raise SystemExit(0)

state = None

for line in sys.stdin:
    try:
        msg = json.loads(line)
        op = msg.get("op")
        if op == "init":
            fn = getattr(module, "init", None)
            with contextlib.redirect_stdout(sys.stderr):
                state = fn(msg.get("config", {})) if callable(fn) else {}
            proto.write(json.dumps({"ok": True}) + "\n")
        elif op == "choose":
            fn = getattr(module, "choose", None)
            if not callable(fn):
                action = {"type": "none"}
            else:
                with contextlib.redirect_stdout(sys.stderr):
                    action = fn(state, msg.get("observation", {}))
            proto.write(json.dumps({"ok": True, "action": action}) + "\n")
        elif op == "close":
            proto.write(json.dumps({"ok": True}) + "\n")
            proto.flush()
            break
        else:
            proto.write(json.dumps({"ok": False, "error": "bad op"}) + "\n")
        proto.flush()
    except Exception:
        proto.write(json.dumps({"ok": False, "error": "policy call failed"}) + "\n")
        proto.flush()
'''


class PolicyProcess:
    def __init__(self, source: str, deadline: float | None = None):
        self.tmp = tempfile.TemporaryDirectory(prefix="frontier_lsm_policy_")
        self.root = Path(self.tmp.name)
        self.solution_path = self.root / "solution.py"
        self.runner_path = self.root / "runner.py"
        self.solution_path.write_text(source, encoding="utf-8")
        self.runner_path.write_text(RUNNER_SOURCE, encoding="utf-8")
        preexec_fn = _solution_preexec()
        if os.name == "posix" and os.geteuid() == 0:
            nobody = pwd.getpwnam("nobody")
            os.chown(self.root, nobody.pw_uid, nobody.pw_gid)
            os.chown(self.solution_path, nobody.pw_uid, nobody.pw_gid)
            os.chown(self.runner_path, nobody.pw_uid, nobody.pw_gid)
            os.chmod(self.root, 0o755)
        self.proto_fd, child_proto_fd = os.pipe()
        os.set_inheritable(child_proto_fd, True)
        os.set_blocking(self.proto_fd, False)
        self.protocol_buffer = b""
        env = {
            "FRONTIER_LSM_SOLUTION": str(self.solution_path),
            "FRONTIER_LSM_PROTO_FD": str(child_proto_fd),
            "PYTHONIOENCODING": "utf-8",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
        }
        try:
            self.proc = subprocess.Popen(
                [sys.executable, str(self.runner_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
                preexec_fn=preexec_fn,
                pass_fds=(child_proto_fd,),
            )
        finally:
            os.close(child_proto_fd)
        self.pgid = self.proc.pid
        assert self.proc.stdin is not None
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proto_fd, selectors.EVENT_READ)
        self.deadline = deadline if deadline is not None else time.perf_counter() + TOTAL_TIMEOUT_SECONDS

    def _kill_group(self) -> None:
        if os.name == "posix":
            try:
                os.killpg(self.pgid, signal.SIGKILL)
                return
            except ProcessLookupError:
                pass
            except Exception:
                pass
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self._request({"op": "close"}, 0.2)
        except Exception:
            pass
        try:
            self._kill_group()
            try:
                self.proc.wait(timeout=0.5)
            except Exception:
                pass
        finally:
            try:
                self.selector.unregister(self.proto_fd)
            except Exception:
                pass
            try:
                os.close(self.proto_fd)
            except Exception:
                pass
            self.tmp.cleanup()

    def _read_protocol_line(self, timeout: float) -> str:
        end_time = min(self.deadline, time.perf_counter() + timeout)
        while True:
            if b"\n" in self.protocol_buffer:
                line, self.protocol_buffer = self.protocol_buffer.split(b"\n", 1)
                return line.decode("utf-8", errors="replace")
            if len(self.protocol_buffer) > MAX_PROTOCOL_LINE_BYTES:
                raise RuntimeError("policy response too large")
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("policy timed out")
            events = self.selector.select(remaining)
            if not events:
                raise TimeoutError("policy timed out")
            try:
                chunk = os.read(self.proto_fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                raise RuntimeError("policy process exited")
            self.protocol_buffer += chunk

    def _request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if self.proc.poll() is not None:
            raise RuntimeError("policy process exited")
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("policy total timeout")
        wait_time = min(timeout if timeout is not None else POLICY_TIMEOUT_SECONDS, remaining)
        self.proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self._read_protocol_line(wait_time)
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "policy failed")))
        return response

    def init(self, config: dict[str, Any]) -> None:
        self._request({"op": "init", "config": config})

    def choose(self, observation: dict[str, Any]) -> Any:
        return self._request({"op": "choose", "observation": observation}).get("action")


class InternalPolicy:
    def init(self, config: dict[str, Any]) -> None:
        pass

    def choose(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {"type": "none"}

    def close(self) -> None:
        pass


class NoopPolicy(InternalPolicy):
    pass


class LeveledPolicy(InternalPolicy):
    def choose(self, observation: dict[str, Any]) -> dict[str, Any]:
        levels = observation["levels"]
        if levels[0]["runs"] >= 5:
            return {"type": "compact", "level": 0, "pick": "oldest", "max_runs": 5}
        for level in levels[:-1]:
            if level["bytes"] > 1.08 * level["target_bytes"]:
                return {"type": "compact", "level": level["level"], "pick": "largest", "max_runs": 4}
        return {"type": "none"}


class ReferencePolicy(InternalPolicy):
    def init(self, config: dict[str, Any]) -> None:
        self.steps = 0
        self.write_ewma = 0.0
        self.read_ewma = 0.0
        self.range_ewma = 0.0
        self.delete_ewma = 0.0
        self.memtable_bytes = float(config.get("memtable_bytes", 4096))

    @staticmethod
    def _best_level(levels: list[dict[str, Any]], key: str) -> int:
        return max(range(len(levels) - 1), key=lambda i: (levels[i][key], levels[i]["runs"]))

    def choose(self, observation: dict[str, Any]) -> dict[str, Any]:
        recent = observation["recent"]
        levels = observation["levels"]
        self.steps += 1
        alpha = 0.30 if self.steps > 1 else 1.0
        self.write_ewma = (1.0 - alpha) * self.write_ewma + alpha * recent["write_frac"]
        self.read_ewma = (1.0 - alpha) * self.read_ewma + alpha * recent["read_frac"]
        self.range_ewma = (1.0 - alpha) * self.range_ewma + alpha * recent["range_frac"]
        self.delete_ewma = (1.0 - alpha) * self.delete_ewma + alpha * recent["delete_frac"]

        l0_runs = int(levels[0]["runs"])
        stall = float(observation["stall_pressure"])
        tomb_level = self._best_level(levels, "tombstone_frac")
        span_level = self._best_level(levels, "span_frac")
        hot_level = self._best_level(levels, "hotness")
        pressure_level, pressure = _overloaded_level(levels[:-1])

        if levels[tomb_level]["runs"] and (
            levels[tomb_level]["tombstone_frac"] > 0.25 or (self.delete_ewma > 0.18 and levels[tomb_level]["tombstone_frac"] > 0.16)
        ):
            return {"type": "compact", "level": tomb_level, "pick": "tombstone", "max_runs": 3}

        l0_target = 5
        if self.memtable_bytes < 2600 or stall > 1.25:
            l0_target = 4
        if self.range_ewma > 0.20 or self.read_ewma > 0.50:
            l0_target = min(l0_target, 4)
        if self.write_ewma > 0.68 and self.range_ewma < 0.08 and stall <= 1.10:
            l0_target = max(l0_target, 6)

        if l0_runs >= l0_target:
            pick = "overlap" if self.range_ewma > 0.16 or levels[0]["span_frac"] > 0.045 else "oldest"
            return {"type": "compact", "level": 0, "pick": pick, "max_runs": max(3, min(6, l0_runs))}

        if self.range_ewma > 0.18 and levels[span_level]["runs"] > 1 and levels[span_level]["span_frac"] > 0.025:
            return {"type": "compact", "level": span_level, "pick": "overlap", "max_runs": 4}

        if self.read_ewma > 0.48 and levels[hot_level]["runs"] > 1 and levels[hot_level]["hotness"] > 0.34:
            return {"type": "compact", "level": hot_level, "pick": "hot", "max_runs": 3}

        if pressure_level is not None and pressure > (0.82 if self.range_ewma > 0.16 or self.delete_ewma > 0.16 else 0.95):
            pick = "overlap" if self.range_ewma > 0.16 and levels[pressure_level]["span_frac"] > 0.020 else "largest"
            return {"type": "compact", "level": pressure_level, "pick": pick, "max_runs": 4}

        return {"type": "none"}


def _overloaded_level(levels: list[dict[str, Any]]) -> tuple[int | None, float]:
    best_level = None
    best_pressure = 1.0
    for level in levels:
        pressure = float(level["bytes"]) / max(float(level["target_bytes"]), 1.0)
        if pressure > best_pressure:
            best_pressure = pressure
            best_level = int(level["level"])
    return best_level, best_pressure


def _case_specs(role: str) -> list[tuple[str, str, int, int]]:
    if "FRONTIER_LSM_CASES" in os.environ:
        return [("mixed", "base", 60, 45), ("tombstone_churn", "base", 60, 45), ("range_scan", "wide_ranges", 60, 45)]
    if role == "feedback":
        return [
            ("write_burst", "base", 95, 45),
            ("hot_reads", "base", 105, 45),
            ("range_scan", "base", 95, 45),
            ("tombstone_churn", "base", 105, 45),
            ("skew_shift", "base", 110, 45),
            ("mixed", "base", 110, 45),
        ]
    return [
        ("write_burst", "base", 150, 60),
        ("hot_reads", "base", 165, 60),
        ("range_scan", "base", 150, 60),
        ("tombstone_churn", "base", 170, 60),
        ("skew_shift", "base", 170, 60),
        ("mixed", "base", 175, 60),
        ("write_burst", "small_memtable", 155, 60),
        ("hot_reads", "cold_hotset", 150, 60),
        ("range_scan", "wide_ranges", 165, 60),
        ("tombstone_churn", "delete_wave", 160, 60),
        ("skew_shift", "fast_shift", 170, 60),
        ("mixed", "phase_switch", 175, 60),
        ("range_scan", "small_targets", 150, 60),
        ("mixed", "wide_phase_switch", 165, 60),
    ]


def _weighted_key(rng: random.Random, keyspace: int, hot_start: int, hot_width: int, hot_prob: float) -> int:
    if rng.random() < hot_prob:
        return (hot_start + rng.randrange(max(1, hot_width))) % keyspace
    return rng.randrange(keyspace)


def _make_epochs(
    rng: random.Random,
    family: str,
    variant: str,
    epochs: int,
    events_per_epoch: int,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    wide_variant = variant in {"wide_ranges", "wide_phase_switch"}
    small_targets = variant in {"small_memtable", "small_targets"}
    keyspace = rng.randrange(220_000, 420_000) if wide_variant else rng.randrange(80_000, 180_000)
    hot_width = rng.randrange(12_000, 55_000) if wide_variant else rng.randrange(1200, 9000)
    hot_start = rng.randrange(keyspace)
    if variant == "small_memtable":
        memtable_bytes = rng.randrange(1250, 2400)
    elif wide_variant:
        memtable_bytes = rng.randrange(5200, 8800)
    else:
        memtable_bytes = rng.randrange(3000, 5200)
    level_targets = [18_000, 72_000, 288_000, 1_150_000, 4_600_000]
    if variant == "small_memtable":
        level_targets = [10_000, 42_000, 170_000, 700_000, 2_900_000]
    elif variant == "small_targets":
        level_targets = [12_000, 38_000, 155_000, 620_000, 2_500_000]
    elif wide_variant:
        level_targets = [26_000, 105_000, 420_000, 1_700_000, 6_800_000]
    config = {
        "levels": 5,
        "memtable_bytes": memtable_bytes,
        "level_targets": level_targets,
        "keyspace": keyspace,
        "valid_picks": ["oldest", "largest", "overlap", "tombstone", "hot"],
    }
    out: list[list[dict[str, Any]]] = []
    for epoch in range(epochs):
        shift_period = max(12, epochs // 6) if variant == "fast_shift" else max(24, epochs // 4)
        if family == "skew_shift" and epoch % shift_period == 0:
            hot_start = rng.randrange(keyspace)
        if family == "write_burst":
            cycle = 11 if variant == "small_memtable" else 18
            write_p = 0.82 if (epoch // cycle) % 3 != 2 else 0.43
            read_p = 0.14
            range_p = 0.04
            delete_p = 0.06
            hot_p = 0.68 if variant == "small_memtable" else 0.72
        elif family == "hot_reads":
            write_p = 0.36 if variant == "cold_hotset" else 0.32
            read_p = 0.52 if variant == "cold_hotset" else 0.58
            range_p = 0.04
            delete_p = 0.04
            hot_p = 0.62 if variant == "cold_hotset" else 0.88
        elif family == "range_scan":
            write_p = 0.38 if variant in {"wide_ranges", "small_targets"} else 0.36
            read_p = 0.22 if variant == "wide_ranges" else 0.25
            range_p = 0.36 if variant in {"wide_ranges", "small_targets"} else 0.32
            delete_p = 0.05
            hot_p = 0.44 if variant == "wide_ranges" else 0.52
        elif family == "tombstone_churn":
            write_p = 0.50 if variant == "delete_wave" else 0.47
            read_p = 0.25 if variant == "delete_wave" else 0.30
            range_p = 0.11 if variant == "delete_wave" else 0.05
            if variant == "delete_wave":
                delete_p = 0.10 if epoch < epochs // 3 else (0.42 if epoch < 2 * epochs // 3 else 0.24)
            else:
                delete_p = 0.30 if epoch > epochs // 5 else 0.12
            hot_p = 0.76
        elif family == "skew_shift":
            write_p = 0.42
            read_p = 0.42
            range_p = 0.12 if variant == "fast_shift" else 0.08
            delete_p = 0.08
            hot_p = 0.76 if variant == "fast_shift" else 0.84
        else:
            if variant in {"phase_switch", "wide_phase_switch"}:
                if epoch < epochs // 3:
                    write_p, read_p, range_p, delete_p, hot_p = 0.72, 0.16, 0.04, 0.05, 0.70
                elif epoch < 2 * epochs // 3:
                    write_p, read_p, range_p, delete_p, hot_p = 0.36, 0.24, 0.32, 0.07, 0.48
                else:
                    write_p, read_p, range_p, delete_p, hot_p = 0.46, 0.28, 0.10, 0.28, 0.72
            else:
                write_p = 0.45 + 0.20 * (math.sin(epoch / 11.0) > 0)
                read_p = 0.32
                range_p = 0.10 + 0.08 * (epoch % 37 < 9)
                delete_p = 0.09 + 0.10 * (epoch % 41 > 31)
                hot_p = 0.68
        epoch_events = []
        for _ in range(events_per_epoch):
            r = rng.random()
            key = _weighted_key(rng, keyspace, hot_start, hot_width, hot_p)
            if r < write_p:
                size = rng.randrange(40, 220)
                is_delete = rng.random() < delete_p
                epoch_events.append({"op": "write", "key": key, "size": size, "delete": is_delete})
            elif r < write_p + read_p:
                epoch_events.append({"op": "read", "key": key})
            else:
                if variant in {"wide_ranges", "wide_phase_switch"}:
                    span = rng.randrange(2200, 28_000)
                elif family == "range_scan":
                    span = rng.randrange(50, 3500)
                else:
                    span = rng.randrange(50, 900)
                epoch_events.append({"op": "range", "start": key, "end": min(keyspace, key + span)})
        out.append(epoch_events)
    return config, out


def _summarize_levels(levels: list[list[dict[str, Any]]], targets: list[int], keyspace: int) -> list[dict[str, Any]]:
    summary = []
    for index, runs in enumerate(levels):
        total = sum(run["size"] for run in runs)
        tomb = sum(run["tomb"] for run in runs)
        hot = sum(run["hot"] * run["size"] for run in runs)
        span = sum(max(1, run["max"] - run["min"] + 1) for run in runs)
        summary.append(
            {
                "level": index,
                "runs": len(runs),
                "bytes": round(total, 3),
                "target_bytes": targets[index],
                "tombstone_frac": round(tomb / max(total, 1.0), 4),
                "hotness": round(hot / max(total, 1.0), 4),
                "span_frac": round(span / max(1, keyspace * max(len(runs), 1)), 4),
            }
        )
    return summary


def _overlaps(run: dict[str, Any], start: int, end: int) -> bool:
    return not (run["max"] < start or run["min"] > end)


def _run_key_overlap_cost(levels: list[list[dict[str, Any]]], key: int) -> tuple[int, float]:
    overlaps = 0
    hot_bonus = 0.0
    for level_index, runs in enumerate(levels):
        matched = [run for run in runs if _overlaps(run, key, key)]
        if level_index == 0:
            overlaps += len(matched)
        elif matched:
            overlaps += 1 + min(2, len(matched) - 1)
        hot_bonus += sum(run["hot"] for run in matched) * 0.05
    return overlaps, hot_bonus


def _range_overlap_bytes(levels: list[list[dict[str, Any]]], start: int, end: int) -> float:
    width = max(1, end - start + 1)
    cost = 0.0
    for runs in levels:
        for run in runs:
            if _overlaps(run, start, end):
                span = max(1, run["max"] - run["min"] + 1)
                cost += run["size"] * min(1.0, width / span + 0.04)
    return cost


def _select_runs(runs: list[dict[str, Any]], pick: str, max_runs: int) -> list[dict[str, Any]]:
    if not runs:
        return []
    if pick == "largest":
        ordered = sorted(runs, key=lambda run: run["size"], reverse=True)
    elif pick == "tombstone":
        ordered = sorted(runs, key=lambda run: run["tomb"] / max(run["size"], 1.0), reverse=True)
    elif pick == "hot":
        ordered = sorted(runs, key=lambda run: run["hot"], reverse=True)
    elif pick == "overlap":
        ordered = sorted(runs, key=lambda run: (run["max"] - run["min"], run["size"]), reverse=True)
    else:
        ordered = sorted(runs, key=lambda run: run["age"])
    return ordered[: max(1, min(max_runs, len(ordered)))]


def _apply_compaction(state: dict[str, Any], action: Any) -> float:
    if not isinstance(action, dict):
        state["invalid_actions"] += 1
        return 35.0
    if action.get("type", "none") in {"none", None, ""}:
        return 0.0
    if action.get("type") != "compact":
        state["invalid_actions"] += 1
        return 35.0
    try:
        level = int(action.get("level", 0))
        max_runs = int(action.get("max_runs", 4))
    except Exception:
        state["invalid_actions"] += 1
        return 35.0
    pick = str(action.get("pick", "oldest"))
    levels = state["levels"]
    if (
        level < 0
        or level >= len(levels) - 1
        or max_runs < 1
        or pick not in {"oldest", "largest", "overlap", "tombstone", "hot"}
    ):
        state["invalid_actions"] += 1
        return 35.0
    selected = _select_runs(levels[level], pick, max_runs)
    if not selected:
        return 2.0
    min_key = min(run["min"] for run in selected)
    max_key = max(run["max"] for run in selected)
    lower_matches = [] if pick == "tombstone" else [run for run in levels[level + 1] if _overlaps(run, min_key, max_key)]
    inputs = selected + lower_matches
    input_ids = {run["id"] for run in inputs}
    input_size = sum(run["size"] for run in inputs)
    live = sum(run["live"] for run in inputs)
    tomb = sum(run["tomb"] for run in inputs)
    hot = sum(run["hot"] * run["size"] for run in inputs) / max(input_size, 1.0)
    if pick == "tombstone":
        new_tomb = 0.0
    else:
        drop_tomb = level + 1 == len(levels) - 1
        new_tomb = 0.0 if drop_tomb else tomb * 0.72
    new_size = max(live + new_tomb, input_size * 0.18)
    state["next_run_id"] += 1
    new_run = {
        "id": state["next_run_id"],
        "size": new_size,
        "live": live,
        "tomb": new_tomb,
        "min": min(run["min"] for run in inputs),
        "max": max(run["max"] for run in inputs),
        "hot": min(1.0, hot * 0.94),
        "age": state["clock"],
    }
    levels[level] = [run for run in levels[level] if run["id"] not in input_ids]
    levels[level + 1] = [run for run in levels[level + 1] if run["id"] not in input_ids]
    output_level = level if pick == "tombstone" else level + 1
    levels[output_level].append(new_run)
    factor = 0.18 + 0.03 * level if pick == "tombstone" else 0.30 + 0.05 * level
    return input_size * factor


def _flush_memtable(state: dict[str, Any]) -> None:
    mem = state["mem"]
    if mem["size"] <= 0:
        return
    state["next_run_id"] += 1
    size = float(mem["size"])
    live = float(mem["live"])
    tomb = float(mem["tomb"])
    run = {
        "id": state["next_run_id"],
        "size": size,
        "live": live,
        "tomb": tomb,
        "min": mem["min"],
        "max": mem["max"],
        "hot": mem["hot"] / max(mem["events"], 1),
        "age": state["clock"],
    }
    state["levels"][0].append(run)
    state["mem"] = {"size": 0.0, "live": 0.0, "tomb": 0.0, "min": 10**18, "max": -1, "hot": 0.0, "events": 0}


def _observe(state: dict[str, Any], recent: dict[str, float], config: dict[str, Any]) -> dict[str, Any]:
    levels = _summarize_levels(state["levels"], config["level_targets"], int(config["keyspace"]))
    l0_pressure = max(0.0, levels[0]["runs"] - 6) / 3.0
    byte_pressure = max(
        [0.0]
        + [
            level["bytes"] / max(level["target_bytes"], 1.0) - 1.0
            for level in levels[:-1]
        ]
    )
    total_events = max(recent["events"], 1.0)
    return {
        "step": state["clock"],
        "levels": levels,
        "memtable_bytes": round(state["mem"]["size"], 3),
        "stall_pressure": round(1.0 + l0_pressure + byte_pressure, 4),
        "recent": {
            "events": int(recent["events"]),
            "write_frac": round(recent["writes"] / total_events, 4),
            "read_frac": round(recent["reads"] / total_events, 4),
            "range_frac": round(recent["ranges"] / total_events, 4),
            "delete_frac": round(recent["deletes"] / max(recent["writes"], 1.0), 4),
            "avg_write_bytes": round(recent["write_bytes"] / max(recent["writes"], 1.0), 3),
        },
    }


def _simulate(policy: Any, case: Case) -> tuple[float, int]:
    config = json.loads(json.dumps(case.config))
    policy.init(config)
    state: dict[str, Any] = {
        "levels": [[] for _ in range(config["levels"])],
        "mem": {"size": 0.0, "live": 0.0, "tomb": 0.0, "min": 10**18, "max": -1, "hot": 0.0, "events": 0},
        "next_run_id": 0,
        "clock": 0,
        "invalid_actions": 0,
    }
    total_cost = 0.0
    recent = {"events": 0.0, "writes": 0.0, "reads": 0.0, "ranges": 0.0, "deletes": 0.0, "write_bytes": 0.0}
    for epoch in case.epochs:
        for event in epoch:
            state["clock"] += 1
            recent["events"] += 1
            op = event["op"]
            if op == "write":
                size = float(event["size"])
                key = int(event["key"])
                is_delete = bool(event.get("delete"))
                recent["writes"] += 1
                recent["write_bytes"] += size
                recent["deletes"] += 1 if is_delete else 0
                total_cost += 0.32 * size
                mem = state["mem"]
                mem["size"] += size
                mem["live"] += 0.0 if is_delete else size
                mem["tomb"] += size if is_delete else 0.0
                mem["min"] = min(mem["min"], key)
                mem["max"] = max(mem["max"], key)
                mem["hot"] += 1.0 if key % 97 < 19 else 0.25
                mem["events"] += 1
                if mem["size"] >= config["memtable_bytes"]:
                    _flush_memtable(state)
            elif op == "read":
                recent["reads"] += 1
                overlaps, hot_bonus = _run_key_overlap_cost(state["levels"], int(event["key"]))
                total_cost += 18.0 + 5.8 * overlaps - min(3.0, hot_bonus)
            else:
                recent["ranges"] += 1
                overlap_bytes = _range_overlap_bytes(state["levels"], int(event["start"]), int(event["end"]))
                width = max(1, int(event["end"]) - int(event["start"]) + 1)
                total_cost += 25.0 + 0.0025 * width + 0.011 * overlap_bytes
        _flush_memtable(state)
        observation = _observe(state, recent, config)
        action = policy.choose(observation)
        total_cost += _apply_compaction(state, action)
        total_size = sum(run["size"] for runs in state["levels"] for run in runs) + state["mem"]["size"]
        total_cost += 0.0030 * total_size
        l0_runs = len(state["levels"][0])
        if l0_runs > 8:
            total_cost += 45.0 * (l0_runs - 8) ** 2
        for idx, target in enumerate(config["level_targets"][:-1]):
            level_bytes = sum(run["size"] for run in state["levels"][idx])
            if level_bytes > 1.35 * target:
                total_cost += 0.015 * (level_bytes - 1.35 * target)
        recent = {key: value * 0.35 for key, value in recent.items()}
    for _ in range(6):
        observation = _observe(state, recent, config)
        total_cost += _apply_compaction(state, policy.choose(observation))
    return total_cost, int(state["invalid_actions"])


def _seed_for_role(role: str) -> int:
    base = SEED if _TRIAL_SEED is None else _TRIAL_SEED
    return base + (19_000_001 if role == "final" else 0)


def _build_cases(role: str) -> list[Case]:
    seed = _seed_for_role(role)
    cases: list[Case] = []
    for idx, (family, variant, epochs, events_per_epoch) in enumerate(_case_specs(role)):
        rng = random.Random(seed + 1009 * idx)
        config, case_epochs = _make_epochs(rng, family, variant, epochs, events_per_epoch)
        blank = Case(f"case_{idx:02d}", family, config, case_epochs, 0.0, 0.0)
        noop_cost, _ = _simulate(NoopPolicy(), blank)
        leveled_cost, _ = _simulate(LeveledPolicy(), blank)
        reference_cost, _ = _simulate(ReferencePolicy(), blank)
        baseline_cost = leveled_cost if reference_cost < 0.99 * leveled_cost else noop_cost
        if reference_cost >= baseline_cost:
            reference_cost = baseline_cost * 0.995
        cases.append(Case(blank.name, family, config, case_epochs, baseline_cost, reference_cost))
    return cases


def _benchmark(role: str) -> list[Case]:
    if role not in _BENCHMARKS:
        _BENCHMARKS[role] = _build_cases(role)
    return _BENCHMARKS[role]


def prepare() -> dict[str, Any]:
    global _TRIAL_SEED
    _TRIAL_SEED = int(os.environ.get("FRONTIER_LSM_SEED") or secrets.randbits(63))
    _BENCHMARKS.clear()
    feedback_cases = _benchmark("feedback")
    final_cases = _benchmark("final")
    return {"feedback_cases": len(feedback_cases), "final_cases": len(final_cases), "trial_seeded": True}


def _invalid(message: str, metrics: dict[str, Any] | None = None):
    return 0.0, 0.0, message, metrics or {}


def _case_score(case: Case, cost: float) -> float:
    denom = max(case.baseline_cost - case.reference_cost, 1e-9)
    return 100.0 * max(0.0, min(1.0, (case.baseline_cost - cost) / denom))


def evaluate(solution_path: str):
    role = _role()
    cases = _benchmark(role)
    try:
        source = _candidate_source(Path(solution_path))
        _check_source_safety(source)
    except Exception as exc:
        return _invalid(f"role={role}; invalid submission", {"role": role, "valid": 0, "reason": str(exc)[:120]})

    total_cost = 0.0
    invalid_actions = 0
    family_scores: dict[str, list[float]] = {}
    started = time.perf_counter()
    deadline = started + TOTAL_TIMEOUT_SECONDS
    try:
        for case in cases:
            if time.perf_counter() >= deadline:
                raise TimeoutError("policy total timeout")
            policy = PolicyProcess(source, deadline)
            try:
                cost, bad_actions = _simulate(policy, case)
                total_cost += cost
                invalid_actions += bad_actions
                family_scores.setdefault(case.family, []).append(_case_score(case, cost))
            finally:
                policy.close()
    except TimeoutError:
        return _invalid(f"role={role}; policy timed out", {"role": role, "valid": 0, "cases": len(cases)})
    except Exception as exc:
        return _invalid(f"role={role}; policy failed", {"role": role, "valid": 0, "reason": type(exc).__name__})

    family_means = sorted(sum(scores) / len(scores) for scores in family_scores.values())
    macro = sum(family_means) / len(family_means)
    worst = family_means[0]
    second = family_means[min(1, len(family_means) - 1)]
    bottom_two = sum(family_means[: min(2, len(family_means))]) / min(2, len(family_means))
    score = min(
        0.55 * macro + 0.25 * worst + 0.20 * bottom_two,
        10.0 + 0.90 * worst,
        55.0 + 0.45 * second,
    )
    if invalid_actions:
        score *= max(0.0, 1.0 - min(0.35, invalid_actions * 0.01))
    elapsed = time.perf_counter() - started
    score = max(0.0, min(100.0, score))
    metrics = {
        "role": role,
        "valid": 1,
        "cases": len(cases),
        "total_cost": total_cost,
        "invalid_actions": invalid_actions,
        "candidate_seconds": elapsed,
    }
    message = (
        f"role={role}; valid=1; cases={len(cases)}; total_cost={total_cost:.3f}; "
        f"invalid_actions={invalid_actions}; candidate_seconds={elapsed:.6f}; score={score:.6f}"
    )
    return score, score, message, metrics


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: evaluator.py /path/to/solution.py", file=sys.stderr)
        raise SystemExit(2)
    bounded, unbounded, detail, metrics = evaluate(sys.argv[1])
    print(detail)
    print(json.dumps(metrics, indent=2))
    print(f"{bounded:.12f} {unbounded:.12f}")
