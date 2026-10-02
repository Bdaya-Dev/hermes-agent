"""cgroup-aware memory sampling (upstream #118117).

Inside a container ``/proc/meminfo`` is the HOST view (no lxcfs), so the
kanban memory guard used to size a container's budget from the host. These
tests pin the contract:

* a FINITE ``memory.max`` is the real ceiling — both the total the derived cap
  uses and the headroom the pressure classifier sees;
* ``memory.max == "max"`` means the cgroup cannot OOM this process, so the
  sample flags it and the dispatcher fails open;
* a missing/unreadable cgroup tree keeps the host view (fail open, non-Linux
  and read errors unchanged).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway import lifecycle_ledger
from hermes_cli import kanban_db_dispatch as kbd

GIB = 1024 * 1024  # KiB per GiB
MIB = 1024


def _write_cgroup(root: Path, max_bytes, current_bytes=None) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.max").write_text(
        "max" if max_bytes is None else str(int(max_bytes)), encoding="utf-8"
    )
    if current_bytes is not None:
        (root / "memory.current").write_text(str(int(current_bytes)), encoding="utf-8")
    return str(root)


@pytest.fixture
def fake_proc(monkeypatch):
    """Host /proc/meminfo: 32 GiB box with 3.5 GiB free (our production shape)."""
    host_total, host_available = 32099 * MIB, 3576 * MIB
    monkeypatch.setattr(
        lifecycle_ledger,
        "_proc_fields",
        lambda path, wanted: {
            "rss_kib": 64 * MIB,
            "mem_total_kib": host_total,
            "mem_available_kib": host_available,
        } if path == "/proc/self/status" else (
            {
                "mem_total_kib": host_total,
                "mem_available_kib": host_available,
                "SwapTotal": 0,
                "SwapFree": 0,
            } if path == "/proc/meminfo" else {}
        ),
    )
    return host_total, host_available


# ---------------------------------------------------------------------------
# finite cgroup limit — the small-mem_limit overcommit case stays protected
# ---------------------------------------------------------------------------


def test_finite_cgroup_limit_caps_reported_total(tmp_path, fake_proc):
    """1 GiB container on a 32 GiB host: the effective total must be 1 GiB, so
    the derived cap floors at 2 workers instead of allowing the host's 8."""
    root = _write_cgroup(tmp_path / "cg", 1 * GIB * 1024, current_bytes=256 * MIB * 1024)
    sample = lifecycle_ledger.sample_memory(root)
    assert sample["cgroup_max_kib"] == 1 * GIB
    assert sample["mem_total_kib"] == 1 * GIB
    # Headroom is cgroup-internal, not the host's 3.5 GiB of free pages.
    assert sample["mem_available_kib"] == 768 * MIB
    assert kbd.derive_default_max_in_progress(sample) == 2


def test_finite_cgroup_limit_near_ceiling_is_elevated(tmp_path, fake_proc):
    from gateway.memory_status import classify_pressure

    # 2 GiB limit, 1.8 GiB used -> 0.2 GiB headroom (10% of the limit).
    root = _write_cgroup(tmp_path / "cg", 2 * GIB * 1024, current_bytes=int(1.8 * GIB * 1024))
    sample = lifecycle_ledger.sample_memory(root)
    assert classify_pressure(sample["mem_available_kib"], sample["mem_total_kib"]) in (
        "elevated",
        "critical",
    )


def test_finite_cgroup_limit_with_low_usage_is_ok(tmp_path, fake_proc):
    from gateway.memory_status import classify_pressure

    # 4 GiB limit, 0.5 GiB used -> 3.5 GiB headroom, even though the HOST is at
    # 11% available and would classify 'elevated' on its own.
    root = _write_cgroup(tmp_path / "cg", 4 * GIB * 1024, current_bytes=512 * MIB * 1024)
    sample = lifecycle_ledger.sample_memory(root)
    # 3.5 GiB of cgroup headroom, but only 3.5 GiB (3576 MiB) of HOST headroom
    # exists and both are visible to this process: the sample takes the smaller.
    assert sample["mem_available_kib"] == 3576 * MIB
    assert sample["mem_total_kib"] == 4 * GIB
    assert classify_pressure(sample["mem_available_kib"], sample["mem_total_kib"]) == "ok"
    assert kbd._memory_pressure_level(sample) == "ok"


def test_finite_cgroup_limit_exhausted_is_critical(tmp_path, fake_proc):
    root = _write_cgroup(tmp_path / "cg", 1 * GIB * 1024, current_bytes=1 * GIB * 1024)
    sample = lifecycle_ledger.sample_memory(root)
    assert sample["mem_available_kib"] == 0
    assert kbd._memory_pressure_level(sample) == "critical"


# ---------------------------------------------------------------------------
# memory.max == "max" — the production regression
# ---------------------------------------------------------------------------


def test_unlimited_cgroup_flags_unconstrained_and_keeps_host_numbers(tmp_path, fake_proc):
    host_total, host_available = fake_proc
    root = _write_cgroup(tmp_path / "cg", None, current_bytes=1.4 * GIB * 1024)
    sample = lifecycle_ledger.sample_memory(root)
    assert sample["cgroup_unconstrained"] is True
    assert "cgroup_max_kib" not in sample
    # Host numbers stay visible for the dashboard; only classification fails open.
    assert sample["mem_total_kib"] == host_total
    assert sample["mem_available_kib"] == host_available


def test_unlimited_cgroup_fails_open_instead_of_inheriting_host_pressure(tmp_path, fake_proc):
    """THE regression: host MemAvailable/MemTotal = 0.111 classifies 'elevated'
    on its own; an unlimited cgroup must not let that throttle the fleet."""
    from gateway.memory_status import classify_pressure

    root = _write_cgroup(tmp_path / "cg", None, current_bytes=1.4 * GIB * 1024)
    sample = lifecycle_ledger.sample_memory(root)
    assert classify_pressure(
        sample["mem_available_kib"], sample["mem_total_kib"]
    ) == "elevated"  # the host verdict the guard used to inherit
    assert kbd._memory_pressure_level(sample) == "unknown"  # the fix


def test_unlimited_cgroup_dispatch_spawns_more_than_one(
    tmp_path, fake_proc, monkeypatch,
):
    """End-to-end through dispatch_once: the elevated clamp must not fire.

    This is the shape that has clamped the fleet to one worker per minute —
    four ready cards and 24 idle slots must yield MORE than one spawn.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    import hermes_cli.profiles as profmod

    # 'alice' is a real profile in this isolated home (the conftest helper of the
    # same name only lives under tests/hermes_cli, and this is a gateway test).
    monkeypatch.setattr(profmod, "profile_exists", lambda name: name == "alice")

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    root = _write_cgroup(tmp_path / "cg", None, current_bytes=1.4 * GIB * 1024)
    sample = lifecycle_ledger.sample_memory(root)
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: sample)
    spawned = []

    def fake_spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 42

    with kbc.connect() as conn:
        for title in ("a", "b", "c", "d"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn, max_in_progress=8)

    assert len(spawned) > 1
    assert res.memory_pressure is None


# ---------------------------------------------------------------------------
# fail open: no cgroup tree, cgroup v1, malformed, non-Linux
# ---------------------------------------------------------------------------


def test_missing_cgroup_tree_keeps_host_view(tmp_path, fake_proc):
    host_total, host_available = fake_proc
    sample = lifecycle_ledger.sample_memory(str(tmp_path / "nope"))
    assert sample["mem_total_kib"] == host_total
    assert sample["mem_available_kib"] == host_available
    assert "cgroup_unconstrained" not in sample
    assert kbd._memory_pressure_level(sample) == "elevated"  # unchanged upstream behaviour


def test_malformed_memory_max_fails_open(tmp_path, fake_proc):
    host_total, _ = fake_proc
    root = tmp_path / "cg"
    root.mkdir()
    (root / "memory.max").write_text("not-a-number", encoding="utf-8")
    sample = lifecycle_ledger.sample_memory(str(root))
    assert sample["mem_total_kib"] == host_total
    assert "cgroup_max_kib" not in sample


def test_unreadable_memory_max_fails_open(tmp_path, fake_proc, monkeypatch):
    host_total, _ = fake_proc

    def boom(*_a, **_k):
        raise PermissionError("denied")

    monkeypatch.setattr(lifecycle_ledger, "open", boom, raising=False)
    sample = lifecycle_ledger.sample_memory(str(tmp_path / "cg"))
    assert sample["mem_total_kib"] == host_total


def test_non_linux_returns_empty_sample(monkeypatch):
    """No /proc and no cgroup tree: an empty sample, so the guard imposes no cap
    and no pressure restriction — the pre-existing fail-open contract."""
    monkeypatch.setattr(lifecycle_ledger, "_proc_fields", lambda path, wanted: {})

    def boom(*_a, **_k):
        raise OSError("no /proc on this platform")

    monkeypatch.setattr(lifecycle_ledger, "open", boom, raising=False)
    sample = lifecycle_ledger.sample_memory("/nonexistent-cgroup")
    assert sample.get("mem_total_kib") is None
    assert kbd.derive_default_max_in_progress(sample) is None
    assert kbd._memory_pressure_level(sample) == "unknown"
