"""Tests for the measurement helpers."""

import numpy as np
import pytest

from spamedge.bench import latency_stats, size_on_disk, summarize_runs, time_single_calls


def close(got, expected, tol=1e-9):
    assert abs(got - expected) <= tol, f"ECHEC : obtenu {got}, attendu {expected}"


def test_latency_stats_by_hand():
    # 1 ms, 2 ms, ..., 100 ms given in ns. With linear interpolation the q-th
    # percentile sits at position 99 * q in the sorted list (0-based):
    #   p50: 49.5  -> between 50 and 51 ms -> 50.5
    #   p95: 94.05 -> 95 + 0.05 * (96 - 95) -> 95.05
    #   p99: 98.01 -> 99 + 0.01 * (100 - 99) -> 99.01
    s = latency_stats(np.arange(1, 101) * 1_000_000)
    close(s["p50_ms"], 50.5)
    close(s["p95_ms"], 95.05)
    close(s["p99_ms"], 99.01)
    close(s["mean_ms"], 50.5)
    assert s["n"] == 100, f"ECHEC : obtenu {s['n']}, attendu 100"


def test_timing_skips_warmup_and_cycles_the_texts():
    seen = []
    durations = time_single_calls(seen.append, ["a", "b", "c"], warmup=2, timed=4)
    # 2 warm-up calls (a, b) then 4 timed ones (a, b, c, a).
    expected = ["a", "b", "a", "b", "c", "a"]
    assert seen == expected, f"ECHEC : obtenu {seen}, attendu {expected}"
    assert len(durations) == 4 and np.all(durations >= 0), f"ECHEC : durées {durations}"


def test_summarize_runs_by_hand():
    # p50 of three runs: 2, 4, 6 ms -> mean 4; sample std = sqrt(((2-4)^2 + 0 + (6-4)^2) / 2) = 2.
    runs = [{"p50_ms": 2.0}, {"p50_ms": 4.0}, {"p50_ms": 6.0}]
    s = summarize_runs(runs, keys=("p50_ms",))["p50_ms"]
    close(s["mean"], 4.0)
    close(s["std"], 2.0)


def test_size_on_disk_adds_up_a_directory(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 32)
    expected = 10 + 32
    got = size_on_disk(tmp_path)
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"


def test_size_on_disk_of_a_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        size_on_disk(tmp_path / "nothing")
