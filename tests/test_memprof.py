"""
Tests for ``memprof.py`` — lightweight per-phase memory profiling.

These run on the POSIX branch of the module (the Windows branch is guarded by
``sys.platform == "win32"`` and can't execute here). The ``phase`` formatting
and peak-growth detection are tested by monkeypatching the counter functions so
the assertions don't depend on real allocator behavior.
"""

from __future__ import annotations

import sys

import pytest

import ampm.memprof as memprof

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX counter branch only"
)

GIB = memprof._GIB


@posix_only
class TestCounters:
    def test_rss_is_positive_int(self):
        value = memprof.rss()
        assert isinstance(value, int) and value > 0

    def test_peak_rss_is_positive_int(self):
        value = memprof.peak_rss()
        assert isinstance(value, int) and value > 0

    def test_commit_aliases_rss_on_posix(self):
        # On POSIX the module defines commit = rss, peak_commit = peak_rss.
        assert memprof.commit is memprof.rss
        assert memprof.peak_commit is memprof.peak_rss


class TestPhase:
    def test_disabled_is_silent(self, monkeypatch, capsys):
        monkeypatch.setattr(memprof, "enabled", False)
        with memprof.phase("anything"):
            pass
        assert capsys.readouterr().out == ""

    def test_disabled_still_yields(self, monkeypatch):
        monkeypatch.setattr(memprof, "enabled", False)
        ran = False
        with memprof.phase("x"):
            ran = True
        assert ran

    def test_enabled_prints_enter_and_exit_with_name(self, monkeypatch, capsys):
        monkeypatch.setattr(memprof, "enabled", True)
        for name in ("rss", "peak_rss", "commit", "peak_commit"):
            monkeypatch.setattr(memprof, name, lambda: GIB)
        with memprof.phase("my-stage"):
            pass
        out = capsys.readouterr().out
        assert ">>" in out and "<<" in out
        assert out.count("my-stage") == 2

    def test_peak_growth_marker_appears(self, monkeypatch, capsys):
        monkeypatch.setattr(memprof, "enabled", True)
        monkeypatch.setattr(memprof, "rss", lambda: GIB)
        monkeypatch.setattr(memprof, "commit", lambda: GIB)
        monkeypatch.setattr(memprof, "peak_commit", lambda: GIB)
        # peak_rss read twice: before then after; 1 GiB growth > 64 MiB threshold.
        seq = iter([GIB, 2 * GIB])
        monkeypatch.setattr(memprof, "peak_rss", lambda: next(seq))
        with memprof.phase("growing"):
            pass
        assert "peak grew" in capsys.readouterr().out

    def test_no_marker_when_peak_flat(self, monkeypatch, capsys):
        monkeypatch.setattr(memprof, "enabled", True)
        for name in ("rss", "peak_rss", "commit", "peak_commit"):
            monkeypatch.setattr(memprof, name, lambda: GIB)
        with memprof.phase("flat"):
            pass
        assert "peak grew" not in capsys.readouterr().out

    def test_marker_below_threshold_suppressed(self, monkeypatch, capsys):
        # 32 MiB growth is under the 64 MiB callout threshold.
        monkeypatch.setattr(memprof, "enabled", True)
        monkeypatch.setattr(memprof, "rss", lambda: GIB)
        monkeypatch.setattr(memprof, "commit", lambda: GIB)
        monkeypatch.setattr(memprof, "peak_commit", lambda: GIB)
        seq = iter([GIB, GIB + 32 * 2**20])
        monkeypatch.setattr(memprof, "peak_rss", lambda: next(seq))
        with memprof.phase("small-bump"):
            pass
        assert "peak grew" not in capsys.readouterr().out

    def test_exception_inside_phase_still_prints_exit(self, monkeypatch, capsys):
        monkeypatch.setattr(memprof, "enabled", True)
        for name in ("rss", "peak_rss", "commit", "peak_commit"):
            monkeypatch.setattr(memprof, name, lambda: GIB)
        with pytest.raises(RuntimeError):
            with memprof.phase("boom"):
                raise RuntimeError("boom")
        out = capsys.readouterr().out
        assert "<<" in out  # finally-block exit line still ran
