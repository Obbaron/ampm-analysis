"""
Lightweight memory profiling for AMPM pipeline phases.

Wrap suspect stages in ``with phase("name"):`` to print working-set and
commit-charge readings before and after each stage, plus an explicit
callout when the process-lifetime peak grew *during* that stage — which
pinpoints OOM culprits even when the absolute numbers look survivable.

Set ``AMPM_MEMPROF=0`` in the environment to silence all output.

Usage
-----
    from ampm.memprof import phase

    with phase("mask: materialize filtered DataFrame"):
        df = df.filter(keep)
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

enabled = os.environ.get("AMPM_MEMPROF", "0") != "0"

_GIB = 2**30

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as _wt

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", _wt.DWORD),
            ("PageFaultCount", _wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        _GetProcessMemoryInfo = _kernel32.K32GetProcessMemoryInfo
    except AttributeError:
        _GetProcessMemoryInfo = ctypes.WinDLL(
            "psapi", use_last_error=True
        ).GetProcessMemoryInfo
    _GetProcessMemoryInfo.argtypes = [
        _wt.HANDLE,
        ctypes.POINTER(_PMC),
        _wt.DWORD,
    ]
    _GetProcessMemoryInfo.restype = _wt.BOOL
    _GetCurrentProcess = _kernel32.GetCurrentProcess
    _GetCurrentProcess.argtypes = []
    _GetCurrentProcess.restype = _wt.HANDLE

    _warned_failure = False

    def _counters() -> _PMC:
        global _warned_failure
        counter = _PMC()
        counter.cb = ctypes.sizeof(counter)
        ok = _GetProcessMemoryInfo(
            _GetCurrentProcess(), ctypes.byref(counter), counter.cb
        )
        if not ok and not _warned_failure:
            _warned_failure = True
            print(
                f"  [mem] WARNING: GetProcessMemoryInfo failed "
                f"(WinError {ctypes.get_last_error()}); readings will be 0.",
                flush=True,
            )
        return counter

    def rss() -> int:
        """Current working set, bytes."""
        return _counters().WorkingSetSize

    def peak_rss() -> int:
        """Process-lifetime peak working set, bytes."""
        return _counters().PeakWorkingSetSize

    def commit() -> int:
        """Current commit charge (private bytes) — what OOM kills are
        actually judged against on Windows."""
        return _counters().PagefileUsage

    def peak_commit() -> int:
        """Process-lifetime peak commit charge, bytes."""
        return _counters().PeakPagefileUsage

else:
    import resource

    _PAGE = os.sysconf("SC_PAGE_SIZE")

    def rss() -> int:
        """Current resident set size, bytes."""
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE

    def peak_rss() -> int:
        """Process-lifetime peak RSS in bytes."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    commit = rss
    peak_commit = peak_rss


@contextmanager
def phase(name: str):
    """Print memory counters entering and leaving a pipeline stage."""
    if not enabled:
        yield
        return
    r0, p0, c0, pc0 = rss(), peak_rss(), commit(), peak_commit()
    print(
        f"  [mem] >> {name}: rss={r0/_GIB:.2f} commit={c0/_GIB:.2f} "
        f"(peaks {p0/_GIB:.2f}/{pc0/_GIB:.2f}) GiB",
        flush=True,
    )
    try:
        yield
    finally:
        r1, p1, c1, pc1 = rss(), peak_rss(), commit(), peak_commit()
        grew = max(p1 - p0, pc1 - pc0)
        marker = (
            f"   << peak grew +{grew/_GIB:.2f} GiB IN THIS PHASE"
            if grew > 64 * 2**20
            else ""
        )
        print(
            f"  [mem] << {name}: rss={r1/_GIB:.2f} (\u0394{(r1-r0)/_GIB:+.2f}) "
            f"commit={c1/_GIB:.2f} (peaks {p1/_GIB:.2f}/{pc1/_GIB:.2f}) GiB"
            f"{marker}",
            flush=True,
        )
