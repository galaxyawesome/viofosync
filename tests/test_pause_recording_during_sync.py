"""PAUSE_RECORDING_DURING_SYNC: stop recording while a cycle touches the
camera's SD card and always start it again afterwards.

The camera, its record flag and the cycle's card work are faked; the tests
check the order of events and that every exit path restores recording.
"""
from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from web.db import Database
from web.services import sync_worker as swm
from web.services.sync_worker import SyncWorker


class _Hub:
    def __init__(self):
        self.events = []

    async def broadcast(self, event):
        self.events.append(event)


class _Cam:
    """Fake camera: record flag plus a log of everything done to it."""

    def __init__(self, recording=1, stop_ok=True, start_ok=True):
        self.rec = recording
        self.stop_ok = stop_ok
        self.start_ok = start_ok
        self.log = []

    def record_state(self, address):
        self.log.append(("state", self.rec))
        return self.rec

    def set_recording(self, address, on, attempts=3):
        ok = self.start_ok if on else self.stop_ok
        self.log.append(("start" if on else "stop", ok))
        if ok and self.rec is not None:
            self.rec = 1 if on else 0
        return ok


def _worker(tmp_path, monkeypatch, cam, enabled=True, online_result=True,
            online_exc=None):
    monkeypatch.setattr(swm, "PAUSE_RECORDING_DURING_SYNC", enabled)
    monkeypatch.setattr(swm, "RECORD_PAUSE_SETTLE_S", 0)
    monkeypatch.setattr(swm, "RECORD_RESUME_QUIET_S", 0)
    monkeypatch.setattr(swm.control, "record_state", cam.record_state)
    monkeypatch.setattr(swm.control, "set_recording", cam.set_recording)
    sw = SyncWorker.__new__(SyncWorker)
    sw.db = Database(str(tmp_path / "v.db"))
    sw.hub = _Hub()
    sw._paused = threading.Event()
    sw._stop = asyncio.Event()
    sw._online = None
    sw._last_rec_resume = None
    sw._active_address = None
    sw._active_source = None

    async def _noop():
        return None

    async def _writable():
        return True

    async def _select():
        return "1.2.3.4", "primary"

    async def _online():
        cam.log.append(("card work", cam.rec))
        if online_exc:
            raise online_exc
        return online_result

    sw._emit_disk_pct = _noop
    sw._check_recordings_writable = _writable
    sw._select_active_address = _select
    sw._cycle_online = _online
    return sw


async def test_off_by_default_no_record_commands(tmp_path, monkeypatch):
    cam = _Cam(recording=1)
    sw = _worker(tmp_path, monkeypatch, cam, enabled=False)
    assert await sw._cycle() is True
    assert cam.log == [("card work", 1)]


async def test_stops_before_card_work_and_restarts_after(tmp_path, monkeypatch):
    cam = _Cam(recording=1)
    sw = _worker(tmp_path, monkeypatch, cam)
    assert await sw._cycle() is True
    assert cam.log == [("state", 1), ("stop", True), ("card work", 0),
                       ("start", True)]
    assert cam.rec == 1
    assert sw.db.kv_get(swm._REC_PAUSED_KV_KEY) == "0"


async def test_not_recording_left_alone(tmp_path, monkeypatch):
    cam = _Cam(recording=0)          # e.g. user stopped it on purpose
    sw = _worker(tmp_path, monkeypatch, cam)
    assert await sw._cycle() is True
    assert cam.log == [("state", 0), ("card work", 0)]
    assert cam.rec == 0


async def test_unknown_state_skips_cycle(tmp_path, monkeypatch):
    cam = _Cam(recording=None)
    sw = _worker(tmp_path, monkeypatch, cam)
    assert await sw._cycle() is False
    assert ("card work", None) not in cam.log


async def test_stop_refused_skips_cycle_and_restores(tmp_path, monkeypatch):
    cam = _Cam(recording=1, stop_ok=False)
    sw = _worker(tmp_path, monkeypatch, cam)
    assert await sw._cycle() is False
    assert [e[0] for e in cam.log] == ["state", "stop", "start"]
    assert cam.rec == 1


async def test_restarts_even_when_card_work_raises(tmp_path, monkeypatch):
    cam = _Cam(recording=1)
    sw = _worker(tmp_path, monkeypatch, cam, online_exc=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await sw._cycle()
    assert cam.log[-1] == ("start", True)
    assert cam.rec == 1


async def test_failed_restart_is_retried_next_cycle(tmp_path, monkeypatch,
                                                    caplog):
    cam = _Cam(recording=1, start_ok=False)
    sw = _worker(tmp_path, monkeypatch, cam)
    with caplog.at_level(logging.WARNING, logger="viofosync.sync_worker"):
        await sw._cycle()
    assert sw.db.kv_get(swm._REC_PAUSED_KV_KEY) == "1"
    assert any("could not restart recording" in r.getMessage()
               for r in caplog.records)
    # camera comes back: the next cycle restarts recording FIRST
    cam.start_ok = True
    cam.log.clear()
    await sw._cycle()
    assert cam.log[0] == ("start", True)
    assert sw.db.kv_get(swm._REC_PAUSED_KV_KEY) == "0"
    assert cam.rec == 1


async def test_pause_left_by_previous_run_repaired(tmp_path, monkeypatch):
    cam = _Cam(recording=0)          # stopped by a run that crashed
    sw = _worker(tmp_path, monkeypatch, cam)
    sw.db.kv_set(swm._REC_PAUSED_KV_KEY, "1")
    await sw._cycle()
    assert cam.log[0] == ("start", True)
    # then the normal pause/resume around this cycle's card work
    assert cam.log[1:] == [("state", 1), ("stop", True), ("card work", 0),
                           ("start", True)]


async def test_quiet_period_after_restart(tmp_path, monkeypatch):
    cam = _Cam(recording=1)
    sw = _worker(tmp_path, monkeypatch, cam)
    monkeypatch.setattr(swm, "RECORD_RESUME_QUIET_S", 0.3)
    slept = []
    real_sleep = asyncio.sleep

    async def _sleep(s):
        slept.append(s)
        await real_sleep(0)

    monkeypatch.setattr(swm.asyncio, "sleep", _sleep)
    await sw._cycle()                 # first cycle: no wait
    assert not any(s > 0.1 for s in slept)
    await sw._cycle()                 # right after a restart: waits
    assert any(0.1 < s <= 0.3 for s in slept)


async def test_paused_worker_never_touches_recording(tmp_path, monkeypatch):
    cam = _Cam(recording=1)
    sw = _worker(tmp_path, monkeypatch, cam)
    sw._paused.set()
    assert await sw._cycle() is False
    assert cam.log == []


def test_set_recording_confirms_via_status(monkeypatch):
    from viofosync_lib import _control as c
    sent, states = [], iter([1, 0])
    monkeypatch.setattr(c, "VERIFY_SETTLE", 0)
    monkeypatch.setattr(c, "RECORD_SETTLE", 0)
    monkeypatch.setattr(c, "_send", lambda a, cmd, par=None, s=None:
                        sent.append((cmd, par)) or {"Status": "0"})
    monkeypatch.setattr(c, "record_state", lambda a: next(states))
    assert c.set_recording("1.2.3.4", False) is True
    assert sent == [(2001, 0)]


def test_set_recording_gives_up_without_raising(monkeypatch):
    from viofosync_lib import _control as c
    monkeypatch.setattr(c, "VERIFY_SETTLE", 0)
    monkeypatch.setattr(c, "RECORD_SETTLE", 0)

    def _down(*a, **k):
        raise c.CameraUnreachable("gone")

    monkeypatch.setattr(c, "_send", _down)
    assert c.set_recording("1.2.3.4", True) is False
