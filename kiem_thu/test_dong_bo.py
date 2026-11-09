"""Kiểm thử đồng bộ git giữa các máy (de_cuong.dong_bo).

LÝ THUYẾT: Tái hiện sự cố 09/11/2026: máy chủ trì làm thay đơn vị của máy phụ vắng mặt, máy phụ quay lại làm cùng đơn
vị, hai máy ghi cùng một tệp với nội dung khác nhau; `git pull --rebase` dừng giữa chừng và trước đây máy phụ kẹt
vĩnh viễn. Kiểm thử dùng một remote trần và hai bản sao trong thư mục tạm.
THƯ VIỆN: subprocess (git), pytest (tmp_path, monkeypatch).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GOC = Path(__file__).resolve().parents[1]
for p in (GOC / "ma_nguon", GOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from de_cuong import dong_bo as DB      # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="can git")


def _g(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def hai_may(tmp_path):
    _g(tmp_path, "init", "-q", "--bare", "-b", "main", "remote.git")
    for m in ("A", "B"):
        _g(tmp_path, "clone", "-q", str(tmp_path / "remote.git"), m)
        for k, v in (("user.name", m), ("user.email", f"{m}@x"), ("core.autocrlf", "false")):
            _g(tmp_path / m, "config", k, v)
    (tmp_path / "A" / "ket_qua").mkdir()
    (tmp_path / "A" / "ket_qua" / "goc.txt").write_text("goc\n")
    _g(tmp_path / "A", "add", "-A")
    _g(tmp_path / "A", "commit", "-qm", "goc")
    _g(tmp_path / "A", "push", "-q", "origin", "main")
    _g(tmp_path / "B", "pull", "-q")
    return tmp_path


def _ghi(may_dir, ten, noi_dung):
    p = may_dir / "ket_qua" / ten
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(noi_dung)


def test_hai_may_cung_ghi_mot_tep_khong_ket(hai_may, monkeypatch):
    A, B = hai_may / "A", hai_may / "B"
    for m in (A, B):
        _ghi(m, "_doan/doan_000.pkl", b"\x80" + os.urandom(32))     # cùng đơn vị, khác nội dung
        _ghi(m, f"rieng_{m.name}.pkl", b"\x80" + os.urandom(8))
    monkeypatch.setattr(DB, "GOC", A)
    assert DB.day("A", A / "ket_qua", log=lambda s: None)
    monkeypatch.setattr(DB, "GOC", B)
    assert DB.day("B", B / "ket_qua", log=lambda s: None)
    assert _g(B, "rev-list", "--count", "@{u}..HEAD").stdout.strip() == "0"
    monkeypatch.setattr(DB, "GOC", A)
    assert DB.keo(log=lambda s: None)
    assert (A / "ket_qua/_doan/doan_000.pkl").read_bytes() == (B / "ket_qua/_doan/doan_000.pkl").read_bytes()
    assert (A / "ket_qua/rieng_B.pkl").exists() and (B / "ket_qua/rieng_A.pkl").exists()


def test_go_trang_thai_rebase_do_dang(hai_may, monkeypatch):
    A, B = hai_may / "A", hai_may / "B"
    for m, nd in ((A, b"\x80A"), (B, b"\x80B")):
        _ghi(m, "x.pkl", nd)
        _g(m, "add", "-A")
        _g(m, "commit", "-qm", m.name)
    _g(A, "push", "-q")
    assert _g(B, "pull", "--rebase").returncode != 0                  # B kẹt giữa rebase như sự cố 17:24
    monkeypatch.setattr(DB, "GOC", B)
    assert DB.keo(log=lambda s: None)
    assert "rebase" not in _g(B, "status").stdout.splitlines()[0]
    assert DB.day("B", B / "ket_qua", log=lambda s: None)              # đẩy được commit gộp dù không có tệp mới
    assert _g(B, "rev-list", "--count", "@{u}..HEAD").stdout.strip() == "0"
    assert (B / "ket_qua/x.pkl").read_bytes() == b"\x80A"              # tệp trùng giữ bản đã có trên GitHub
