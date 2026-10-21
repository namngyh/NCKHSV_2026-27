"""Kiểm thử mô hình nền GARCH(1,1) (garch_nen): khôi phục tham số, khớp gói arch, xử lý phiên chuyển hợp đồng.

Chạy: python -m pytest kiem_thu/test_garch_nen.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

GOC = Path(__file__).resolve().parents[1]
for p in (GOC / "ma_nguon", GOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax.numpy as jnp                      # noqa: E402

from de_cuong import garch_nen as GN         # noqa: E402
from de_cuong import phan_phoi as PP         # noqa: E402


def _sinh(n, mu, om, al, be, pp="chuan", sh=(), J=None, seed=1):
    rng = np.random.default_rng(seed)
    z = PP.lay_mau(n, pp, np.asarray(sh, float), rng)
    h = np.empty(n)
    r = np.empty(n)
    h[0] = om / (1 - al - be)
    for t in range(n):
        if t:
            e2 = h[t - 1] if (J is not None and J[t - 1]) else (r[t - 1] - mu) ** 2
            h[t] = om + al * e2 + be * h[t - 1]
        r[t] = mu + np.sqrt(h[t]) * z[t]
    return r, h


def _dl(r, J=None):
    n = len(r)
    J = np.zeros(n, bool) if J is None else J
    r0 = np.where(J, 0.0, r)
    return (jnp.asarray(r0), jnp.asarray(J), jnp.zeros((n, 0)), jnp.asarray(~J), jnp.asarray(np.log(np.var(r[:20], ddof=1))))


@pytest.mark.parametrize("pp,sh", [("chuan", ()), ("student", (6.0,))])
def test_khoi_phuc_tham_so(pp, sh):
    r, _ = _sinh(6000, 0.05, 0.05, 0.08, 0.88, pp, sh)
    m = GN.mle(pp, _dl(r), so_diem=8, rng=np.random.default_rng(0))
    mu, om, al, be = m["psi"][:4]
    assert abs(mu - 0.05) < 0.05 and abs(al - 0.08) < 0.03 and abs(be - 0.88) < 0.05 and abs(om - 0.05) < 0.05
    assert al + be < 1
    assert m["so_gan_tot"] >= 2
    if pp == "student":
        assert 4.0 < m["psi"][4] < 9.5


def test_khop_goi_arch():
    arch = pytest.importorskip("arch")
    r, _ = _sinh(4000, 0.02, 0.04, 0.07, 0.90, seed=7)
    m = GN.mle("chuan", _dl(r), so_diem=8, rng=np.random.default_rng(0))
    kq = arch.arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
    p = kq.params
    assert abs(m["psi"][0] - p["mu"]) < 0.01
    assert abs(m["psi"][2] - p["alpha[1]"]) < 0.02 and abs(m["psi"][3] - p["beta[1]"]) < 0.04
    assert abs(m["ll"] - kq.loglikelihood) < 3.0      # khác nhau do cách khởi tạo phương sai đầu


def test_phien_chuyen_hop_dong_va_du_bao():
    n = 1500
    J = np.zeros(n, bool)
    J[100::22] = True
    r, _ = _sinh(n, 0.0, 0.05, 0.08, 0.88, J=J, seed=3)
    dl = _dl(r, J)
    m = GN.mle("chuan", dl, so_diem=6, rng=np.random.default_rng(0))
    assert np.isfinite(m["ll"]) and m["psi"][2] + m["psi"][3] < 1
    db = GN.du_bao_cam("chuan", m["psi"], dl, 61 + 1000, 61 + 1100, (0.05, 0.01))
    for k in ("lnh", "h", "mu", "ls", "pit"):
        assert len(db[k]) == 100
    assert np.all(db["h"] > 0)
    assert np.all(db["var"][0.01]["mua"] > db["var"][0.05]["mua"] - 1e-9)
    assert np.all(db["es"][0.01]["mua"] >= db["var"][0.01]["mua"] - 1e-9)
