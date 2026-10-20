"""Bước 0 của thiết kế: đo tốc độ NUTS (NumPyro/JAX) và kiểm tra hàm log-hợp lý JAX khớp bản numba.

Chạy:  python scripts/do_toc_do_nuts.py [--warmup 300 --samples 300 --chains 1]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpyro                                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--samples", type=int, default=300)
ap.add_argument("--chains", type=int, default=1)
a = ap.parse_args()
numpyro.set_host_device_count(a.chains)

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
from numpyro.infer import MCMC, NUTS                             # noqa: E402

from dc_chay import lay_mau, nap, tao_Z                          # noqa: E402
from de_cuong import jax_mo_hinh as jm                           # noqa: E402
from de_cuong.du_lieu import BO_QUA                              # noqa: E402
from de_cuong.mo_hinh import Cau_hinh, diem_khoi_tao, uoc_luong  # noqa: E402

d, blk = nap()
s1 = blk["XT"][1]
f, r, J, lh0 = lay_mau(d, BO_QUA, s1)
cfg = Cau_hinh("skewt", (0, 1))
Z = tao_Z(cfg, f)
sl = slice(BO_QUA, s1)
rr, JJ, ZZ = r[sl], J[sl], np.ascontiguousarray(Z[sl])
print(f"cửa sổ: {len(rr)} phiên, {int((~JJ).sum())} có lợi suất")

p, nll, _, _ = uoc_luong(cfg, rr, JJ, ZZ, lh0, diem_khoi_tao(cfg, f["r"][sl], 6, np.random.default_rng(1)))
mu, om, ka, th, be, b, sp = cfg.tach(p)
ll_jax = float(jm.log_hop_ly(mu, om, ka, th, be, jnp.array(b), sp[0], sp[1], jnp.array(rr), jnp.array(JJ),
                             jnp.array(ZZ), lh0))
print(f"log-hợp-lý numba {-nll:.10f}  JAX {ll_jax:.10f}  chênh {abs(-nll - ll_jax):.2e}")

g = jax.jit(jax.value_and_grad(lambda q: jm.log_hop_ly(q[0], q[1], q[2], q[3], q[4], q[5:10], q[10], q[11],
                                                       jnp.array(rr), jnp.array(JJ), jnp.array(ZZ), lh0)))
q0 = jnp.array(p)
g(q0)[0].block_until_ready()
t = time.time()
for _ in range(200):
    g(q0)[0].block_until_ready()
print(f"JAX value+grad: {(time.time()-t)/200*1000:.2f} ms/lần")

kernel = NUTS(jm.mo_hinh_bayes, dense_mass=True, target_accept_prob=0.9)
mcmc = MCMC(kernel, num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
            chain_method="parallel" if a.chains > 1 else "sequential", progress_bar=False)
t = time.time()
mcmc.run(jax.random.PRNGKey(0), jnp.array(rr), jnp.array(JJ), jnp.array(ZZ), lh0, 2,
         extra_fields=("num_steps", "diverging"))
mcmc.get_samples()["mu"].block_until_ready()
dt = time.time() - t
ex = mcmc.get_extra_fields()
print(f"NUTS {a.chains} chuỗi, {a.warmup}+{a.samples} vòng: {dt:.0f} giây; số bước leapfrog trung bình "
      f"{np.asarray(ex['num_steps']).mean():.1f}; phân kỳ {int(np.asarray(ex['diverging']).sum())}")
print("Ước tính 4 chuỗi x (1000 + 1000) vòng, chạy tuần tự:",
      f"{dt / a.chains / (a.warmup + a.samples) * 2000 * 4 / 3600:.2f} giờ cho một mốc")
