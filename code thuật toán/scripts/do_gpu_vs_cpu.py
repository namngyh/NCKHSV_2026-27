"""Do TRUC TIEP CPU vs GPU cho bo loc tien MS-GARCH-X.

Script TU CHUA (chi can torch), de chay duoc trong mot moi truong ao chi cai
torch ban CUDA ma khong dung gi vao moi truong dang chay pipeline.

Phan than vong lap duoc sao y nguyen tu
`msgarchx/models/ms_garch.py::ms_garch_forward_filter` (ban da toi uu): cung
cong thuc, cung so phep toan moi buoc. Muc dich khong phai lay ket qua thong
ke ma la do CHI PHI DIEU PHOI, nen tham so va du lieu deu la ngau nhien.

Do ca ban BATCH (B bo tham so doc lap trong mot vong lap) vi day moi la tinh
huong GPU co co hoi: chi phi kernel launch duoc chia deu cho B bo tham so.

Chay:
    .venv_gpu_test/Scripts/python.exe scripts/do_gpu_vs_cpu.py
"""
from __future__ import annotations

import math
import time

import torch

_LOG_PI = math.log(math.pi)
T_DEFAULT = 1735          # bang T tap huan luyen VN30F1M


def filter_batched(mu, omega, alpha, beta, gamma, P, nu, r, X, floor=1e-8):
    """mu,omega,alpha,beta: (B,2)  nu: (B,1)  gamma: (B,2,k)  P: (B,2,2)."""
    T = r.shape[0]
    B = mu.shape[0]
    p12, p21 = P[:, 0, 1], P[:, 1, 0]
    pi1 = (p21 / (p12 + p21)).detach()
    xi = torch.stack([pi1, 1.0 - pi1], dim=1)
    h_prev = torch.var(r).detach().expand(B, 2).clone()

    gX = torch.einsum("bsk,tk->tbs", gamma, X)
    const = (torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0)
             - 0.5 * torch.log(nu - 2.0) - 0.5 * _LOG_PI)
    half_nu1 = (nu + 1.0) / 2.0
    nu_m2 = nu - 2.0
    mu2 = mu * mu
    Pt = P.transpose(1, 2).contiguous()

    bar_mu = (xi * mu).sum(1)
    bar_h = (xi * (h_prev + mu2)).sum(1) - bar_mu * bar_mu
    bar_eps = r[0] - bar_mu
    loglik = torch.zeros(B, dtype=r.dtype, device=r.device)

    for t in range(1, T):
        xi_pred = torch.bmm(Pt, xi.unsqueeze(2)).squeeze(2)
        h_t = torch.clamp(omega + alpha * (bar_eps * bar_eps).unsqueeze(1)
                          + beta * bar_h.unsqueeze(1) + gX[t], min=floor)
        z = (r[t] - mu) / torch.sqrt(h_t)
        log_f = const - half_nu1 * torch.log1p(z * z / nu_m2) - 0.5 * torch.log(h_t)
        log_joint = torch.log(xi_pred) + log_f
        ll_t = torch.logsumexp(log_joint, dim=1)
        xi = torch.exp(log_joint - ll_t.unsqueeze(1))
        bar_mu = (xi * mu).sum(1)
        bar_h = (xi * (h_t + mu2)).sum(1) - bar_mu * bar_mu
        bar_eps = r[t] - bar_mu
        loglik = loglik + ll_t
    return loglik


def make(B, k, device, dtype):
    g = torch.Generator().manual_seed(0)
    def rnd(*shape):
        return torch.rand(*shape, generator=g).to(device=device, dtype=dtype)
    return dict(
        mu=(rnd(B, 2) * 0.1).requires_grad_(True),
        omega=(0.1 + rnd(B, 2)).requires_grad_(True),
        alpha=(0.05 + rnd(B, 2) * 0.1).requires_grad_(True),
        beta=(0.6 + rnd(B, 2) * 0.2).requires_grad_(True),
        gamma=(rnd(B, 2, k) * 0.1).requires_grad_(True),
        P=torch.stack([torch.stack([torch.full((B,), 0.95), torch.full((B,), 0.05)], 1),
                       torch.stack([torch.full((B,), 0.05), torch.full((B,), 0.95)], 1)], 1
                      ).to(device=device, dtype=dtype).requires_grad_(True),
        nu=(4.0 + rnd(B, 1)).requires_grad_(True),
    )


def bench(device, dtype, B, T=T_DEFAULT, k=2, n=3, backward=True):
    g = torch.Generator().manual_seed(1)
    r = (torch.randn(T, generator=g) * 1.5).to(device=device, dtype=dtype)
    X = torch.randn(T, k, generator=g).to(device=device, dtype=dtype)
    p = make(B, k, device, dtype)

    def once():
        ps = {kk: v.detach().clone().requires_grad_(True) for kk, v in p.items()}
        ll = filter_batched(**ps, r=r, X=X)
        if backward:
            (-ll.sum()).backward()
        if device == "cuda":
            torch.cuda.synchronize()

    once()                                   # lam nong
    t0 = time.perf_counter()
    for _ in range(n):
        once()
    return (time.perf_counter() - t0) / n


def main() -> None:
    torch.set_default_dtype(torch.float64)
    print(f"torch {torch.__version__} | build CUDA {torch.version.cuda} | "
          f"cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU threads: {torch.get_num_threads()} | T={T_DEFAULT}, k=2\n")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dtype, ten in [(torch.float64, "float64 (dung trong du an)"),
                        (torch.float32, "float32 (chi de tham chieu)")]:
        print(f"=== {ten} — forward+backward ===")
        print(f"{'B':>5} | {'CPU (s)':>10} | {'GPU (s)':>10} | {'CPU s/bo':>10} | "
              f"{'GPU s/bo':>10} | ket luan")
        for B in (1, 16, 64, 256):
            kq = {}
            for dev in devices:
                try:
                    kq[dev] = bench(dev, dtype, B)
                except RuntimeError as e:
                    kq[dev] = float("nan")
                    print(f"   (B={B}, {dev}: {str(e)[:60]})")
            c, gp = kq.get("cpu", float("nan")), kq.get("cuda", float("nan"))
            if gp == gp:
                ratio = c / gp
                kl = f"GPU nhanh hon {ratio:.2f}x" if ratio > 1 else f"GPU CHAM hon {1/ratio:.2f}x"
            else:
                kl = "khong co GPU"
            print(f"{B:>5} | {c:>10.3f} | {gp:>10.3f} | {c/B:>10.4f} | {gp/B:>10.4f} | {kl}")
        print()


if __name__ == "__main__":
    main()
