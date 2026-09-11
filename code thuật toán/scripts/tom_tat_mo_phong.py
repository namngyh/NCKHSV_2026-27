"""Sinh bao cao tong ket hai nghien cuu khoi phuc tham so (PR-A va PR-B).

Doc tu cac file da gop (`results/07_simulation_S*.pt`,
`results/mo_phong_egarch/E*.pt`) va tu checkpoint tung lan lap, roi ghi ra
`results/tom_tat_mo_phong.txt`.

Chay lai bat cu luc nao -- ke ca khi PR-B moi xong mot phan; phan chua co se
duoc ghi ro la "chua chay" thay vi bo trong.

    python scripts/tom_tat_mo_phong.py
"""
from __future__ import annotations

import glob
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "tom_tat_mo_phong.txt"

PP = ["mle", "vb_mean_field", "vb_full_rank"]
TEN_PP = {"mle": "MLE", "vb_mean_field": "VB-MF", "vb_full_rank": "VB-FR"}

# Gia tri that cua ba kich ban PR-A -- doc THANG tu scenarios.py thay vi chep
# lai vao day. Chep lai la tao ra ban sao thu hai cua cung mot su that; neu ai
# do doi tham so kich ban, bao cao nay se im lang bao sai.
from msgarchx.simulation.run_simulation import _named_params  # noqa: E402
from msgarchx.simulation.scenarios import default_scenarios  # noqa: E402

_SCN = default_scenarios(T_default=2000, B_default=100)
# Chi giu cac tham so co trong bang, theo dung thu tu muon in.
_KHOA = ["omega1", "omega2", "alpha1", "alpha2", "beta1", "beta2", "p11", "p22", "nu"]
THAT_MS = {}
for _ma in ("S1", "S2", "S3"):
    _ten = _named_params(_SCN[_ma].params)
    # P trong scenarios nam o truong P0 rieng, khong phai params.P -- lay dung nguon.
    _ten["p11"] = float(_SCN[_ma].P0[0, 0])
    _ten["p22"] = float(_SCN[_ma].P0[1, 1])
    THAT_MS[_ma] = {k: _ten[k] for k in _KHOA}

MO_TA_MS = {
    "S1": "Moc nen -- hai trang thai phan biet ro",
    "S2": "Nhan dang yeu -- hai trang thai gan nhau",
    "S3": "Duoi rat day -- nu_0 = 4, sat bien ton tai phuong sai",
}
MO_TA_E = {
    "E1": "Don bay ro -- theta_0 = -0,10",
    "E2": "KHONG don bay -- theta_0 = 0  (kich ban quan trong nhat)",
    "E3": "Duoi day -- nu_0 = 4, co don bay",
}

_dong: list[str] = []


def p(s: str = "") -> None:
    _dong.append(s)


def _cov_tb(blk: dict) -> float:
    v = [float(x) for x in blk.get("coverage", {}).values()]
    return sum(v) / len(v) if v else float("nan")


def _dem_checkpoint(mau: str) -> int:
    return len(glob.glob(str(RESULTS / "parts" / "sim" / mau)))


def phan_pr_a() -> None:
    p("=" * 78)
    p("PHAN 1 -- PR-A: MS-GARCH-X Student-t, hai trang thai (15 tham so)")
    p("=" * 78)
    p("  Thiet ke: 3 kich ban x 3 phuong phap x B=100 lan lap, n = 2000 quan sat")
    p("            burn-in 200, k = 2 bien ngoai sinh")
    p()

    ds = {}
    for ma in THAT_MS:
        f = RESULTS / f"07_simulation_{ma}.pt"
        if f.exists():
            ds[ma] = torch.load(f, weights_only=False)
    if not ds:
        p("  (chua co ket qua)")
        p()
        return

    p("  SO LAN LAP HOAN TAT  (B=100 la muc tieu; thieu = MLE phan ky NaN)")
    p(f"    {'kich ban':<10}" + "".join(f"{TEN_PP[m]:>10}" for m in PP) + f"{'dao nhan':>26}")
    for ma, d in ds.items():
        n = "".join(f"{d[m]['n_replications']:>10}" if m in d else f"{'--':>10}" for m in PP)
        dao = "  ".join(f"{TEN_PP[m]}:{d[m].get('n_dao_nhan', '?')}" for m in PP if m in d)
        p(f"    {ma:<10}{n}      {dao:>20}")
    p()
    p("    'dao nhan' = so lan lap bo uoc luong tra ve nhan trang thai NGUOC quy uoc,")
    p("    da duoc hoan vi truoc khi gop. VB dao gan 50% -- tuc khong nhan dang nhan.")
    p()

    p("  COVERAGE TRUNG BINH CUA KHOANG TIN CAY 95%  (ly tuong 0,95)")
    p(f"    {'kich ban':<10}{'VB-MF':>10}{'VB-FR':>10}   {'mo ta'}")
    for ma, d in ds.items():
        p(f"    {ma:<10}{_cov_tb(d.get('vb_mean_field', {})):>10.3f}"
          f"{_cov_tb(d.get('vb_full_rank', {})):>10.3f}   {MO_TA_MS[ma]}")
    p()
    p("    MLE khong co cot nay: quy trinh nay khong sinh khoang tin cay cho MLE.")
    p()

    p("  COVERAGE THEO TUNG THAM SO")
    ten = list(next(iter(ds.values()))["vb_mean_field"]["coverage"])
    p(f"    {'tham so':<11}" + "".join(f"{ma + ' ' + TEN_PP[m][3:]:>9}"
                                       for ma in ds for m in ("vb_mean_field", "vb_full_rank")))
    for t in ten:
        p(f"    {t:<11}" + "".join(f"{ds[ma][m]['coverage'][t]:>9.2f}"
                                   for ma in ds for m in ("vb_mean_field", "vb_full_rank")))
    p()

    for ma, d in ds.items():
        p(f"  UOC LUONG TRUNG BINH -- {ma}: {MO_TA_MS[ma]}")
        p(f"    {'tham so':<10}{'gia tri that':>14}" + "".join(f"{TEN_PP[m]:>11}" for m in PP))
        for t, v in THAT_MS[ma].items():
            hang = "".join(f"{v + d[m]['bias'][t]:>11.3f}" if m in d else f"{'--':>11}" for m in PP)
            p(f"    {t:<10}{v:>14.3f}{hang}")
        p()


def phan_pr_b() -> None:
    p("=" * 78)
    p("PHAN 2 -- PR-B: EGARCH-X Student-t, mot trang thai (8 tham so)")
    p("=" * 78)
    p("  Thiet ke: 3 kich ban x 3 phuong phap x B=100 lan lap, n = 2000 quan sat")
    p()
    edir = RESULTS / "mo_phong_egarch"
    pdir = edir / "parts"
    n_parts = len(glob.glob(str(pdir / "*.pt"))) if pdir.exists() else 0
    p(f"  Tien do: {n_parts}/900 lan lap co checkpoint")
    p()
    co = False
    for ma in ("E1", "E2", "E3"):
        f = edir / f"{ma}.pt"
        if not f.exists():
            p(f"    {ma:<5}chua xong    {MO_TA_E[ma]}")
            continue
        co = True
        d = torch.load(f, weights_only=False)
        that = next(d[m]["tham_so_that"] for m in PP if m in d)
        p(f"    {ma}: {MO_TA_E[ma]}")
        p(f"        theta that = {that['theta']:+.3f}   nu that = {that.get('nu', float('nan')):.2f}")
        p(f"        {'phuong phap':<14}{'n':>5}{'bias(theta)':>14}{'rmse(theta)':>14}{'cov(theta)':>13}")
        for m in PP:
            if m not in d:
                continue
            b = d[m]
            c = b["coverage"].get("theta")
            p(f"        {TEN_PP[m]:<14}{b['n_replications']:>5}"
              f"{b['bias']['theta']:>+14.4f}{b['rmse']['theta']:>14.4f}"
              f"{(f'{c:.3f}' if c is not None else '--'):>13}")
        p()
    if not co:
        p("  (chua kich ban nao hoan tat -- chay lai script nay sau khi xong)")
        p()
    p("  DOC KET QUA: voi E2 (theta_0 = 0), bias(theta) gan 0 nghia la mo hinh KHONG")
    p("  bia ra don bay gia. Neu bias lech khoi 0 mot cach he thong thi moi ket luan")
    p("  've hieu ung don bay cua VN30F1M' deu phai xem lai.")
    p()


def phan_luu_y() -> None:
    p("=" * 78)
    p("PHAN 3 -- NHUNG DIEU PHAI NEU RO KHI BAO CAO")
    p("=" * 78)
    p()
    p("  1. BRIER HIEN CHUA DUNG DUOC.")
    p("     Brier duoc tinh trong tung lan lap tu xac suat loc xi_t, TRUOC khi sap")
    p("     nhan trang thai, va checkpoint chi luu lai con so vo huong chu khong luu")
    p("     xi_t. Vi vay khong sua lai duoc tu du lieu da co. Bang chung cho thay no")
    p("     that su bi hong: Brier cua VB-FR la 0,63-0,68, TE HON ca doan ngau nhien")
    p("     (0,50) -- dieu chi xay ra khi gan nhan sai he thong. Muon co Brier dung")
    p("     phai chay lai bo loc tien mot luot voi tham so da luu (~25 phut), khong")
    p("     can uoc luong lai.")
    p()
    p("  2. THIEU 3 LAN LAP: b=28 cua MLE o CA BA kich ban.")
    p("     Ca ba diem khoi tao deu phan ky ve NaN. Do la loi TAI LAP duoc, khong")
    p("     phai ngau nhien: chi so b quyet dinh dong thoi du lieu (manual_seed(b))")
    p("     va ba diem khoi tao (base_seed+i). Lan lap that bai khong sinh")
    p("     checkpoint nen moi lan chay lai deu thu lai va hong lai (~20 giay).")
    p("     Nen bao cao thanh TY LE KHONG HOI TU cua MLE thay vi giau di.")
    p()
    p("  3. UOC LUONG VB CUA p_ss BI TIEN NGHIEM KEO.")
    p("     Tien nghiem p_ss ~ Beta(8; 2), trung binh 0,800. Gia tri that cua S1 la")
    p("     p11 = 0,98. VB-MF cho 0,790 -- gan nhu dung bang trung binh tien nghiem;")
    p("     VB-FR cho 0,886; MLE (khong co tien nghiem) cho 0,963. Coverage cua")
    p("     p11/p22 bang 0,00 o MOI kich ban va MOI phuong phap Bayes la HE QUA CUA")
    p("     DO CHECH nay, khong phai do khoang tin cay qua hep. Ket qua nay khop voi")
    p("     phan tich do nhay tien nghiem da co (doi sang Beta(20; 2) thi p11 tren du")
    p("     lieu that nhay tu 0,733 len 0,986).")
    p()
    p("  4. COVERAGE CAO O gamma VA mu KHONG PHAI DAU HIEU TOT.")
    p("     Ca ba kich ban deu dat mu_1 = mu_2 = 0 va gamma_1 = gamma_2 = (0,05; 0,05),")
    p("     tuc hai tham so nay GIONG NHAU o ca hai trang thai. Chung mien nhiem voi")
    p("     van de hoan vi nhan, nen coverage 0,72-1,00 cua chung khong noi len dieu")
    p("     gi ve kha nang phan biet trang thai.")
    p()
    p("  5. NU LA THAM SO VO HUONG, khong dinh gi den hoan vi nhan. Coverage cua no")
    p("     (0,00 o S1/S3 nhung 0,75-0,94 o S2) la ket qua sach, dung duoc ngay.")
    p()


def main() -> None:
    p("=" * 78)
    p("TOM TAT NGHIEN CUU KHOI PHUC THAM SO (PARAMETER RECOVERY)")
    p("MS-GARCH-X Student-t va EGARCH-X Student-t cho VN30F1M")
    p("=" * 78)
    p(f"Sinh luc: {datetime.now():%Y-%m-%d %H:%M:%S}")
    p(f"Thu muc:  {RESULTS}")
    p()
    p("Da SAP NHAN TRANG THAI truoc khi gop (tieu chi: phuong sai vo dieu kien")
    p("omega_s/(1-alpha_s-beta_s), trang thai 1 = bien dong thap).")
    p()
    phan_pr_a()
    phan_pr_b()
    phan_luu_y()

    noi_dung = "\n".join(_dong) + "\n"
    OUT.write_text(noi_dung, encoding="utf-8")
    print(noi_dung)
    print(f"-> da ghi {OUT}")


if __name__ == "__main__":
    main()
