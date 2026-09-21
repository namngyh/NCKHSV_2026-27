"""Kich ban mo phong "do thien vi hieu chinh" (calibration bias) cho
MS-GARCH-X Student-t.

CAU HOI KIEM DINH: neu chon theta_0 = nghiem uoc luong THAT tren VN30F1M cua
mot phuong phap (MLE, VB mean-field, hay VB full-rank), roi mo phong lai va
uoc luong lai bang CA BA phuong phap, thi phuong phap TRUNG voi nguon hieu
chinh co he thong "thang" (bias/RMSE thap hon hai phuong phap con lai) khong?
Neu dung tren ca ba hang -> xac nhan moi lo circularity da neu khi ban ve
viec dung MOT kich ban hieu chinh don le tu du lieu that (xem CLAUDE.md,
muc "YEU CAU DANG BAN").

Khac voi S1-S8 (scenarios.py, tham so MINH HOA), theta_0 o day la NGHIEM THAT
lay tu results/02_mle.pt / 03_vb.pt -- vi vay module nay CHU DINH KHONG tu
doc file ket qua (thu vien msgarchx/ khong gan voi duong dan du an cu the,
dung quy uoc nhu run_egarch.py). Xem scripts/chay_mo_phong_hieu_chinh.py de
nap ket qua that roi goi build_calibration_scenarios().

Luoi 3x3: 3 nguon hieu chinh (hang) x 3 phuong phap uoc luong lai (cot).
Doc theo hang: o duong cheo (phuong phap = nguon) luon tot hon o ngoai duong
cheo hay khong.
"""
from __future__ import annotations

from msgarchx.inference.param_spec import ModelConfig
from msgarchx.models.ms_garch import MSGARCHXParams
from msgarchx.simulation.scenarios import SimulationScenario

# Khop voi config that da dung khi uoc luong MS-GARCH-X Student-t tren
# VN30F1M (xem results/02_mle.pt: config_m1 = ModelConfig(n_exog=2,
# state_specific_nu=False)).
CALIBRATION_CONFIG = ModelConfig(n_exog=2, state_specific_nu=False)

TEN_NGUON = {
    "mle": "Hop ly cuc dai (MLE)",
    "vb_mean_field": "Bayes bien phan mean-field",
    "vb_full_rank": "Bayes bien phan full-rank",
}

# Thu tu co dinh dung xuyen suot (hang cua luoi 3x3).
CAC_NGUON = ("mle", "vb_mean_field", "vb_full_rank")


def ma_kich_ban(nguon: str) -> str:
    return f"CAL_{nguon}"


def build_calibration_scenarios(
    theta_by_source: dict[str, MSGARCHXParams],
    T: int = 2000,
    B: int = 100,
    burn_in: int = 200,
) -> dict[str, SimulationScenario]:
    """Dung 3 SimulationScenario tu theta_0 = nghiem THAT cua tung phuong phap.

    `theta_by_source`: vd {"mle": MSGARCHXParams(...), "vb_mean_field": ...,
    "vb_full_rank": ...}. Moi MSGARCHXParams phai co truong `P` la ma tran
    chuyen trang thai THAT (KHONG phai placeholder eye(2) nhu scenarios.py) --
    day chinh la P0 dung de sinh du lieu, khac voi quy uoc cua S1-S8.

    Tra ve dict ma_kich_ban(nguon) -> SimulationScenario.
    """
    thieu = set(theta_by_source) - set(CAC_NGUON)
    if thieu:
        raise ValueError(f"Nguon hieu chinh khong hop le: {thieu}. Hop le: {CAC_NGUON}")

    out: dict[str, SimulationScenario] = {}
    for nguon, params in theta_by_source.items():
        out[ma_kich_ban(nguon)] = SimulationScenario(
            code=ma_kich_ban(nguon),
            description=f"theta_0 = nghiem that VN30F1M cua {TEN_NGUON[nguon]}",
            params=params,
            P0=params.P,
            T=T, B=B, burn_in=burn_in,
            misspecified=False,
        )
    return out
