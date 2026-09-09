"""Danh muc mo hinh -- NGUON SU THAT DUY NHAT ve cac mo hinh duoc so sanh.

Pipeline, bao cao va test deu doc danh sach tu day thay vi tu viet lai. Them
mot mo hinh = them MOT dong trong CATALOGUE.

Quy uoc dat ten: `name` la ten day du dung trong bai bao va bang bao cao (bo
han cac ma B1/B3/M1 cu); `code` chi la khoa ky thuat de dat ten file checkpoint.

Thiet ke la mot luoi hai chieu, moi cap ke nhau chi khac DUNG MOT yeu to:

              | chuan        | Student-t      | Student-t + X
    ----------+--------------+----------------+-----------------
    GARCH     | garch_n  (4) | garch_t  (5)   | garch_x_t  (7)
    EGARCH    | egarch_n (5) | egarch_t (6)   | egarch_x_t (8)

cong hai mo hinh chuyen trang thai: ms_garch_n (10) va ms_garch_x_t (15).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str          # ten day du, dung trong bang bao cao
    code: str          # khoa ky thuat, dung dat ten file checkpoint
    family: str        # "garch" hoac "egarch"
    dist: str          # "normal" hoac "student_t"
    use_exog: bool     # co dung bien ngoai sinh X hay khong
    n_regimes: int     # 1 hoac 2

    def n_params(self, k: int) -> int:
        """So tham so tu do; `k` la so bien ngoai sinh co san trong du lieu.

        Mo hinh khong dung ngoai sinh thi bo qua k. Con so tra ve khop dung
        voi Bang "Sau mo hinh mot trang thai" trong bai bao.
        """
        k_eff = k if self.use_exog else 0
        if self.n_regimes == 1:
            # mu, omega, alpha, beta (+ theta neu EGARCH) + gamma + (nu)
            base = 4 + (1 if self.family == "egarch" else 0)
            return base + k_eff + (1 if self.dist == "student_t" else 0)
        # Hai che do: moi he so phuong sai nhan doi, cong p11/p22 va nu dung chung.
        base = 8 + (2 if self.family == "egarch" else 0)
        return base + 2 * k_eff + 2 + (1 if self.dist == "student_t" else 0)

    @property
    def is_bayes_ready(self) -> bool:
        """Da co mo hinh Pyro tuong ung de chay VB/MCMC chua.

        Hien tai: mot trang thai (bayes_single.py) va MS-GARCH-X hai trang
        thai (bayes_model.py). MS-EGARCH chua co -- xem CLAUDE.md.
        """
        if self.n_regimes == 1:
            return True
        return self.family == "garch"


CATALOGUE: tuple[ModelSpec, ...] = (
    # --- mot trang thai, ho GARCH -----------------------------------------
    ModelSpec("GARCH chuẩn", "garch_n", "garch", "normal", False, 1),
    ModelSpec("GARCH Student-t", "garch_t", "garch", "student_t", False, 1),
    ModelSpec("GARCH-X Student-t", "garch_x_t", "garch", "student_t", True, 1),
    # --- mot trang thai, ho EGARCH ----------------------------------------
    ModelSpec("EGARCH chuẩn", "egarch_n", "egarch", "normal", False, 1),
    ModelSpec("EGARCH Student-t", "egarch_t", "egarch", "student_t", False, 1),
    ModelSpec("EGARCH-X Student-t", "egarch_x_t", "egarch", "student_t", True, 1),
    # --- hai trang thai ---------------------------------------------------
    ModelSpec("MS-GARCH chuẩn", "ms_garch_n", "garch", "normal", False, 2),
    ModelSpec("MS-GARCH-X Student-t", "ms_garch_x_t", "garch", "student_t", True, 2),
)


def get(code: str) -> ModelSpec:
    """Lay mot ModelSpec theo code; loi ro rang neu go sai ten."""
    for spec in CATALOGUE:
        if spec.code == code:
            return spec
    hop_le = ", ".join(s.code for s in CATALOGUE)
    raise KeyError(f"Khong co mo hinh voi code={code!r}. Hop le: {hop_le}")
