"""Danh muc mo hinh -- NGUON SU THAT DUY NHAT ve cac mo hinh duoc so sanh.

Luoi hai chieu, hai o ke nhau chi khac DUNG MOT yeu to:

              | chuan         | Student-t      | Student-t + X
    ----------+---------------+----------------+-----------------
    GARCH     | garch_n  (4)  | garch_t  (5)   | garch_x_t  (7)
    EGARCH    | egarch_n (5)  | egarch_t (6)   | egarch_x_t (8)

(so trong ngoac la so tham so tu do d khi k = 2 bien ngoai sinh).

File nay cung chua hai ham doi qua lai giua bo tham so (GARCHParams /
EGARCHParams) va dict {ten: tensor} -- dang luu tru trong results/*.pt, de
file ket qua khong phu thuoc ten lop Python.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from garch_vn30.mo_hinh.egarch import EGARCHParams, egarch_forward, egarch_loglik
from garch_vn30.mo_hinh.garch import GARCHParams, garch_forward, garch_loglik


@dataclass(frozen=True)
class ModelSpec:
    name: str        # ten day du, dung trong bang bao cao
    code: str        # khoa ky thuat, dung dat ten file ket qua
    family: str      # "garch" hoac "egarch"
    dist: str        # "normal" hoac "student_t"
    use_exog: bool   # co dung bien ngoai sinh X hay khong

    def n_params(self, k: int) -> int:
        """So tham so tu do d: mu, omega, alpha, beta (+theta) + k gamma (+nu)."""
        base = 4 + (1 if self.family == "egarch" else 0)
        return base + (k if self.use_exog else 0) + (1 if self.dist == "student_t" else 0)


CATALOGUE: tuple[ModelSpec, ...] = (
    ModelSpec("GARCH chuẩn", "garch_n", "garch", "normal", False),
    ModelSpec("GARCH Student-t", "garch_t", "garch", "student_t", False),
    ModelSpec("GARCH-X Student-t", "garch_x_t", "garch", "student_t", True),
    ModelSpec("EGARCH chuẩn", "egarch_n", "egarch", "normal", False),
    ModelSpec("EGARCH Student-t", "egarch_t", "egarch", "student_t", False),
    ModelSpec("EGARCH-X Student-t", "egarch_x_t", "egarch", "student_t", True),
)


def get(code: str) -> ModelSpec:
    """Lay ModelSpec theo code; bao loi ro rang neu go sai ten."""
    for spec in CATALOGUE:
        if spec.code == code:
            return spec
    hop_le = ", ".join(s.code for s in CATALOGUE)
    raise KeyError(f"Khong co mo hinh voi code={code!r}. Hop le: {hop_le}")


def lay_X(spec: ModelSpec, X_all: torch.Tensor) -> torch.Tensor:
    """Ma tran ngoai sinh cho mo hinh: X_all neu co dung X, nguoc lai 0 cot."""
    if spec.use_exog:
        return X_all
    return torch.zeros(X_all.shape[0], 0, dtype=X_all.dtype)


def forward_cua(family: str):
    """Ham tinh duong phuong sai tuong ung ho mo hinh."""
    return {"garch": garch_forward, "egarch": egarch_forward}[family]


def loglik_cua(family: str):
    """Ham log-hop-ly tuong ung ho mo hinh."""
    return {"garch": garch_loglik, "egarch": egarch_loglik}[family]


def tham_so_sang_dict(params) -> dict[str, torch.Tensor | None]:
    """GARCHParams / EGARCHParams -> dict {ten: tensor} de luu vao file .pt."""
    return {k: (None if v is None else v.detach().clone()) for k, v in params._asdict().items()}


def tham_so_tu_dict(d: dict, family: str):
    """Chieu nguoc cua tham_so_sang_dict."""
    lop = GARCHParams if family == "garch" else EGARCHParams
    return lop(**{k: d.get(k) for k in lop._fields})
