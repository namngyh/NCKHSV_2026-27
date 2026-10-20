"""Danh mục mô hình -- NGUỒN SỰ THẬT DUY NHẤT về các mô hình được so sánh.

Script, báo cáo và kiểm thử đều đọc danh sách từ đây; thêm một mô hình chỉ cần
thêm một dòng vào CATALOGUE. Danh mục là một lưới hai chiều, hai ô kề nhau chỉ
khác ĐÚNG MỘT yếu tố nên chênh lệch kết quả giữa chúng cho biết đóng góp của
yếu tố đó:

              | chuẩn         | Student-t      | Student-t + X
    ----------+---------------+----------------+-----------------
    GARCH     | garch_n  (4)  | garch_t  (5)   | garch_x_t  (7)
    EGARCH    | egarch_n (5)  | egarch_t (6)   | egarch_x_t (8)

(số trong ngoặc là số tham số tự do d khi có k = 2 biến ngoại sinh).

File này cũng chứa các hàm tiện ích dùng chung:
    - lay_X: ma trận ngoại sinh đúng cho từng mô hình;
    - forward_cua, loglik_cua: chọn hàm tính theo họ GARCH hay EGARCH;
    - tham_so_sang_dict / tham_so_tu_dict: đổi bộ tham số <-> dict tensor. Tệp
      kết quả lưu dict chứ không lưu đối tượng Python, để đọc lại được mà không
      phụ thuộc cấu trúc mã nguồn.

THƯ VIỆN
    dataclasses (thư viện chuẩn Python): @dataclass(frozen=True) tạo lớp dữ liệu
    bất biến, tự sinh __init__ và phép so sánh.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from garch_vn30.mo_hinh.egarch import EGARCHParams, egarch_forward, egarch_loglik
from garch_vn30.mo_hinh.garch import GARCHParams, garch_forward, garch_loglik


@dataclass(frozen=True)
class ModelSpec:
    """Đặc tả một mô hình trong danh mục."""
    name: str        # tên đầy đủ, dùng trong bảng báo cáo
    code: str        # khóa kỹ thuật, dùng đặt tên tệp kết quả (vd. "egarch_x_t")
    family: str      # họ phương trình phương sai: "garch" hoặc "egarch"
    dist: str        # phân phối sai số: "normal" hoặc "student_t"
    use_exog: bool   # có dùng biến ngoại sinh X hay không

    def n_params(self, k: int) -> int:
        """Số tham số tự do d = mu, omega, alpha, beta (+ theta nếu EGARCH)
        + k hệ số gamma (nếu dùng X) + nu (nếu Student-t). Dùng trong AIC, BIC."""
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
    """Tra ModelSpec theo mã (vd. get("egarch_x_t")); báo lỗi rõ ràng nếu gõ sai tên."""
    for spec in CATALOGUE:
        if spec.code == code:
            return spec
    hop_le = ", ".join(s.code for s in CATALOGUE)
    raise KeyError(f"Khong co mo hinh voi code={code!r}. Hop le: {hop_le}")


def lay_X(spec: ModelSpec, X_all: torch.Tensor) -> torch.Tensor:
    """Ma trận ngoại sinh cho mô hình `spec`: X_all nếu mô hình dùng X, ngược lại
    một ma trận 0 cột (khi đó gamma là vector rỗng và gamma^T X = 0)."""
    if spec.use_exog:
        return X_all
    return torch.zeros(X_all.shape[0], 0, dtype=X_all.dtype)


def forward_cua(family: str):
    """Trả về hàm tính đường phương sai của họ mô hình (garch_forward hoặc egarch_forward)."""
    return {"garch": garch_forward, "egarch": egarch_forward}[family]


def loglik_cua(family: str):
    """Trả về hàm log-hợp lý của họ mô hình (garch_loglik hoặc egarch_loglik)."""
    return {"garch": garch_loglik, "egarch": egarch_loglik}[family]


def tham_so_sang_dict(params) -> dict[str, torch.Tensor | None]:
    """GARCHParams / EGARCHParams -> dict {tên: tensor} để lưu vào tệp .pt.

    `.detach().clone()` tách tensor khỏi đồ thị đạo hàm và tạo bản sao độc lập.
    """
    return {k: (None if v is None else v.detach().clone()) for k, v in params._asdict().items()}


def tham_so_tu_dict(d: dict, family: str):
    """Chiều ngược của tham_so_sang_dict: dict -> GARCHParams (family "garch") hoặc EGARCHParams."""
    lop = GARCHParams if family == "garch" else EGARCHParams
    return lop(**{k: d.get(k) for k in lop._fields})
