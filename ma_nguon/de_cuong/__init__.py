"""Gói thực hiện đúng theo đề cương EGARCH-X ký quỹ động (bản tháng 10/2026).

Tách riêng khỏi `garch_vn30` (thiết kế sáu mô hình cũ) để không làm đổi kết quả đã công bố.
Suy diễn gồm bốn phương pháp: hợp lý cực đại (MLE), Bayes biến phân mean-field và full-rank (VB-MF, VB-FR) và NUTS.
"""
