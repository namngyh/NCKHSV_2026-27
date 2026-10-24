# Phụ lục: sai lệch so với đề cương và các sự kiện cần công bố

Hồ sơ chạy: `day_du`.

## Tham số khác đề cương chi tiết

Giá trị tham chiếu là của đề cương chi tiết (`tai_lieu/de_cuong/`). Các điểm về khối UL/XT/KT và GED tham gia chọn phân phối
là có chủ đích: nhóm bám đề cương đã đăng ký BM02, vốn khác đề cương chi tiết ở các điểm này.

- `du_lieu.xu_ly_dao_han`: đề cương loai, đang dùng f2m
- `mo_phong.khuyet_phien_chuyen`: đề cương True, đang dùng False
- `khoi_bm02.XT`: đề cương 420, đang dùng 925
- `bam_sat_bm02.ged_tham_gia_chon`: đề cương False, đang dùng True
- `mo_phong.B_mle`: đề cương 100, đang dùng 1000
- `mo_phong_bayes.B_vb`: đề cương 100, đang dùng 1000
- `mo_phong_bayes.B_nuts`: đề cương 100, đang dùng 300
- `ky_quy.nguong_tang`: đề cương 0.0, đang dùng 0.01
- `ky_quy.nguong_giam`: đề cương 0.0, đang dùng 0.02

## Cổng chuyển giai đoạn được chấp nhận khi không đạt

- Không có.

## Thông số thể chế (Bảng 3) chưa xác minh

- Không có.

## Đơn vị mô phỏng thất bại

- Không có.

## Chẩn đoán Bayes không đạt (b08)

- Các mốc NUTS không đạt (R_hat, ESS, phân kỳ): []
- Pareto k_hat của VB đạt theo mốc: {'vb_mf': [False, False, True, False, False, True, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, True, True, False, True, False, False, False, True, False, True, False, False, False, False, False, True, False, True, True, False, False, True, False, False, False, True, False, False, False, True, False, False, False, True, False, True, False, False, False, True, False, False], 'vb_fr': [True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, True, False, True, True, True]}

## Bổ sung ngoài đề cương (lựa chọn thiết kế của nhóm nghiên cứu, chốt trước khi mở khối KT)

- Vùng không điều chỉnh của tỷ lệ ký quỹ (bước 6): tăng khi mức mục tiêu cao hơn mức áp dụng từ 1 điểm %, giảm khi thấp hơn từ 2 điểm %; đối chứng không ngưỡng và hai mức ngưỡng khác ở b12. BM02 chỉ nêu "giới hạn thay đổi giữa hai phiên".
- Đối chứng cách tính của VSDC (QĐ 61/QĐ-VSD 2022, QĐ 12/QĐ-HĐTV 2023, QĐ 26/QĐ-HĐTV 2025), n = 2, kỳ quan sát 90/120/250 ngày, cập nhật hằng ngày và theo lịch 1, 10, 20 (b12). Đây là tính lại theo văn bản, không phải chuỗi VSDC công bố.
- Quy đổi ký quỹ và thiếu hụt ra đồng mỗi hợp đồng bằng hệ số nhân 100.000 đồng/điểm (b12).

## Lựa chọn của nhóm nghiên cứu ở những chỗ đề cương không quy định

Xem `cau_hinh/de_cuong.toml` (chú thích từng giá trị) và THIET_KE_CODE.md mục 6b.
