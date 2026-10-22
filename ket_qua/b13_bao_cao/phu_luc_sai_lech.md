# Phụ lục: sai lệch so với đề cương và các sự kiện cần công bố

Hồ sơ chạy: `day_du`.

## Tham số khác đề cương

- `khoi_bm02.XT`: đề cương 420, đang dùng 925
- `bam_sat_bm02.ged_tham_gia_chon`: đề cương False, đang dùng True
- `mo_phong.B_mle`: đề cương 100, đang dùng 1000
- `mo_phong_bayes.B_vb`: đề cương 100, đang dùng 1000
- `mo_phong_bayes.B_nuts`: đề cương 100, đang dùng 300

## Cổng chuyển giai đoạn được chấp nhận khi không đạt

- Không có.

## Thông số thể chế (Bảng 3) chưa xác minh

- Không có.

## Đơn vị mô phỏng thất bại

- Không có.

## Chẩn đoán Bayes không đạt (b08)

- Các mốc NUTS không đạt (R_hat, ESS, phân kỳ): [0, 5, 7, 22]
- Pareto k_hat của VB đạt theo mốc: {'vb_mf': [False, False, False, False, False, True, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, True, False, True, False, True, False, False, False, True, True], 'vb_fr': [False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]}

## Lựa chọn của nhóm nghiên cứu ở những chỗ đề cương không quy định

Xem `cau_hinh/de_cuong.toml` (chú thích từng giá trị) và THIET_KE_CODE.md mục 6b.
