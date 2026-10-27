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
- `mo_hinh.bien_ung_vien`: đề cương ['A', 'G', 'F', 'B', 'O'], đang dùng ['A', 'G', 'F', 'B', 'O', 'RV']
- `cong.kiem_dinh_do_phu`: đề cương nguong_cung, đang dùng nhi_thuc
- `cong.kiem_tra_mo_hinh_chon`: đề cương False, đang dùng True
- `ky_quy.x_tuong_lai`: đề cương ar1, đang dùng ar1_kem_z
- `du_lieu.dai_dien_phuong_sai`: đề cương gk, đang dùng rv
- `ky_quy.phia`: đề cương max_hai_phia, đang dùng tri_tuyet_doi

## Cổng chuyển giai đoạn được chấp nhận khi không đạt

- Không có.

## Thông số thể chế (Bảng 3) chưa xác minh

- Không có.

## Đơn vị mô phỏng thất bại

- Không có.

## Chẩn đoán Bayes không đạt (b08)

- Các mốc NUTS không đạt (R_hat, ESS, phân kỳ): [2, 3, 30, 37, 52]
- Pareto k_hat của VB đạt theo mốc: {'vb_mf': [False, False, False, False, False, True, True, False, False, True, False, True, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, True, False, False, False, True, False, False, False, False, False, False, True], 'vb_fr': [False, True, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]}

## Bổ sung ngoài đề cương (lựa chọn thiết kế của nhóm nghiên cứu, chốt trước khi mở khối KT)

- Vùng không điều chỉnh của tỷ lệ ký quỹ (bước 6): tăng khi mức mục tiêu cao hơn mức áp dụng từ 1 điểm %, giảm khi thấp hơn từ 2 điểm %; đối chứng không ngưỡng và hai mức ngưỡng khác ở b12. BM02 chỉ nêu "giới hạn thay đổi giữa hai phiên".
- Đối chứng cách tính của VSDC (QĐ 61/QĐ-VSD 2022, QĐ 12/QĐ-HĐTV 2023, QĐ 26/QĐ-HĐTV 2025), n = 2, kỳ quan sát 90/120/250 ngày, cập nhật hằng ngày và theo lịch 1, 10, 20 (b12). Đây là tính lại theo văn bản, không phải chuỗi VSDC công bố.
- Quy đổi ký quỹ và thiếu hụt ra đồng mỗi hợp đồng bằng hệ số nhân 100.000 đồng/điểm (b12).

## Thay đổi sau khi khối KT đã được nhìn thấy và các sự kiện cần công bố

- Khối KT (435 phiên cuối, từ 17/10/2024) không phải mẫu giữ lại thật sự: nghiên cứu gốc tháng 9/2026 đã đánh giá ngoài mẫu trên đúng 435 phiên này (và đã dùng log RV làm biến ngoại sinh); đợt 20/10/2026 dùng khối KT 990 phiên bao trùm chúng; cấu hình đã được khóa hai lần ngày 24/10/2026 (skew-t | G, rồi GED | G) và b08-b15 đã chạy trên KT trước các thay đổi dưới đây.
- Biến ngoại sinh RV (phương sai thực hiện đóng-đến-đóng từ nến phút, tách nghỉ trưa và qua đêm, qua đêm sau đáo hạn dùng giá F2M) đã có trong nghiên cứu gốc tháng 9/2026 nhưng bị bỏ sót khi viết đề cương: đề cương giả định 'không có dữ liệu trong ngày' (mục 2.9) nên bảng ứng viên chỉ có A, G, F, B, O, trong khi dữ liệu là nến phút và chính đề cương nêu phương sai thực hiện tốt hơn biên độ giá khi có dữ liệu trong ngày (mục 1.6, 2.4 và 2.9). RV được đưa lại vào tập ứng viên ngày 25/10/2026 để sửa tiền đề sai này, không dựa vào kết quả trên KT; thiết kế có RV chưa từng được đánh giá trên KT (lần chạy 25/10 dừng ở cổng b07).
- 26/10/2026: cùng lý do, đại lượng đại diện phương sai cho QLIKE và c_rho đổi từ Garman-Klass cộng qua đêm sang RV đóng-đến-đóng ([du_lieu].dai_dien_phuong_sai).
- 26/10/2026: điều kiện (i) của cổng đổi từ ngưỡng cứng sang kiểm định nhị thức một phía (Holm) và thêm mô phỏng khôi phục quanh mô hình được chọn (b06c). Thay đổi được làm sau khi thấy cổng không đạt ở lần chạy 25/10 (độ phủ nhỏ nhất 0,899 so với 0,90; kết quả mô phỏng, không phải KT). Lý do: với B = 1.000 sai số chuẩn của độ phủ khoảng 0,009, nên ngưỡng cứng quyết định bằng nhiễu mô phỏng; kết luận theo ngưỡng cứng vẫn được ghi song song (dat_nguong_cung).
- 26/10/2026: ký quỹ một tỷ lệ cho cả hai phía được so với L = |R| (mục 7.4) nên mức ký quỹ là phân vị (1 - alpha) của |R| ([ky_quy].phia = tri_tuyet_doi). Đề cương lấy max(VaR mua, VaR bán) ở mức alpha, khi đó P(|R| > m) xấp xỉ 2 alpha và kiểm định Kupiec ở mức alpha sai về thiết kế; các đối chứng lịch sử và EWMA dùng cùng định nghĩa.
- 26/10/2026: sửa lỗi lập trình trước lần chạy lại: (a) bình phương lợi suất qua khoảng nghỉ trong RV cộng theo từng khoảng; (b) số phiên đến đáo hạn ở cuối mẫu tính tới kỳ đáo hạn của tháng hiện tại nếu chưa qua; (c) c_rho chỉ dùng cửa sổ ước lượng (bỏ 61 phiên đầu); (d) mô phỏng lịch sử có lọc dùng E|z| của phần dư được lấy mẫu; (e) thống kê mô tả lợi suất của b01 chỉ tính trên UL + XT; (f) cờ Hessian không xác định dương được ghi kèm sai số chuẩn; (g) mã băm của mỗi bước gồm cả quy_trinh/_chung.py và các tệp bước khác mà nó gọi, để sửa mã luôn làm bước chạy lại; (h) nhóm phiên căng thẳng trên KT dùng cả lịch sử trước KT cho cửa sổ 250 phiên.
- 26/10/2026: các chuỗi NUTS được khởi tạo phân tán theo sai số chuẩn của từng tham số (xi_MLE + 2 se_xi N(0, 1), [bayes].nuts_he_so_phan_tan) để R_hat có ý nghĩa; bản trước cộng nhiễu 0,05 như nhau, nhỏ hơn độ lệch chuẩn hậu nghiệm của beta. Đề cương không quy định cách khởi tạo.
- 26/10/2026: hồi quy chẩn đoán z^2 theo biến lịch dùng đủ ba biến giả (bản trước bỏ D_sau) và hiệp phương sai HC3; hồi quy theo số phiên đến đáo hạn cũng dùng HC3.
- 26/10/2026: mô phỏng H phiên cho biến ngoại sinh tương lai phụ thuộc cú sốc cùng phiên (x_tuong_lai = ar1_kem_z); đề cương dùng AR(1) độc lập.
- Khoảng Wald của kappa thiếu độ phủ có hệ thống trong mô phỏng b03a (lần chạy 25/10: trung bình 0,912 trên 16 ô, thấp nhất ở mọi ô so với danh nghĩa 0,95).
- Giá đóng cửa VN30F2M ngày 15/08/2019 (877,0) bổ sung từ bản tin KIS/Bloomberg, chưa đối chiếu với HNX hoặc DataPro (du_lieu/goc/VN30F2M_bo_sung.md).

## Lựa chọn của nhóm nghiên cứu ở những chỗ đề cương không quy định

Xem `cau_hinh/de_cuong.toml` (chú thích từng giá trị) và THIET_KE_CODE.md mục 6b.
