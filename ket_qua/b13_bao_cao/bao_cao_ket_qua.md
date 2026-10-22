# Kết quả theo đề cương EGARCH-X ký quỹ động (hồ sơ: day_du)

Sinh tự động lúc 2026-10-22 12:54. Hồ sơ `nhanh` chỉ để kiểm tra đường ống.

## 1. Dữ liệu (mục 2)

2171 phiên, 2017-11-06 – 2026-07-17; 105 phiên chuyển hợp đồng; 0 phiên khóa biên; c_rho (UL) = 1.4970.

| Khối | Từ | Đến | Số phiên | Phiên chuyển |
|---|---|---|---|---|
| UL | 2018-01-31 | 2021-01-28 | 750 | 36 |
| XT | 2021-01-29 | 2024-10-16 | 925 | 44 |
| KT | 2024-10-17 | 2026-07-17 | 435 | 22 |

## 2. Kiểm tra cài đặt và đối chiếu độc lập (Phụ lục A, C.5)

Bảng 11 lệch lớn nhất 0.00046; đối chiếu gói arch: EGARCH chênh tham số lớn nhất 0.0004, MCS trùng tập = True, SPA cùng kết luận = True.

### Cỡ và lực của các kiểm định (mô phỏng, N = khối KT)

| Kiểm định | Tỷ lệ bác bỏ 5% | Sai số MC |
|---|---|---|
| berkowitz|co | 0.0510 | 0.0031 |
| christoffersen|a0.01|co | 0.0214 | 0.0020 |
| christoffersen|a0.025|co | 0.0276 | 0.0023 |
| christoffersen|a0.05|co | 0.0494 | 0.0031 |
| dm|ar1 | 0.0858 | 0.0040 |
| dm|iid | 0.0576 | 0.0033 |
| dq_hac|a0.01|H2 | 0.1258 | 0.0047 |
| dq_hac|a0.01|H3 | 0.0910 | 0.0041 |
| dq_hac|a0.025|H2 | 0.0836 | 0.0039 |
| dq_hac|a0.025|H3 | 0.1000 | 0.0042 |
| dq_hac|a0.05|H2 | 0.0658 | 0.0035 |
| dq_hac|a0.05|H3 | 0.0750 | 0.0037 |
| dq|a0.01|co | 0.1596 | 0.0052 |
| dq|a0.025|co | 0.0818 | 0.0039 |
| dq|a0.05|co | 0.0628 | 0.0034 |
| gw|iid | 0.0490 | 0.0031 |
| kupiec|a0.01|co | 0.0256 | 0.0022 |
| kupiec|a0.01|luc | 0.3780 | 0.0069 |
| kupiec|a0.025|co | 0.0720 | 0.0037 |
| kupiec|a0.025|luc | 0.8242 | 0.0054 |
| kupiec|a0.05|co | 0.0492 | 0.0031 |
| kupiec|a0.05|luc | 0.9778 | 0.0021 |

## 3. Mô phỏng khôi phục tham số bằng MLE (mục 4.4)

| Cấu hình | theta chệch | theta độ phủ | beta độ phủ | bác bỏ theta=0 | bác bỏ gamma=0 |
|---|---|---|---|---|---|
| chuan|mle|E1 | -0.0012 | 0.9400 | 0.9400 | 1.0000 | 1.0000 |
| chuan|mle|E2 | 0.0002 | 0.9520 | 0.9450 | 0.0480 | 1.0000 |
| chuan|mle|E3 | -0.0007 | 0.9320 | 0.9440 | 1.0000 | 1.0000 |
| chuan|mle|E4 | -0.0016 | 0.9450 | 0.9420 | 1.0000 | 0.0680 |
| skewt|mle|E1 | -0.0007 | 0.9570 | 0.9330 | 0.9990 | 1.0000 |
| skewt|mle|E2 | -0.0010 | 0.9540 | 0.9400 | 0.0460 | 1.0000 |
| skewt|mle|E3 | -0.0023 | 0.9320 | 0.9400 | 0.9990 | 1.0000 |
| skewt|mle|E4 | -0.0020 | 0.9380 | 0.9320 | 0.9990 | 0.0710 |
| student|mle|E1 | -0.0005 | 0.9490 | 0.9450 | 1.0000 | 1.0000 |
| student|mle|E2 | -0.0007 | 0.9370 | 0.9360 | 0.0630 | 1.0000 |
| student|mle|E3 | -0.0007 | 0.9450 | 0.9270 | 0.9980 | 1.0000 |
| student|mle|E4 | -0.0011 | 0.9470 | 0.9480 | 1.0000 | 0.0450 |

*Mô phỏng Bayes và SBC (b03b) đang chạy; báo cáo sẽ được làm lại khi xong.*

## 4. MLE trên khối UL (mục 4.2)

**chuan|A+G**: logL = -1201.98, AIC = 2423.96, BIC = 2469.67, LR(gamma=0) p = 0.0000, 19/20 điểm khởi tạo về cùng nghiệm.

| Tham số | Ước lượng | SE Hessian | SE bánh kẹp |
|---|---|---|---|
| mu | -0.0043 | 0.0404 | 0.0424 |
| omega | 0.1075 | 0.0298 | 0.0513 |
| kappa | 0.0297 | 0.0814 | 0.1175 |
| theta | -0.2449 | 0.0435 | 0.0650 |
| beta | 0.7250 | 0.0537 | 0.0862 |
| gamma_A | -0.1261 | 0.0342 | 0.0516 |
| gamma_G | 0.3436 | 0.0704 | 0.1063 |
| delta_dh | 0.6549 | 0.2487 | 0.2981 |
| delta_sau | 0.0481 | 0.3295 | 0.5277 |
| delta_nghi | 1.9749 | 0.5198 | 0.7667 |

**student|A+G**: logL = -1152.55, AIC = 2327.11, BIC = 2377.39, LR(gamma=0) p = 0.0003, 18/20 điểm khởi tạo về cùng nghiệm.

| Tham số | Ước lượng | SE Hessian | SE bánh kẹp |
|---|---|---|---|
| mu | 0.0505 | 0.0333 | 0.0335 |
| omega | 0.0848 | 0.0520 | 0.0622 |
| kappa | 0.1683 | 0.1013 | 0.1040 |
| theta | -0.2086 | 0.0611 | 0.0735 |
| beta | 0.7387 | 0.0846 | 0.1134 |
| gamma_A | -0.0718 | 0.0440 | 0.0473 |
| gamma_G | 0.3149 | 0.1030 | 0.1277 |
| delta_dh | 0.7927 | 0.3209 | 0.3196 |
| delta_sau | 0.1098 | 0.4436 | 0.5228 |
| delta_nghi | 1.9234 | 0.8334 | 1.1781 |
| nu | 4.2242 | 0.6772 | 0.7389 |

**skewt|A+G**: logL = -1148.45, AIC = 2320.90, BIC = 2375.75, LR(gamma=0) p = 0.0001, 20/20 điểm khởi tạo về cùng nghiệm.

| Tham số | Ước lượng | SE Hessian | SE bánh kẹp |
|---|---|---|---|
| mu | 0.0014 | 0.0381 | 0.0391 |
| omega | 0.0879 | 0.0492 | 0.0579 |
| kappa | 0.1461 | 0.0892 | 0.0874 |
| theta | -0.1869 | 0.0557 | 0.0662 |
| beta | 0.7563 | 0.0794 | 0.1048 |
| gamma_A | -0.0867 | 0.0416 | 0.0477 |
| gamma_G | 0.3086 | 0.1003 | 0.1259 |
| delta_dh | 0.7721 | 0.3165 | 0.3301 |
| delta_sau | -0.0367 | 0.4133 | 0.4921 |
| delta_nghi | 1.7372 | 0.7414 | 0.9009 |
| nu | 4.5000 | 0.7729 | 0.8188 |
| lambda | -0.1522 | 0.0535 | 0.0555 |

**ged|A+G**: logL = -1157.46, AIC = 2336.92, BIC = 2387.20, LR(gamma=0) p = 0.0002, 4/20 điểm khởi tạo về cùng nghiệm.

| Tham số | Ước lượng | SE Hessian | SE bánh kẹp |
|---|---|---|---|
| mu | 0.0448 |  |  |
| omega | 0.0834 |  |  |
| kappa | 0.1145 |  |  |
| theta | -0.2328 |  |  |
| beta | 0.7281 |  |  |
| gamma_A | -0.0954 |  |  |
| gamma_G | 0.3327 |  |  |
| delta_dh | 0.7884 |  |  |
| delta_sau | 0.0470 |  |  |
| delta_nghi | 1.9294 |  |  |
| nu | 1.1083 |  |  |

### Kiểm tra dự báo tiên nghiệm (mục 4.3.1): phân vị của thống kê dữ liệu trong phân phối dự báo tiên nghiệm

| Phân phối | độ lệch chuẩn | độ nhọn | max |r| | chạm chặn ln h |
|---|---|---|---|---|
| chuan | 0.2120 | 0.0880 | 0.1470 | 0.5530 |
| student | 0.1940 | 0.0750 | 0.1450 | 0.5520 |
| skewt | 0.2140 | 0.0490 | 0.1560 | 0.5590 |
| ged | 0.1940 | 0.0670 | 0.1360 | 0.5790 |

## 5. Chọn biến và phân phối trên XT (mục 6)

Tập biến được chọn: **G**; SPA p = 0.1435; Holm: {'G': 7.720835970559392e-08, 'F': 0.03947289750979266, 'B': 0.23669287312814258, 'A': 0.3241871569122532, 'O': 0.750112156761689}; VIF: [1.0].
Phân phối được chọn: **skewt**.

## 6. Cổng chuyển giai đoạn (mục 8.1)

(i) True; (ii) True; (iii) True; tổng: **True**.

## 7. Độ nhạy tiên nghiệm (mục 4.3.1): chênh lệch trung bình hậu nghiệm theo đơn vị độ lệch chuẩn gốc

| Phương án | mu | omega | kappa | theta | beta | gamma_G | delta_dh | delta_sau | delta_nghi | nu | lambda |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rong |  |  |  |  |  |  |  |  |  |  |  |
| hep |  |  |  |  |  |  |  |  |  |  |  |

## 8. Đánh giá ngoài mẫu trên KT (2024-10-17 – 2026-07-17) (mục 5)

| Mô hình | QLIKE | LS | Vi phạm 1% | Kupiec p | FZ0 2,5% | Berkowitz p |
|---|---|---|---|---|---|---|
| MLE|skewt|EGARCH | 1.6200 | -1.5294 | 4 | 0.9485 | 1.4697 | 0.3900 |
| MLE|skewt|A+G | 1.6983 | -1.5177 | 4 | 0.9485 | 1.4890 | 0.2578 |
| MLE|chuan|G | 1.6668 | -1.6790 | 7 | 0.1967 | 1.5760 | 0.5359 |
| MLE|student|G | 1.7030 | -1.5160 | 5 | 0.6771 | 1.4969 | 0.2655 |
| MLE|skewt|G | 1.6900 | -1.5172 | 4 | 0.9485 | 1.4861 | 0.2535 |
| MLE|ged|G | 1.7016 | -1.5317 | 5 | 0.6771 | 1.5168 | 0.5200 |
| MLE|skewt|GARCH | 1.5946 | -1.5331 | 4 | 0.9485 | 1.5230 | 0.5637 |
| vb_mf|skewt|G | 1.6763 | -1.5168 | 4 | 0.9485 | 1.4684 | 0.2510 |
| vb_fr|skewt|G | 1.6807 | -1.5194 | 4 | 0.9485 | 1.4886 | 0.2267 |
| nuts|skewt|G | 1.6769 | -1.5172 | 4 | 0.9485 | 1.4887 | 0.2497 |

Tập mô hình tin cậy: {'LS': ['MLE|skewt|EGARCH', 'MLE|skewt|A+G', 'MLE|chuan|G', 'MLE|student|G', 'MLE|skewt|G', 'MLE|ged|G', 'MLE|skewt|GARCH', 'vb_mf|skewt|G', 'vb_fr|skewt|G', 'nuts|skewt|G'], 'QLIKE': ['MLE|skewt|EGARCH', 'MLE|skewt|A+G', 'MLE|chuan|G', 'MLE|student|G', 'MLE|skewt|G', 'MLE|ged|G', 'MLE|skewt|GARCH', 'vb_mf|skewt|G', 'vb_fr|skewt|G', 'nuts|skewt|G'], 'FZ0_0.01': ['MLE|skewt|EGARCH', 'MLE|skewt|A+G', 'MLE|chuan|G', 'MLE|student|G', 'MLE|skewt|G', 'MLE|ged|G', 'MLE|skewt|GARCH', 'vb_mf|skewt|G', 'vb_fr|skewt|G', 'nuts|skewt|G'], 'FZ0_0.025': ['MLE|skewt|EGARCH', 'MLE|skewt|A+G', 'MLE|chuan|G', 'MLE|student|G', 'MLE|skewt|G', 'MLE|ged|G', 'MLE|skewt|GARCH', 'vb_mf|skewt|G', 'vb_fr|skewt|G', 'nuts|skewt|G']}

CH1 (EGARCH-X so với EGARCH): {'LS': {'t': -1.5538163857944793, 'p_mot_phia_goc_tot_hon': 0.06011411042438509}, 'QLIKE': {'t': 1.5788865545258037, 'p_mot_phia_goc_tot_hon': 0.9428189597239415}, 'FZ0_0.01': {'t': 0.19389285313857968, 'p_mot_phia_goc_tot_hon': 0.5768701100612776}, 'FZ0_0.025': {'t': 0.25908006773213177, 'p_mot_phia_goc_tot_hon': 0.602213268240666}}

### So sánh bốn phương pháp tại mốc cuối (s0 = 2155): tỷ số độ lệch chuẩn so với NUTS

| Phương pháp | mu | omega | kappa | theta | beta | gamma_G | delta_dh | delta_sau | delta_nghi | nu | lambda |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mle | 1.0057 | 0.9885 | 0.9804 | 1.0972 | 1.1249 | 1.0892 | 0.9720 | 1.0771 | 1.2072 | 0.9866 | 0.9577 |
| vb_mf | 0.8192 | 0.4253 | 0.8349 | 0.8723 | 0.3277 | 0.3928 | 0.6889 | 0.6964 | 0.9603 | 0.6909 | 0.8433 |
| vb_fr | 0.9919 | 0.8881 | 1.0622 | 0.9634 | 0.9163 | 0.8640 | 0.9930 | 0.9843 | 0.9299 | 0.8079 | 0.9724 |

## 9. Ký quỹ động (mục 7), thiết kế trình bày H2|a0.01

Siêu tham số chọn trên XT: {'w': 0.0, 'b': 0.0, 'phi': 0.5, 'eta': 0.0}.

| Quy tắc | m tb | Tỷ lệ vượt | Thiếu hụt tb | Cùng m_tb: thiếu hụt/phiên | DM p |
|---|---|---|---|---|---|
| dong_mle | 0.0545 | 0.0128 | 0.0165 | 0.0002 | 0.5000 |
| dong_egarch_mle | 0.0539 | 0.0128 | 0.0175 | 0.0002 | 0.2550 |
| dong_vb_mf | 0.0554 | 0.0128 | 0.0165 | 0.0002 | 0.0308 |
| dong_vb_fr | 0.0557 | 0.0153 | 0.0154 | 0.0003 | 0.0743 |
| dong_nuts | 0.0552 | 0.0128 | 0.0165 | 0.0002 | 0.0323 |
| VaR_tho_mle | 0.0512 | 0.0153 | 0.0184 | 0.0002 | 0.1069 |
| FHS_EGARCH-X | 0.0543 | 0.0102 | 0.0252 | 0.0003 | 0.0999 |
| EWMA | 0.0416 | 0.0409 | 0.0156 | 0.0003 | 0.0932 |
| lich_su_cua_so | 0.0557 | 0.0179 | 0.0328 | 0.0006 | 0.1209 |
| co_dinh_khop_XT | 0.0775 | 0.0051 | 0.0184 | 0.0005 | 0.1290 |
| VSD_hang_so | 0.1700 | 0.0000 | 0.0000 | 0.0005 | 0.1290 |
| VSD_lich_su | 0.1700 | 0.0000 | 0.0000 | 0.0005 | 0.1290 |
| moc_tren_H_x_bien_do | 0.1400 | 0.0000 | 0.0000 | 0.0005 | 0.1290 |

![Ký quỹ](hinh_ky_quy.png)

![Đường biên](hinh_duong_bien.png)
