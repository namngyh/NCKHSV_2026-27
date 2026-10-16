# EGARCH-X với skewed-t và GED: ước lượng MLE

## Thiết kế

Giữ nguyên dữ liệu VN30F1M và điểm chia của `results/00_data.pt`: 2.170 phiên
(07/11/2017–17/07/2026), trong đó 1.735 phiên huấn luyện và 435 phiên kiểm định
(bắt đầu 17/10/2024). Hai biến ngoại sinh là log khối lượng và log RV của phiên
trước, đã chuẩn hóa bằng thống kê tập huấn luyện. Mô hình trung bình có hằng số
`mu`; phương trình phương sai giống `egarch.py`:

`log h[t] = omega + alpha*(abs(z[t-1])-EabsZ) + theta*z[t-1] + beta*log h[t-1] + gamma'*X[t]`.

Skewed-t dùng tham số hóa chuẩn hóa của Hansen (1994), gồm bậc tự do `eta > 2`
và độ lệch `lambda` (lambda = 0 đúng bằng Student-t chuẩn hóa). GED dùng hàm mũ
lũy thừa với tham số hình dạng `shape > 0` và thang tỷ lệ được chọn để phương
sai bằng 1. Cả hai đều dùng `E|z|` đúng theo phân phối tương ứng. Log-hợp lý,
AIC và BIC chỉ dùng các quan sát huấn luyện từ t=1; điểm đầu dùng làm điều kiện
khởi tạo, giống mã gốc. Mỗi mô hình tối ưu từ 10 điểm khởi tạo L-BFGS-B.

## Kết quả

| Mô hình | d | Log-hợp lý train | AIC | BIC | QLIKE test | NLS test | VaR 1% vi phạm | VaR 5% vi phạm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EGARCH-X Student-t (gốc) | 8 | -2795,434 | 5606,868 | 5650,538 | 0,92344 | **1,52031** | 6 | 23 |
| EGARCH-X skewed-t | 9 | **-2787,182** | **5592,364** | **5641,493** | 0,91823 | 1,52348 | 6 | 20 |
| EGARCH-X GED | 8 | -2804,426 | 5624,852 | 5668,522 | **0,90893** | 1,53584 | 7 | 20 |

`d` là số tham số tự do. QLIKE/NLS nhỏ hơn là tốt hơn. AIC/BIC chỉ tính cho MLE.
Các giá trị Student-t đọc từ kết quả gốc; log-hợp lý của mô hình này được tính
lại bằng mã mở rộng ở `lambda=0` và khớp tới sai số dưới 1e-6.

Skewed-t ước lượng `eta=4,46793`, `lambda=-0,13193` và cho AIC/BIC thấp hơn
Student-t. Kiểm định LR cho ràng buộc lồng nhau `lambda=0` có thống kê
`2*(LL_skewt-LL_t)=16,5042`, p=0,0000485 theo chi bình phương 1 bậc tự do.
NLS ngoài mẫu của skewed-t cao hơn Student-t khoảng 0,00316; vì vậy cải thiện
trong mẫu chưa chuyển thành cải thiện điểm mật độ ngoài mẫu ở tập kiểm định này.

GED ước lượng `shape=1,13196`, cho QLIKE thấp nhất trong ba mô hình nhưng
AIC/BIC và NLS cao hơn Student-t. Các chỉ tiêu đo mục tiêu khác nhau: QLIKE
đánh giá dự báo phương sai so với RV; NLS đánh giá mật độ lợi suất quan sát.
Chỉ riêng thứ hạng điểm số chưa chứng minh chênh lệch ngoài mẫu có ý nghĩa
thống kê. VaR trong bảng là số vi phạm, chưa phải kết luận từ kiểm định bao phủ.

## Tham số MLE

| Tham số | Student-t gốc | Skewed-t | GED |
|---|---:|---:|---:|
| `mu` | 0,079626 | 0,032978 | 0,077761 |
| `omega` | 0,159887 | 0,159019 | 0,142387 |
| `alpha` | 0,072144 | 0,073222 | 0,065164 |
| `theta` (bất đối xứng EGARCH) | -0,145171 | -0,130624 | -0,117431 |
| `beta` | 0,683154 | 0,692005 | 0,696702 |
| `gamma_volume` | -0,069185 | -0,067484 | -0,063211 |
| `gamma_rv` | 0,298596 | 0,285588 | 0,283908 |
| `nu` (bậc tự do Student-t) | 4,290965 | — | — |
| `eta` (bậc tự do skewed-t) | — | 4,467930 | — |
| `lambda_skew` | — | -0,131932 | — |
| `shape` (GED) | — | — | 1,131956 |
| `E|z|` (hằng số suy ra) | 0,717491 | 0,722824 | 0,727908 |

Giá trị đầy đủ và chẩn đoán 10 lượt tối ưu nằm trong
`egarch_x_skewt_ged_mle.json`. Tất cả lượt tối ưu đều báo hội tụ và đạt cùng
log-hợp lý tới sáu chữ số thập phân.

## Chạy lại

Từ thư mục `code thuật toán`:

```powershell
python -m pip install -r requirements.txt
python scripts/uoc_luong_egarch_x_phan_phoi_moi.py --self-test
python scripts/uoc_luong_egarch_x_phan_phoi_moi.py --starts 10
```

Mã chỉ đọc `results/00_data.pt` và `results/pipeline/egarch_x_t_mle.pt`; tệp
kết quả mới được ghi trong `results/phan_phoi_moi/`. Phần mở rộng này thực hiện
MLE, chưa thêm hai phân phối vào luồng VB hoặc MCMC của nghiên cứu gốc.

Tham khảo tham số hóa skewed-t: Hansen, B. E. (1994),
[Autoregressive Conditional Density Estimation](https://www.ssc.wisc.edu/~bhansen/papers/ier_94.pdf).
