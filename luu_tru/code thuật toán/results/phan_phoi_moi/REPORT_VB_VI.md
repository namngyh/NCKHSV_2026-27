# EGARCH-X skewed-t và GED: ước lượng Bayes biến phân

## Cách thực hiện

Dùng đúng dữ liệu `results/00_data.pt` của nghiên cứu gốc: 1.735 phiên huấn
luyện và 435 phiên kiểm định. Hai mô hình mới dùng cùng log-hợp lý với bản MLE
trong `uoc_luong_egarch_x_phan_phoi_moi.py`, kể cả hằng số trung tâm `E|z|`
theo từng phân phối. Đạo hàm của log-hợp lý trong bước VB được tính bằng sai phân
trung tâm; phép kiểm tra `--self-test` đối chiếu giá trị và gradient với
log-hợp lý Student-t gốc tại trường hợp skewed-t có `lambda=0`.

Cho các tham số chung, giữ tiên nghiệm của repo: `mu`, `omega`, `theta` và hai
`gamma` theo `N(0,1)`; `alpha`, `beta` theo `Beta(2,2)`; bậc tự do của skewed-t
có `eta-2 ~ Gamma(2,0.2)` như Student-t gốc. Tham số mới dùng
`(lambda+1)/2 ~ Beta(2,2)` và `shape_GED ~ Gamma(2,1)`.

Mỗi cặp mô hình/guide được tối ưu với seed 0, Adam 0,01, 3.000 bước SVI,
`Trace_ELBO` một mẫu mỗi bước; rút 1.000 mẫu hậu nghiệm và dùng 200 mẫu để
đánh giá ngoài mẫu. `mean-field` dùng `AutoNormal`; `full-rank` dùng
`AutoMultivariateNormal`, như quy trình VB gốc. Student-t trong bảng là kết
quả VB đã lưu của repo, không ước lượng lại.

## Chỉ tiêu ngoài mẫu

| Phân phối | Guide | ELBO cuối¹ | QLIKE ↓ | NLS ↓ | VaR 1% vi phạm | VaR 5% vi phạm |
|---|---|---:|---:|---:|---:|---:|
| Student-t gốc | Mean-field | -2824,151 | 0,92806 | **1,52149** | 6 | 24 |
| Skewed-t | Mean-field | -2819,449 | **0,92605** | 1,52541 | 5 | 20 |
| GED | Mean-field | -2833,519 | 0,92716 | 1,53373 | 6 | 21 |
| Student-t gốc | Full-rank | -2823,089 | 0,92578 | **1,52093** | 6 | 24 |
| Skewed-t | Full-rank | -2817,949 | 0,92400 | 1,52393 | 5 | 20 |
| GED | Full-rank | -2832,696 | **0,91907** | 1,53499 | 6 | 20 |

¹ ELBO là trung bình của 50 bước cuối. Theo quy ước báo cáo gốc, không dùng
ELBO để xếp hạng các đặc tả phân phối khác nhau. QLIKE so sánh phương sai dự
báo với RV; NLS so sánh mật độ dự báo với lợi suất thực tế. VaR trong bảng chỉ
là số vi phạm, chưa phải kết luận từ kiểm định bao phủ.

## Trung bình hậu nghiệm của toàn bộ tham số

### Mean-field

| Tham số | Student-t gốc | Skewed-t | GED |
|---|---:|---:|---:|
| `mu` | 0,097877 | 0,052810 | 0,091256 |
| `omega` | 0,194142 | 0,196987 | 0,195701 |
| `alpha` | 0,108124 | 0,107465 | 0,105649 |
| `theta` | -0,140673 | -0,123077 | -0,118555 |
| `beta` | 0,626380 | 0,636458 | 0,622536 |
| `gamma_volume` | -0,087705 | -0,082472 | -0,082626 |
| `gamma_rv` | 0,322771 | 0,331998 | 0,319975 |
| `nu` (Student-t) | 4,389701 | — | — |
| `eta` (skewed-t) | — | 4,649082 | — |
| `lambda_skew` | — | -0,141787 | — |
| `shape` (GED) | — | — | 1,103728 |

### Full-rank

| Tham số | Student-t gốc | Skewed-t | GED |
|---|---:|---:|---:|
| `mu` | 0,091925 | 0,037890 | 0,091569 |
| `omega` | 0,189684 | 0,193269 | 0,182961 |
| `alpha` | 0,106553 | 0,123601 | 0,105294 |
| `theta` | -0,124090 | -0,145414 | -0,101949 |
| `beta` | 0,639044 | 0,638427 | 0,637252 |
| `gamma_volume` | -0,077157 | -0,082553 | -0,071811 |
| `gamma_rv` | 0,343946 | 0,330598 | 0,343901 |
| `nu` (Student-t) | 4,364673 | — | — |
| `eta` (skewed-t) | — | 4,531658 | — |
| `lambda_skew` | — | -0,132117 | — |
| `shape` (GED) | — | — | 1,120182 |

`theta` là hệ số bất đối xứng trong phương trình EGARCH; `lambda_skew` là
tham số lệch của phân phối nhiễu. Với skewed-t, khoảng hậu nghiệm 95% cho
`lambda_skew` là [-0,19850; -0,08463] ở mean-field và
[-0,19819; -0,06209] ở full-rank. Với GED, khoảng hậu nghiệm 95% cho
`shape` lần lượt là [1,03024; 1,18550] và [1,02524; 1,21587].
Các khoảng này phụ thuộc tiên nghiệm và xấp xỉ VB đã nêu, chưa được kiểm tra
bằng MCMC cho hai phân phối mới.

## Tệp kết quả và chạy lại

`egarch_x_skewt_ged_vb.json` lưu trung bình, độ lệch chuẩn, phân vị 2,5% và
97,5% của mọi tham số cùng chỉ tiêu ngoài mẫu. Bốn tệp
`egarch_x_<phan_phoi>_<guide>_vb.pt` lưu cả mẫu hậu nghiệm và đường ELBO.

Từ thư mục `code thuật toán`:

```powershell
python -m pip install -r requirements.txt
python scripts/uoc_luong_vb_egarch_x_phan_phoi_moi.py --self-test
python scripts/uoc_luong_vb_egarch_x_phan_phoi_moi.py --steps 3000 --draws 1000
```

Script bỏ qua các cặp đã có đủ 3.000 bước. Các tệp MLE và kết quả nghiên cứu
gốc không bị ghi đè.
