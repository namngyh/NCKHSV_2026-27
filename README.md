# MS-GARCH-X Student-t cho VN30F1M — Tài liệu thuật toán và tính toán

Kho mã nguồn này cài đặt khung nghiên cứu **Markov-Switching GARCH-X với sai số
Student-t chuẩn hóa** (viết tắt: MS-GARCH-X Student-t) cho chuỗi lợi suất hợp
đồng tương lai VN30F1M, và so sánh ba phương pháp suy diễn trên **cùng một hàm
hợp lý**: ước lượng hợp lý cực đại (MLE), Variational Bayes (VB) và Markov Chain
Monte Carlo (MCMC).

Tài liệu này trình bày đầy đủ cơ sở toán học, thuật toán và quy trình tính toán
được cài đặt. Hướng dẫn cài đặt thư viện, chạy thử và các lưu ý vận hành nằm ở
[`code thuật toán/README.md`](code%20thuật%20toán/README.md).

**Mục lục**

1. [Câu hỏi nghiên cứu](#1-câu-hỏi-nghiên-cứu)
2. [Bảng ký hiệu](#2-bảng-ký-hiệu)
3. [Dữ liệu và biến số](#3-dữ-liệu-và-biến-số)
4. [Mô hình](#4-mô-hình)
5. [Bộ lọc tiến và hàm hợp lý](#5-bộ-lọc-tiến-và-hàm-hợp-lý)
6. [Tham số hóa](#6-tham-số-hóa)
7. [Ba phương pháp suy diễn](#7-ba-phương-pháp-suy-diễn)
8. [Dự báo](#8-dự-báo)
9. [Đánh giá ngoài mẫu](#9-đánh-giá-ngoài-mẫu)
10. [Kiểm định thống kê ban đầu](#10-kiểm-định-thống-kê-ban-đầu)
11. [Nghiên cứu mô phỏng](#11-nghiên-cứu-mô-phỏng)
12. [Hệ thống mô hình](#12-hệ-thống-mô-hình)
13. [Chi phí tính toán](#13-chi-phí-tính-toán)
14. [Ánh xạ công thức sang mã nguồn](#14-ánh-xạ-công-thức-sang-mã-nguồn)
15. [Quy trình chạy](#15-quy-trình-chạy)
16. [Những điểm chưa hoàn tất](#16-những-điểm-chưa-hoàn-tất)

---

## 1. Câu hỏi nghiên cứu

Toàn bộ thiết kế tính toán phục vụ sáu câu hỏi:

| Mã | Câu hỏi | Bằng chứng dùng để trả lời |
|---|---|---|
| RQ1 | Student-t có cải thiện so với phân phối chuẩn không? | So sánh B1 với B2; AIC/BIC, điểm log, kiểm định VaR |
| RQ2 | Chuyển trạng thái có cải thiện so với một trạng thái không? | So sánh B3 với M1; QLIKE, điểm log ngoài mẫu |
| RQ3 | VB có xấp xỉ đủ gần hậu nghiệm của MCMC không? | Tỷ số độ lệch chuẩn $R_{\mathrm{sd},j}$ giữa M2/M3 và M4 |
| RQ4 | Truyền bất định tham số vào dự báo có ích không? | So sánh M1 (điểm) với M2/M3/M4 (hậu nghiệm) |
| RQ5 | Đánh đổi giữa độ chính xác và chi phí tính toán? | Bảng thời gian chạy và chẩn đoán hội tụ |
| RQ6 | Bậc tự do có khác nhau giữa hai trạng thái không? | Mô hình R1, xác suất hậu nghiệm $\Pr(\nu_2<\nu_1\mid \mathcal{D}_T)$ |

Nguyên tắc xuyên suốt: ba phương pháp suy diễn phải dùng **cùng một hàm hợp lý và
cùng một cơ chế thu gọn**. Nếu mỗi phương pháp dùng một xấp xỉ khác nhau thì
chênh lệch kết quả không còn phản ánh thuần túy sự khác biệt của phương pháp.

---

## 2. Bảng ký hiệu

Ký hiệu dưới đây được dùng nhất quán trong toàn bộ tài liệu và trùng với ký hiệu
trong bản thảo.

| Ký hiệu | Ý nghĩa | Miền giá trị |
|---|---|---|
| $t$ | Chỉ số phiên giao dịch | $t=1,\dots,T$ |
| $P_t$ | Giá đóng cửa | $P_t>0$ |
| $r_t$ | Lợi suất logarit, đơn vị phần trăm | $\mathbb{R}$ |
| $RV_t$ | Biến động thực tế trong phiên $t$ | $RV_t\ge 0$ |
| $S_t$ | Trạng thái ẩn tại $t$ | $\{1,2\}$ |
| $s$ | Chỉ số trạng thái | $s\in\{1,2\}$ |
| $\mu_s$ | Trung bình lợi suất trong trạng thái $s$ | $\mathbb{R}$ |
| $\varepsilon_t$ | Sai số | $\mathbb{R}$ |
| $z_t$ | Sai số chuẩn hóa | $\mathbb{R}$ |
| $\nu$ | Bậc tự do của Student-t | $\nu>2$ |
| $h_{t,s}$ | Phương sai có điều kiện trong trạng thái $s$ | $h_{t,s}>0$ |
| $\omega_s$ | Hằng số của phương trình phương sai | $\omega_s>0$ |
| $\alpha_s$ | Hệ số phản ứng với cú sốc (ARCH) | $\alpha_s\ge 0$ |
| $\beta_s$ | Hệ số dai dẳng (GARCH) | $\beta_s\ge 0$ |
| $\boldsymbol{\gamma}_s$ | Vector hệ số của biến ngoại sinh | $\mathbb{R}^k$ |
| $\mathbf{X}_{t-1}$ | Vector biến ngoại sinh, $k$ phần tử | $\mathbb{R}^k$ |
| $k$ | Số biến ngoại sinh | $k=2$ trong nghiên cứu này |
| $p_{ij}$ | Xác suất chuyển từ trạng thái $i$ sang $j$ | $[0,1]$ |
| $\mathbf{P}$ | Ma trận chuyển trạng thái $2\times 2$ | Mỗi hàng cộng bằng 1 |
| $\boldsymbol{\pi}$ | Phân phối dừng của chuỗi Markov | $\pi_1+\pi_2=1$ |
| $d_s$ | Thời gian tồn tại kỳ vọng của trạng thái $s$ | $d_s\ge 1$ |
| $\xi_{t\mid t-1}(s)$ | Xác suất **dự báo** trạng thái | $[0,1]$ |
| $\xi_{t\mid t}(s)$ | Xác suất **lọc** trạng thái | $[0,1]$ |
| $\xi_{t\mid T}(s)$ | Xác suất **làm trơn** trạng thái | $[0,1]$ |
| $\bar\mu_t$ | Trung bình thu gọn | $\mathbb{R}$ |
| $\bar h_t$ | Phương sai thu gọn | $>0$ |
| $\bar\varepsilon_t$ | Phần dư thu gọn | $\mathbb{R}$ |
| $\boldsymbol{\theta}$ | Vector tham số ở miền hợp lệ | $\Theta$ |
| $\tilde{\boldsymbol{\theta}}$ | Vector tham số **không ràng buộc** | $\mathbb{R}^d$ |
| $d$ | Số chiều của vector tham số | 13 với $k=2$ |
| $\ell_T(\boldsymbol{\theta})$ | Log-hàm hợp lý | $\mathbb{R}$ |
| $\mathcal{I}_{t}$ | Tập thông tin đến thời điểm $t$ | — |
| $\mathcal{D}_T$ | Toàn bộ dữ liệu quan sát | — |

Ba loại xác suất trạng thái rất dễ nhầm và được dùng cho ba mục đích khác nhau:

- $\xi_{t\mid t-1}$ dùng **thông tin đến $t-1$**. Đây là xác suất duy nhất được
  phép dùng khi dự báo $r_t$ hoặc phương sai của nó.
- $\xi_{t\mid t}$ dùng thêm chính $r_t$. Dùng để cập nhật đệ quy bộ lọc.
- $\xi_{t\mid T}$ dùng toàn bộ mẫu. Chỉ dùng để mô tả lịch sử, không dùng để dự báo.

---

## 3. Dữ liệu và biến số

### 3.1 Từ dữ liệu phút sang chuỗi ngày

Dữ liệu gốc là nến phút của VN30F1M (`ohlc_export.csv`, khoảng 525.635 dòng từ
06/11/2017 đến 17/07/2026, tương ứng 2.171 phiên). Quy trình gộp:

1. Ghép `TRADING_DATE` và `TRADING_TIME` thành mốc thời gian, sắp xếp tăng dần.
2. Gộp theo ngày: giá mở cửa lấy nến đầu phiên, giá cao nhất và thấp nhất lấy
   cực trị trong phiên, giá đóng cửa lấy nến cuối phiên, khối lượng lấy tổng.
3. Tính lợi suất logarit ngày với hệ số quy đổi $c=100$:

$$r_t = c\,(\log P_t - \log P_{t-1}), \qquad c = 100.$$

Nhân với 100 đưa lợi suất về đơn vị phần trăm. Đây là lựa chọn kỹ thuật quan
trọng: nếu giữ đơn vị thập phân, phương sai có bậc độ lớn $10^{-4}$ và thuật toán
tối ưu hóa dễ gặp vấn đề số học.

### 3.2 Biến động thực tế

Biến động thực tế được tính **thật** từ lợi suất nội phiên, không dùng đại diện
$r_t^2$:

$$RV_t = c^2\sum_{j=1}^{M_t} r_{t,j}^2,$$

trong đó $r_{t,j}$ là lợi suất logarit giữa hai nến phút liên tiếp **trong cùng
phiên $t$**, và $M_t$ là số lợi suất nội phiên khả dụng. Nến đầu mỗi phiên không
có nến liền trước trong cùng ngày nên không tham gia tổng — cách này tránh đưa
biến động qua đêm vào $RV_t$.

Hệ số $c^2$ bảo đảm $RV_t$ và $h_t$ **cùng đơn vị** (phần trăm bình phương). Đây
là điều kiện bắt buộc để các hàm mất mát ở [Mục 9](#9-đánh-giá-ngoài-mẫu) có ý nghĩa.

### 3.3 Biến ngoại sinh

Vector $\mathbf{X}_{t-1}$ chỉ chứa thông tin đã biết trước thời điểm dự báo $r_t$:

$$\mathbf{X}_{t-1} = \big(\log(1+VOL_{t-1}),\; \log(RV_{t-1}+\epsilon)\big)^\top,
\qquad \epsilon = 10^{-6}.$$

Ba biến ứng viên khác trong thiết kế ban đầu — chênh lệch giá so với chỉ số cơ sở
$BASIS_{t-1}$, biến giả ngày đáo hạn $EXP_t$ và biến giả ngày chuyển hợp đồng
$ROLL_t$ — **chưa dùng được**, vì nguồn dữ liệu là một chuỗi liên tục duy nhất do
nhà cung cấp tự tổng hợp, không kèm mã hợp đồng thành phần, lịch đáo hạn hay dữ
liệu chỉ số VN30 cơ sở.

Hai biến giả lịch, nếu có, sẽ **không bị dịch một kỳ** vì chúng được biết chắc
trước; các biến liên tục thì bị dịch một kỳ.

### 3.4 Chuẩn hóa và chia mẫu

Mỗi biến liên tục được chuẩn hóa

$$X^{\text{chuẩn}}_{j} = \frac{X_j - \bar X_j^{\text{huấn luyện}}}{s_j^{\text{huấn luyện}}},$$

với trung bình và độ lệch chuẩn **chỉ tính trên tập huấn luyện**. Dùng thống kê
của toàn mẫu sẽ để thông tin tương lai rò rỉ vào giai đoạn dự báo.

Mẫu được chia theo thời gian, không chia ngẫu nhiên: 80% quan sát đầu làm tập
huấn luyện ($T_{\text{huấn luyện}} \approx 1.735$), 20% cuối làm tập kiểm định.

### 3.5 Thiết kế ngoài mẫu

Do chi phí tính toán của MCMC, nghiên cứu áp dụng thiết kế đơn giản hóa so với
tái ước lượng hằng ngày: tham số được ước lượng **một lần duy nhất** trên tập
huấn luyện, sau đó bộ lọc tiến chạy liên tục qua tập kiểm định với **tham số cố
định**, chỉ cập nhật xác suất trạng thái và phương sai có điều kiện theo dữ liệu
mới. Không có bước tái tối ưu tham số nào trong tập kiểm định.

Thiết kế này vẫn là dự báo ngoài mẫu hợp lệ — tại mỗi $t$ trong tập kiểm định,
dự báo chỉ dùng thông tin đến $t-1$ — nhưng không đo được khả năng thích ứng của
mô hình khi được ước lượng lại định kỳ.

---

## 4. Mô hình

### 4.1 Phân phối Student-t chuẩn hóa

Sai số chuẩn hóa $z_t$ theo phân phối Student-t đã được chuẩn hóa để có phương
sai bằng 1, ký hiệu $t^*_\nu$, với mật độ

$$f_{t^*}(z\mid\nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}
{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\pi(\nu-2)}}
\left(1+\frac{z^2}{\nu-2}\right)^{-\frac{\nu+1}{2}}, \qquad \nu>2.$$

Điểm khác biệt so với Student-t thông thường nằm ở $\sqrt{\pi(\nu-2)}$ thay vì
$\sqrt{\pi\nu}$: phép chuẩn hóa này bảo đảm $\mathbb{E}[z_t]=0$ và
$\mathrm{Var}[z_t]=1$ **với mọi** $\nu>2$. Nhờ đó $h_{t,s}$ giữ đúng nghĩa là
phương sai có điều kiện, không bị lẫn với tham số tỷ lệ của phân phối.

Ràng buộc $\nu>2$ là bắt buộc: khi $\nu\le 2$ phương sai không tồn tại.

Trong cài đặt, log-mật độ được tính trực tiếp để tránh mất chính xác số học:

$$\log f_{t^*}(z\mid\nu) = \log\Gamma\!\left(\tfrac{\nu+1}{2}\right)
-\log\Gamma\!\left(\tfrac{\nu}{2}\right)
-\tfrac{1}{2}\log(\nu-2) -\tfrac{1}{2}\log\pi
-\tfrac{\nu+1}{2}\log\!\left(1+\tfrac{z^2}{\nu-2}\right).$$

### 4.2 Từ GARCH đến GARCH-X

Mô hình GARCH(1,1) một trạng thái:

$$h_t = \omega + \alpha\varepsilon_{t-1}^2 + \beta h_{t-1}.$$

Hệ số $\alpha$ đo phản ứng ngắn hạn với thông tin mới, $\beta$ đo tính dai dẳng,
tổng $\alpha+\beta$ phản ánh tốc độ suy giảm ảnh hưởng của một cú sốc.

Mô hình GARCH-X bổ sung thông tin ngoại sinh vào phương trình phương sai:

$$h_t = \omega + \alpha\varepsilon_{t-1}^2 + \beta h_{t-1}
+ \boldsymbol{\gamma}^\top\mathbf{X}_{t-1}.$$

Bổ sung này tạo ra một vấn đề: khi $\mathbf{X}_{t-1}$ đã chuẩn hóa quanh 0, thành
phần $\boldsymbol{\gamma}^\top\mathbf{X}_{t-1}$ có thể âm, nên các điều kiện
$\omega>0,\alpha\ge 0,\beta\ge 0$ **không còn đủ** để bảo đảm $h_t>0$. Cách xử lý
được cài đặt ở [Mục 6.2](#62-bảo-đảm-phương-sai-dương).

### 4.3 Chuỗi Markov ẩn

Trạng thái $S_t\in\{1,2\}$ là chuỗi Markov bậc một:

$$\Pr(S_t=j\mid S_{t-1}=i, S_{t-2},\dots) = \Pr(S_t=j\mid S_{t-1}=i) = p_{ij},$$

$$\mathbf{P}=\begin{pmatrix} p_{11} & p_{12}\\ p_{21} & p_{22}\end{pmatrix},
\qquad p_{ij}\ge 0,\qquad p_{i1}+p_{i2}=1.$$

Vì mỗi hàng cộng bằng 1, chỉ cần hai tham số tự do $p_{11}$ và $p_{22}$; hai giá
trị còn lại suy ra từ $p_{12}=1-p_{11}$ và $p_{21}=1-p_{22}$.

Thời gian tồn tại kỳ vọng của trạng thái $s$:

$$d_s = \frac{1}{1-p_{ss}}.$$

Ví dụ $p_{11}=0{,}98$ cho $d_1=50$ phiên. Đại lượng này giúp diễn giải kết quả
theo ngôn ngữ thị trường: trạng thái kéo dài bao lâu trước khi chuyển.

Phân phối dừng $\boldsymbol{\pi}$ thỏa $\boldsymbol{\pi}^\top =
\boldsymbol{\pi}^\top\mathbf{P}$ với $\pi_1+\pi_2=1$. Với ma trận $2\times 2$,
nghiệm có dạng đóng:

$$\pi_1 = \frac{p_{21}}{p_{12}+p_{21}}, \qquad \pi_2 = 1-\pi_1.$$

### 4.4 Vấn đề phụ thuộc đường đi và cơ chế thu gọn Gray

Nếu ghép trực tiếp GARCH với chuyển trạng thái, ta có

$$h_t(\mathbf{S}_t) = \omega_{S_t} + \alpha_{S_t}\varepsilon_{t-1}^2
+ \beta_{S_t}h_{t-1}(\mathbf{S}_{t-1}),$$

trong đó $\mathbf{S}_t=(S_1,\dots,S_t)$. Vì $h_{t-1}$ lại phụ thuộc $h_{t-2}$,
và cứ thế lùi về đầu chuỗi, phương sai tại $t$ phụ thuộc **toàn bộ đường đi**
trạng thái. Số đường đi khả dĩ là $2^t$. Với $T=1.735$, đó là $2^{1735}$ nhánh —
không thể tính được.

Giải pháp Gray: thay $h_{t-1}$ phụ thuộc đường đi bằng một **moment thu gọn**,
tức trung bình có trọng số theo xác suất lọc. Trung bình thu gọn:

$$\bar\mu_{t-1} = \sum_{i=1}^{2}\xi_{t-1\mid t-1}(i)\,\mu_i,$$

phương sai thu gọn (theo công thức phương sai của hỗn hợp):

$$\bar h_{t-1} = \sum_{i=1}^{2}\xi_{t-1\mid t-1}(i)\left[h_{t-1,i}+\mu_i^2\right]
- \bar\mu_{t-1}^2,$$

và phần dư thu gọn:

$$\bar\varepsilon_{t-1} = r_{t-1}-\bar\mu_{t-1}.$$

Công thức phương sai hỗn hợp gồm hai phần: trung bình của các phương sai thành
phần, cộng với độ phân tán giữa các trung bình thành phần. Số hạng
$-\bar\mu_{t-1}^2$ chính là phần trừ đi bình phương trung bình chung.

Sau khi thu gọn, tại mỗi $t$ ta chỉ cần lưu **hai** giá trị $h_{t,1}, h_{t,2}$
thay vì $2^t$ nhánh. Cái giá phải trả là hàm hợp lý trở thành một xấp xỉ, chứ
không còn chính xác tuyệt đối.

### 4.5 Mô hình chính

$$
\begin{aligned}
r_t &= \mu_{S_t}+\varepsilon_t,\\
\varepsilon_t &= \sqrt{h_{t,S_t}}\;z_t,\\
z_t &\sim t^*_\nu, \qquad \nu>2,\\
h_{t,s} &= \omega_s + \alpha_s\bar\varepsilon_{t-1}^2 + \beta_s\bar h_{t-1}
+ \boldsymbol{\gamma}_s^\top\mathbf{X}_{t-1}, \qquad s\in\{1,2\}.
\end{aligned}
$$

Điểm cần chú ý ở phương trình cuối: vế trái có chỉ số trạng thái $s$, nhưng vế
phải dùng $\bar\varepsilon_{t-1}$ và $\bar h_{t-1}$ **không có** chỉ số trạng
thái. Đó chính là cơ chế thu gọn: hai trạng thái có tham số riêng nhưng cùng chia
sẻ một lịch sử đã được thu gọn.

Mô hình mở rộng (ký hiệu R1) thay bậc tự do chung bằng bậc tự do theo trạng thái:

$$z_t\mid S_t=s \sim t^*_{\nu_s}, \qquad \nu_s>2,$$

cho phép trả lời trực tiếp RQ6.

### 4.6 Nhận dạng nhãn trạng thái

Mô hình chuyển trạng thái có một tính chất khó chịu: đổi tên trạng thái 1 thành 2
và ngược lại (kèm hoán vị tham số tương ứng) cho **đúng cùng một giá trị hàm hợp
lý**. Hệ quả là các chuỗi MCMC khác nhau có thể "gọi tên" trạng thái ngược nhau,
và việc lấy trung bình hậu nghiệm sẽ trộn lẫn hai trạng thái thành vô nghĩa.

Quy tắc nhận dạng được áp dụng sau ước lượng:

$$\mathbb{E}(h_t\mid S_t=1) < \mathbb{E}(h_t\mid S_t=2),$$

tức trạng thái 1 luôn là trạng thái **biến động thấp**, trạng thái 2 là trạng thái
**biến động cao**. Trong cài đặt, kỳ vọng được ước lượng bằng trung bình theo thời
gian của đường $h_{t,s}$; nếu điều kiện bị vi phạm, toàn bộ tham số được hoán vị:

$$\mu \leftarrow \mu[\sigma],\quad \omega \leftarrow \omega[\sigma],\quad
\alpha \leftarrow \alpha[\sigma],\quad \beta \leftarrow \beta[\sigma],\quad
\boldsymbol{\gamma} \leftarrow \boldsymbol{\gamma}[\sigma],\quad
\mathbf{P} \leftarrow \mathbf{P}[\sigma,\sigma],$$

với $\sigma=(2,1)$ là phép hoán vị. Lưu ý ma trận $\mathbf{P}$ phải hoán vị **cả
hàng lẫn cột**.

---

## 5. Bộ lọc tiến và hàm hợp lý

Đây là phần lõi của toàn bộ dự án. Hàm `ms_garch_forward_filter` là **nơi duy
nhất** tính log-hàm hợp lý; cả ba phương pháp suy diễn đều gọi lại đúng hàm này.

### 5.1 Điều kiện khởi tạo

$$\xi_{0\mid 0} = \boldsymbol{\pi}(\mathbf{P}), \qquad
h_{0,1}=h_{0,2}=\widehat{\mathrm{Var}}(r),$$

$$\bar\mu_0 = \sum_i \xi_{0\mid 0}(i)\mu_i, \qquad
\bar h_0 = \sum_i \xi_{0\mid 0}(i)\left[h_{0,i}+\mu_i^2\right]-\bar\mu_0^2,
\qquad \bar\varepsilon_0 = r_0-\bar\mu_0.$$

Quan sát đầu tiên $r_0$ được dùng làm điều kiện đầu, **không** tính vào hàm hợp
lý. Tổng log-hợp-lý chạy từ $t=1$. Cả $\boldsymbol{\pi}$ và
$\widehat{\mathrm{Var}}(r)$ đều được tách khỏi đồ thị đạo hàm để điều kiện khởi
tạo không tạo thêm đường lan truyền gradient.

### 5.2 Vòng lặp đệ quy

Với $t=1,\dots,T-1$, thực hiện năm bước sau.

**Bước 1 — Dự báo trạng thái.**

$$\xi_{t\mid t-1}(j) = \sum_{i=1}^{2}p_{ij}\,\xi_{t-1\mid t-1}(i),
\qquad\text{tức}\qquad
\boldsymbol{\xi}_{t\mid t-1}^\top = \boldsymbol{\xi}_{t-1\mid t-1}^\top\mathbf{P}.$$

**Bước 2 — Cập nhật phương sai theo từng trạng thái.**

$$h_{t,s}^{\text{thô}} = \omega_s + \alpha_s\bar\varepsilon_{t-1}^2
+ \beta_s\bar h_{t-1} + \boldsymbol{\gamma}_s^\top\mathbf{X}_{t-1},
\qquad
h_{t,s} = \max\!\left(h_{t,s}^{\text{thô}},\, \epsilon_{\text{sàn}}\right),$$

với $\epsilon_{\text{sàn}}=10^{-8}$.

**Bước 3 — Mật độ có điều kiện theo từng trạng thái.**

$$z_{t,s} = \frac{r_t-\mu_s}{\sqrt{h_{t,s}}}, \qquad
\log f_s(r_t) = \log f_{t^*}(z_{t,s}\mid\nu) - \tfrac{1}{2}\log h_{t,s}.$$

Số hạng $-\frac{1}{2}\log h_{t,s}$ là Jacobian của phép đổi biến từ $z$ sang $r$.
Bỏ sót số hạng này là lỗi kinh điển khi cài đặt mô hình GARCH.

**Bước 4 — Đóng góp vào hàm hợp lý và cập nhật xác suất lọc.**

$$\ell_t = \log\left[\sum_{s=1}^{2}\xi_{t\mid t-1}(s)f_s(r_t)\right]
= \operatorname{logsumexp}_s\left[\log\xi_{t\mid t-1}(s)+\log f_s(r_t)\right],$$

$$\xi_{t\mid t}(s) = \exp\left[\log\xi_{t\mid t-1}(s)+\log f_s(r_t)-\ell_t\right].$$

Toàn bộ phép tính diễn ra trên thang logarit và dùng `logsumexp`. Nhân trực tiếp
các mật độ rồi mới lấy logarit sẽ tràn số dưới khi mật độ rất nhỏ.

**Bước 5 — Thu gọn cho bước kế tiếp.**

$$\bar\mu_t = \sum_s\xi_{t\mid t}(s)\mu_s, \qquad
\bar h_t = \sum_s\xi_{t\mid t}(s)\left[h_{t,s}+\mu_s^2\right]-\bar\mu_t^2,
\qquad \bar\varepsilon_t = r_t-\bar\mu_t.$$

### 5.3 Hàm hợp lý

$$\ell_T(\boldsymbol{\theta}) = \sum_{t=1}^{T}\log\left[
\sum_{s=1}^{2}\xi_{t\mid t-1}(s)\,f_s(r_t\mid\mathcal{I}_{t-1},\boldsymbol{\theta})\right].$$

**Vì sao vòng lặp này không thể vector hóa**: $h_t$ phụ thuộc $\bar h_{t-1}$, mà
$\bar h_{t-1}$ lại phụ thuộc $\xi_{t-1\mid t-1}$, tức phụ thuộc chính $r_{t-1}$
đã quan sát. Đây là đệ quy thật sự theo thời gian, không phải phép biến đổi song
song được. Hệ quả trực tiếp là toàn bộ chi phí tính toán ở [Mục 13](#13-chi-phí-tính-toán).

### 5.4 Bộ làm trơn Kim

Sau khi chạy xong bộ lọc, xác suất làm trơn được tính lùi từ $t=T-1$ về $t=0$:

$$\xi_{t\mid T}(i) = \xi_{t\mid t}(i)\sum_{j=1}^{2}
\frac{p_{ij}\,\xi_{t+1\mid T}(j)}{\xi_{t+1\mid t}(j)},
\qquad \xi_{T\mid T} \text{ lấy từ bộ lọc.}$$

Xác suất làm trơn dùng cả thông tin sau thời điểm $t$, nên chỉ phù hợp để mô tả
lịch sử — **không** được dùng cho dự báo hay đánh giá ngoài mẫu.

---

## 6. Tham số hóa

### 6.1 Biến đổi sang không gian không ràng buộc

Thuật toán tối ưu hóa và NUTS đều làm việc trên không gian không ràng buộc
$\tilde{\boldsymbol{\theta}}\in\mathbb{R}^d$. Các phép biến đổi:

| Tham số | Ràng buộc | Biến đổi | Nghịch đảo |
|---|---|---|---|
| $\mu_s$ | không | $\mu_s=\tilde\mu_s$ | — |
| $\omega_s$ | $>0$ | $\omega_s=\exp(\tilde\omega_s)$ | $\tilde\omega_s=\log\omega_s$ |
| $\alpha_s$ | $\ge 0$ | $\alpha_s=\operatorname{softplus}(\eta_{\alpha,s})$ | $\eta=\log(e^{\alpha}-1)$ |
| $\beta_s$ | $\ge 0$ | $\beta_s=\operatorname{softplus}(\eta_{\beta,s})$ | $\eta=\log(e^{\beta}-1)$ |
| $\boldsymbol{\gamma}_s$ | không | giữ nguyên | — |
| $p_{ss}$ | $(0,1)$ | $p_{ss}=\dfrac{1}{1+e^{-\eta_s}}$ | $\eta_s=\log\dfrac{p_{ss}}{1-p_{ss}}$ |
| $\nu$ | $>2$ | $\nu=2+\operatorname{softplus}(\eta_\nu)$ | $\eta_\nu=\log(e^{\nu-2}-1)$ |

trong đó $\operatorname{softplus}(x)=\log(1+e^x)$.

Vector không ràng buộc được xếp theo thứ tự cố định:

$$\tilde{\boldsymbol{\theta}} = \big(
\mu_1,\mu_2,\;
\tilde\omega_1,\tilde\omega_2,\;
\eta_{\alpha,1},\eta_{\alpha,2},\;
\eta_{\beta,1},\eta_{\beta,2},\;
\boldsymbol{\gamma}_1,\boldsymbol{\gamma}_2,\;
\eta_{1},\eta_{2},\;
\eta_\nu\big)^\top.$$

Số chiều: $d = 8 + 2k + 2 + n_\nu$, với $n_\nu=1$ (mô hình chính) hoặc $2$ (R1).
Với $k=2$: $d=13$.

### 6.2 Bảo đảm phương sai dương

Vì $\boldsymbol{\gamma}_s^\top\mathbf{X}_{t-1}$ có thể âm, cần một lớp bảo vệ:

- **Chế độ `clamp`** (mặc định): $h_{t,s}=\max(h_{t,s}^{\text{thô}}, 10^{-8})$.
  Giữ nguyên dạng cộng của phương trình phương sai, chỉ cắt ở đáy. Khả vi hầu
  khắp nơi, phù hợp khi $\boldsymbol{\gamma}^\top\mathbf{X}$ hiếm khi âm mạnh.
- **Chế độ `softplus`**: $h_{t,s}=\operatorname{softplus}(h^{\text{thô}}_{t,s}-\epsilon)+\epsilon$.
  Bảo đảm dương tuyệt đối ở mọi điểm, nhưng $h_{t,s}$ không còn tuyến tính theo
  các thành phần cộng, tức lệch khỏi dạng chuẩn của phương trình phương sai.

Toàn bộ tính toán dùng `float64`. Log-hợp-lý của MS-GARCH nhạy với sai số làm
tròn, đặc biệt ở phép `logsumexp` và tỷ lệ phương sai giữa hai trạng thái.

---

## 7. Ba phương pháp suy diễn

### 7.1 Ước lượng hợp lý cực đại

Mục tiêu:

$$\widehat{\boldsymbol{\theta}}_{\text{MLE}}
= \arg\max_{\boldsymbol{\theta}\in\Theta}\ \ell_T(\boldsymbol{\theta}).$$

Vì hàm hợp lý MS-GARCH có thể đa đỉnh, quy trình dùng **nhiều điểm khởi tạo** và
**hai giai đoạn tối ưu**:

**Giai đoạn 1 — Adam.** Thuật toán bậc nhất với tốc độ học $0{,}02$, chạy 500
bước, có cắt chuẩn gradient ở ngưỡng 50. Adam bền với điểm khởi tạo tệ và độ dốc
lớn, nhiệm vụ của nó là kéo tham số từ vùng vô lý về gần nghiệm.

**Giai đoạn 2 — L-BFGS.** Thuật toán tựa-Newton, xấp xỉ độ cong của hàm mục tiêu,
dùng tìm kiếm đường thẳng Wolfe mạnh, tối đa 250 vòng. Adam tiến nhanh nhưng dao
động quanh nghiệm; L-BFGS tinh chỉnh để đạt độ chính xác cao. Bước này **bắt
buộc** vì chuẩn gradient tại nghiệm và ma trận Hessian chỉ có ý nghĩa khi thực sự
ở điểm cực đại.

Mỗi điểm khởi tạo ghi lại bốn chẩn đoán:

| Chẩn đoán | Định nghĩa |
|---|---|
| Log-hợp-lý | $\ell_T$ tại nghiệm |
| Hội tụ | $\lVert\nabla\ell_T\rVert < 1$ |
| Chuẩn gradient | $\lVert\nabla\ell_T\rVert$ trên thang không ràng buộc |
| Gần biên | $p_{ss}>0{,}995$ hoặc $\alpha_s+\beta_s>0{,}995$ hoặc $\nu<2{,}05$ |

So sánh log-hợp-lý giữa các điểm khởi tạo là cách kiểm tra tính đa đỉnh: nếu tất
cả hội tụ về cùng một giá trị, bằng chứng nghiêng về đơn đỉnh.

**Tiêu chí thông tin.** Với $k_{\text{th}}$ là số tham số tự do:

$$\mathrm{AIC} = -2\ell_T(\widehat{\boldsymbol{\theta}})+2k_{\text{th}},
\qquad
\mathrm{BIC} = -2\ell_T(\widehat{\boldsymbol{\theta}})+k_{\text{th}}\log T.$$

**Sai số chuẩn.** Vì tối ưu hóa diễn ra trên không gian không ràng buộc, ma trận
Hessian thô **không** dùng trực tiếp làm sai số chuẩn cho tham số ở thang tự
nhiên. Quy trình gồm hai bước.

Bước 1, hiệp phương sai trên thang không ràng buộc từ nghịch đảo Hessian quan sát:

$$\widehat{\boldsymbol{\Sigma}}_{\tilde\theta}
= \left[-\nabla^2\ell_T(\tilde{\boldsymbol{\theta}})\right]^{-1}.$$

Bước 2, phương pháp delta để chuyển sang thang tự nhiên:

$$\widehat{\boldsymbol{\Sigma}}_{\theta} = \mathbf{J}\,
\widehat{\boldsymbol{\Sigma}}_{\tilde\theta}\,\mathbf{J}^\top,
\qquad
\mathbf{J} = \frac{\partial\boldsymbol{\theta}}{\partial\tilde{\boldsymbol{\theta}}},
\qquad
\mathrm{SE}(\theta_j)=\sqrt{\left[\widehat{\boldsymbol{\Sigma}}_{\theta}\right]_{jj}}.$$

Nếu Hessian không xác định dương, quy trình trả về giá trị rỗng thay vì báo cáo
sai số chuẩn không hợp lệ.

### 7.2 Suy diễn Bayes: phân phối tiền nghiệm

Theo định lý Bayes,

$$p(\boldsymbol{\theta}\mid\mathcal{D}_T)\ \propto\
\exp\{\ell_T(\boldsymbol{\theta})\}\;p(\boldsymbol{\theta}).$$

Cấu trúc tiền nghiệm được cài đặt (đặt trực tiếp trên miền hợp lệ, để thư viện
tự xử lý phép biến đổi sang không gian không ràng buộc):

| Tham số | Tiền nghiệm | Giá trị mặc định |
|---|---|---|
| $\mu_s$ | $\mathcal{N}(0,\sigma_\mu^2)$ | $\sigma_\mu=1$ |
| $\omega_s$ | $\mathrm{LogNormal}(m_\omega,\sigma_\omega^2)$ | $m_\omega=0,\ \sigma_\omega=1$ |
| $\alpha_s$ | $\mathrm{Beta}(a,a)$ trên $(0,1)$ | $a=2$ |
| $\beta_s$ | $\mathrm{Beta}(a,a)$ trên $(0,1)$ | $a=2$ |
| $\boldsymbol{\gamma}_s$ | $\mathcal{N}(\mathbf{0},\sigma_\gamma^2\mathbf{I})$ | $\sigma_\gamma=1$ |
| $p_{11}, p_{22}$ | $\mathrm{Beta}(a_{\text{giữ}}, a_{\text{chuyển}})$ | $(8,\,2)$ |
| $\nu-2$ | $\mathrm{Gamma}(a_\nu,b_\nu)$ | $(2,\ 0{,}2)$, trung bình 10 |

Ba ghi chú về lựa chọn này:

- Tiền nghiệm $\mathrm{LogNormal}$ cho $\omega_s$ tương đương
  $\log\omega_s\sim\mathcal{N}$, đúng như đặc tả trong bản thảo.
- Với hai trạng thái, $\mathrm{Dirichlet}(a_{s1},a_{s2})$ trên một hàng của
  $\mathbf{P}$ **chính là** $\mathrm{Beta}(a_{s1},a_{s2})$ trên $p_{ss}$. Tham số
  $(8,2)$ tương ứng kỳ vọng $p_{ss}=0{,}8$, phản ánh niềm tin rằng trạng thái có
  tính dai dẳng.
- Tiền nghiệm $\mathrm{Beta}(2,2)$ cho $\alpha_s,\beta_s$ là **lựa chọn của cài
  đặt này**, vì bản thảo không đặc tả tiền nghiệm riêng cho hai tham số đó. Đây
  là tiền nghiệm yếu, đối xứng, ưu tiên nhẹ các giá trị giữa khoảng $(0,1)$.

### 7.3 Variational Bayes

VB thay bài toán tích phân bằng bài toán tối ưu: chọn một họ phân phối
$\mathcal{Q}$ và tìm phần tử gần hậu nghiệm nhất theo phân kỳ Kullback–Leibler:

$$q^*(\boldsymbol{\theta})
= \arg\min_{q\in\mathcal{Q}}\ \mathrm{KL}\!\left(q(\boldsymbol{\theta})\,\Vert\,
p(\boldsymbol{\theta}\mid\mathcal{D}_T)\right).$$

Vì hậu nghiệm chưa biết hằng số chuẩn hóa, bài toán được viết lại thành tối đa
hóa cận dưới bằng chứng:

$$\mathrm{ELBO}(q)
= \mathbb{E}_q\!\left[\log p(\mathcal{D}_T,\boldsymbol{\theta})\right]
- \mathbb{E}_q\!\left[\log q(\boldsymbol{\theta})\right]
= \log p(\mathcal{D}_T) - \mathrm{KL}\!\left(q\,\Vert\,p(\cdot\mid\mathcal{D}_T)\right).$$

Vì $\log p(\mathcal{D}_T)$ không phụ thuộc $q$, tối đa hóa ELBO tương đương tối
thiểu hóa KL. ELBO cũng là cận dưới của log bằng chứng, nên giá trị ELBO cuối
cùng là một chỉ báo hữu ích.

Hai họ xấp xỉ được cài đặt, cả hai đặt trên không gian không ràng buộc:

$$\text{Mean-field:}\quad q(\tilde{\boldsymbol{\theta}})=\prod_{j=1}^{d}
\mathcal{N}\!\left(\tilde\theta_j\mid m_j, s_j^2\right),
\qquad
\text{Gaussian đầy đủ:}\quad q(\tilde{\boldsymbol{\theta}})=
\mathcal{N}\!\left(\mathbf{m},\mathbf{L}\mathbf{L}^\top\right).$$

Mean-field nhanh (chỉ $2d$ tham số) nhưng **giả định các tham số độc lập**, nên
thường đánh giá thấp phương sai hậu nghiệm khi các tham số có tương quan mạnh.
Gaussian đầy đủ có $d+d(d+1)/2$ tham số, nắm bắt được tương quan nhưng tối ưu hóa
khó hơn. So sánh hai họ này chính là nội dung của RQ3.

Cấu hình: thuật toán Adam với tốc độ học $0{,}01$, 3.000 bước, ước lượng ELBO
bằng một mẫu mỗi bước. Sau khi hội tụ, lấy 1.000 mẫu từ $q^*$ để tóm tắt hậu
nghiệm. Giá trị ELBO cuối được lấy trung bình 50 bước cuối để giảm nhiễu.

### 7.4 MCMC bằng NUTS

MCMC sinh dãy mẫu $\boldsymbol{\theta}^{(1)},\dots,\boldsymbol{\theta}^{(M)}$ hội
tụ về hậu nghiệm. Chiến lược được chọn: **lấy tổng trạng thái rời rạc bằng bộ lọc
tiến**, sau đó NUTS chỉ cần lấy mẫu trên các tham số liên tục. Điều này cần thiết
vì NUTS dựa trên gradient nên không xử lý được biến rời rạc.

**Cơ chế NUTS.** Coi $-\log$ hậu nghiệm như một mặt địa hình. Thuật toán gán cho
tham số một động lượng ngẫu nhiên rồi mô phỏng chuyển động theo phương trình
Hamilton bằng phép tích phân **leapfrog**. Mỗi bước leapfrog cần **một lần tính
gradient của log hậu nghiệm**, tức một lần chạy toàn bộ bộ lọc tiến.

Khác với HMC cổ điển vốn phải chọn tay số bước, NUTS tự nhân đôi độ dài quỹ đạo
— 1, 2, 4, 8, … bước — và dừng khi quỹ đạo bắt đầu quay đầu về điểm xuất phát.
Với độ sâu cây tối đa $D$, số bước leapfrog mỗi mẫu tối đa là $2^{D}-1$.

Cấu hình: `target_accept_prob` $=0{,}85$, 4 chuỗi, 250 vòng khởi động (warmup),
400 mẫu mỗi chuỗi. Giai đoạn khởi động là lúc thuật toán thích nghi bước tích
phân và ma trận khối lượng — **cắt ngắn giai đoạn này sẽ làm mẫu bị lệch**, khác
hẳn với việc giảm độ sâu cây (chỉ làm giảm hiệu quả).

**Chẩn đoán hội tụ.**

- $\widehat R$: so sánh phương sai giữa các chuỗi với phương sai trong từng
  chuỗi. Giá trị mong muốn nhỏ hơn $1{,}01$.
- ESS (cỡ mẫu hiệu dụng): số mẫu độc lập tương đương. Mẫu MCMC có tự tương quan
  nên ESS luôn nhỏ hơn số mẫu danh nghĩa.
- Số lần phân kỳ: dấu hiệu quỹ đạo leapfrog gặp vùng cong gắt. **Phân kỳ báo hiệu
  sai lệch thật sự**, khác với việc chạm trần độ sâu cây (chỉ là vấn đề hiệu quả).

### 7.5 Điều bảo đảm ba phương pháp so sánh được với nhau

Đây là yêu cầu cốt lõi của thiết kế:

- MLE gọi trực tiếp `ms_garch_forward_filter`.
- VB và MCMC cùng gọi một hàm mô hình duy nhất (`bayes_model.model`), và hàm đó
  gọi lại **đúng** `ms_garch_forward_filter` thông qua cơ chế cộng thêm log-hợp-lý
  vào mật độ chung.

Nhờ vậy, mọi chênh lệch kết quả giữa ba phương pháp phản ánh sự khác biệt của
**phương pháp suy diễn**, không phải của đặc tả mô hình.

---

## 8. Dự báo

### 8.1 Phương sai dự báo một bước

$$\widehat h_t = \mathrm{Var}(r_t\mid\mathcal{I}_{t-1})
= \sum_{s=1}^{2}\xi_{t\mid t-1}(s)\left[h_{t,s}+\mu_s^2\right]
- \left(\sum_{s=1}^{2}\xi_{t\mid t-1}(s)\mu_s\right)^{2}.$$

Điểm then chốt: công thức dùng $\xi_{t\mid t-1}$ (xác suất **dự báo**), không phải
$\xi_{t\mid t}$ (xác suất **lọc**). Dùng nhầm xác suất lọc sẽ để chính $r_t$ rò rỉ
vào dự báo của nó và làm sai lệch toàn bộ đánh giá ngoài mẫu.

### 8.2 Mật độ dự báo hỗn hợp

$$f(r_t\mid\mathcal{I}_{t-1}) = \sum_{s=1}^{2}\xi_{t\mid t-1}(s)\,
\frac{1}{\sqrt{h_{t,s}}}f_{t^*}\!\left(\frac{r_t-\mu_s}{\sqrt{h_{t,s}}}\;\middle|\;\nu\right).$$

Đây là hỗn hợp hai thành phần Student-t, không phải một Student-t đơn lẻ. Hàm phân
phối tích lũy tương ứng:

$$F(x\mid\mathcal{I}_{t-1}) = \sum_{s=1}^{2}\xi_{t\mid t-1}(s)\,
F_{t_\nu}\!\left(\frac{x-\mu_s}{\sqrt{h_{t,s}}\sqrt{(\nu-2)/\nu}}\right).$$

Hệ số $\sqrt{(\nu-2)/\nu}$ chuyển đổi giữa Student-t chuẩn hóa và Student-t thông
thường của thư viện thống kê.

### 8.3 Phân vị dự báo

Phân vị mức $\alpha$ không có dạng đóng với hỗn hợp, nên được giải bằng phương
pháp Brent trên phương trình đơn điệu:

$$q_{t,\alpha}: \quad F(q_{t,\alpha}\mid\mathcal{I}_{t-1}) - \alpha = 0,$$

trên khoảng tìm kiếm $\left[\min_s\mu_s - 20\sqrt{\max_s h_{t,s}},\
\max_s\mu_s + 20\sqrt{\max_s h_{t,s}}\right]$.

### 8.4 Dự báo nhiều bước

Với tầm dự báo lớn hơn 1, không có công thức đóng; phải mô phỏng đồng thời bốn
thành phần:

1. Trạng thái tương lai theo ma trận $\mathbf{P}$;
2. Sai số chuẩn hóa $z\sim t^*_\nu$;
3. Phương sai có điều kiện theo đệ quy;
4. Biến ngoại sinh tương lai (truyền vào theo kịch bản).

### 8.5 Dự báo có tính đến bất định tham số

Với MLE, dự báo dùng một bộ tham số điểm. Với VB và MCMC, dự báo được lấy trung
bình trên $M$ mẫu hậu nghiệm:

$$\widehat h_t^{\text{Bayes}} = \frac{1}{M}\sum_{m=1}^{M}\widehat h_t^{(m)},
\qquad
\log \widehat f_t^{\text{Bayes}}(r_t)
= \log\frac{1}{M}\sum_{m=1}^{M}f_t^{(m)}(r_t).$$

Chú ý mật độ được lấy trung bình **trên thang tự nhiên** rồi mới lấy logarit
(cài đặt bằng `logsumexp`), không phải lấy trung bình các log-mật độ. Đây chính
là điểm phân biệt RQ4: dự báo Bayes truyền bất định tham số vào phân phối dự báo,
còn dự báo điểm thì không.

---

## 9. Đánh giá ngoài mẫu

### 9.1 Hàm mất mát cho dự báo biến động

Với $N$ quan sát ngoài mẫu, so sánh $\widehat h_t$ với $RV_t$:

$$\mathrm{MAE} = \frac{1}{N}\sum_{t=1}^{N}\left|RV_t-\widehat h_t\right|,
\qquad
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{t=1}^{N}\left(RV_t-\widehat h_t\right)^2},$$

$$\mathrm{QLIKE} = \frac{1}{N}\sum_{t=1}^{N}
\left[\log\widehat h_t+\frac{RV_t}{\widehat h_t}\right].$$

Giá trị nhỏ hơn là tốt hơn. QLIKE được chọn làm tiêu chí chính vì nó **phù hợp**
theo nghĩa thống kê khi đại diện biến động có nhiễu: dùng $RV_t$ thay cho phương
sai thật không làm thay đổi thứ hạng kỳ vọng của các mô hình. MAE và RMSE không
có tính chất này.

### 9.2 Dự báo mật độ

Điểm log âm trung bình:

$$\mathrm{NLS} = -\frac{1}{N}\sum_{t=1}^{N}\log\widehat f_t(r_t).$$

Biến đổi tích phân xác suất:

$$u_t = \widehat F_t(r_t).$$

Nếu mật độ dự báo được hiệu chỉnh tốt, dãy $u_t$ xấp xỉ phân phối đều trên
$(0,1)$. Biểu đồ tần suất của $u_t$ lệch về hai biên cho thấy mô hình đánh giá
thấp rủi ro đuôi; dồn vào giữa cho thấy dự báo quá thận trọng.

CRPS cũng được cài đặt dưới dạng ước lượng từ mẫu:

$$\mathrm{CRPS}(F,r)\approx \frac{1}{n}\sum_{i}|x_i-r|
-\frac{1}{2n^2}\sum_{i}\sum_{j}|x_i-x_j|, \qquad x_i\sim F.$$

### 9.3 Kiểm định VaR

Giá trị chịu rủi ro ở mức $\alpha$ được báo cáo dưới dạng mức lỗ dương:

$$\mathrm{VaR}_{t,\alpha} = -q_{t,\alpha},
\qquad
I_{t,\alpha} = \mathbf{1}\{r_t < q_{t,\alpha}\}.$$

**Kiểm định Kupiec** kiểm tra tỷ lệ vi phạm vô điều kiện. Với $N$ quan sát và $x$
lần vi phạm, $\hat\pi = x/N$:

$$LR_{uc} = -2\log\frac{(1-\alpha)^{N-x}\alpha^{x}}
{(1-\hat\pi)^{N-x}\hat\pi^{x}}\ \sim\ \chi^2(1).$$

**Kiểm định Christoffersen** bổ sung kiểm tra tính độc lập. Gọi $n_{ij}$ là số
lần chuyển từ trạng thái vi phạm $i$ sang $j$, $\pi_{01}=n_{01}/(n_{00}+n_{01})$,
$\pi_{11}=n_{11}/(n_{10}+n_{11})$, $\pi=(n_{01}+n_{11})/\sum n_{ij}$:

$$LR_{ind} = -2\log\frac{(1-\pi)^{n_{00}+n_{10}}\pi^{n_{01}+n_{11}}}
{(1-\pi_{01})^{n_{00}}\pi_{01}^{n_{01}}(1-\pi_{11})^{n_{10}}\pi_{11}^{n_{11}}}
\ \sim\ \chi^2(1),$$

$$LR_{cc} = LR_{uc}+LR_{ind}\ \sim\ \chi^2(2).$$

Ý nghĩa thực tế: một mô hình có thể có đúng số lần vi phạm nhưng các vi phạm dồn
thành cụm trong giai đoạn căng thẳng — đó là rủi ro nghiêm trọng mà chỉ
$LR_{ind}$ phát hiện được.

### 9.4 Kiểm định Diebold–Mariano

So sánh độ chính xác dự báo giữa hai mô hình qua chênh lệch hàm mất mát
$d_t = L_{1,t}-L_{2,t}$:

$$DM = \frac{\bar d}{\sqrt{\widehat{\mathrm{Var}}(\bar d)}},
\qquad
\widehat{\mathrm{Var}}(\bar d)=\frac{1}{N}\left[\hat\gamma_0
+2\sum_{k=1}^{L}\left(1-\frac{k}{L+1}\right)\hat\gamma_k\right],$$

với $L=h-1$ và $\hat\gamma_k$ là hiệp phương sai trễ bậc $k$. Trọng số
$1-k/(L+1)$ là nhân Bartlett, bảo đảm ước lượng phương sai dài hạn không âm.

Hiệu chỉnh mẫu nhỏ Harvey–Leybourne–Newbold:

$$DM^* = DM\sqrt{\frac{N+1-2h+h(h-1)/N}{N}},$$

và giá trị $p$ tính từ phân phối $t(N-1)$. Quy ước dấu: $DM>0$ nghĩa là mô hình
thứ nhất có mất mát lớn hơn, tức **tệ hơn**.

---

## 10. Kiểm định thống kê ban đầu

Trước khi ước lượng, chuỗi lợi suất được kiểm định để xác nhận các giả định nền:

| Kiểm định | Giả thuyết gốc | Vai trò |
|---|---|---|
| ADF | Chuỗi có nghiệm đơn vị | Kiểm tra tính dừng |
| Phillips–Perron | Chuỗi có nghiệm đơn vị | Như ADF, hiệu chỉnh phi tham số |
| KPSS | Chuỗi **dừng** | Đảo chiều giả thuyết, tránh kết luận một chiều |
| Ljung–Box trên $r_t$ | Không có tự tương quan | Kiểm tra cấu trúc trung bình |
| Ljung–Box trên $r_t^2$ | Không có tự tương quan | Dấu hiệu tập trung biến động |
| ARCH-LM | Không có hiệu ứng ARCH | Biện minh cho mô hình GARCH |
| Jarque–Bera | Phân phối chuẩn | Biện minh cho Student-t |

Kết hợp ADF hoặc Phillips–Perron **với** KPSS là cách tránh diễn giải chỉ dựa
trên một giả thuyết gốc. Cần phân biệt rõ "bác bỏ giả thuyết gốc" với "không đủ
bằng chứng bác bỏ" — không dùng cụm "chấp nhận giả thuyết gốc".

Kết quả trên dữ liệu thật đã xác nhận đúng các giả định: độ nhọn khoảng 9,0 và
Jarque–Bera bác bỏ mạnh phân phối chuẩn; ARCH-LM có $p<0{,}001$; ADF và
Phillips–Perron bác bỏ nghiệm đơn vị trong khi KPSS không bác bỏ tính dừng.

---

## 11. Nghiên cứu mô phỏng

Mô phỏng cho phép đánh giá chất lượng ước lượng khi **biết trước tham số thật** —
điều không bao giờ có với dữ liệu thật.

### 11.1 Cơ chế sinh dữ liệu

Với tham số thật $\boldsymbol{\theta}_0$ và ma trận chuyển $\mathbf{P}_0$:

1. Sinh $S_1$ từ phân phối dừng của $\mathbf{P}_0$;
2. Sinh $S_t$ từ hàng $S_{t-1}$ của $\mathbf{P}_0$;
3. Sinh biến ngoại sinh (mặc định: nhiễu chuẩn độc lập);
4. Cập nhật phương sai **theo đường đi thật**:
   $h_t=\omega_{S_t}+\alpha_{S_t}\varepsilon_{t-1}^2+\beta_{S_t}h_{t-1}
   +\boldsymbol{\gamma}_{S_t}^\top\mathbf{X}_t$;
5. Sinh $z_t\sim t^*_{\nu_0}$ và $r_t=\mu_{S_t}+\sqrt{h_t}\,z_t$;
6. Bỏ giai đoạn khởi động (mặc định 200 quan sát đầu).

Khác biệt quan trọng ở bước 4: khi **sinh** dữ liệu, trạng thái đã biết nên dùng
được đường đi thật; khi **ước lượng**, trạng thái không quan sát được nên phải
dùng cơ chế thu gọn. Chênh lệch giữa hai cơ chế này chính là sai số xấp xỉ mà
nghiên cứu mô phỏng đo được.

Giá trị khởi tạo $h_0=\omega_s/\max(1-\alpha_s-\beta_s,\,0{,}05)$ là phương sai
không điều kiện của trạng thái.

### 11.2 Tám kịch bản

| Mã | Đặc điểm | Mục đích |
|---|---|---|
| S1 | Hai trạng thái phân biệt rõ, $\nu_0=8$, $p_{ss}$ cao | Trường hợp thuận lợi, làm chuẩn đối chiếu |
| S2 | Hai trạng thái gần nhau | Kiểm tra khi nhận dạng trạng thái yếu |
| S3 | $\nu_0=4$ (đuôi rất dày) | Kiểm tra khả năng ước lượng bậc tự do nhỏ |
| S4 | $\nu_0=30$ (gần phân phối chuẩn) | Kiểm tra khi Student-t gần như thừa |
| S5 | Trạng thái biến động cao hiếm | Kiểm tra khi một trạng thái ít quan sát |
| S6 | Cỡ mẫu nhỏ ($T=500$) | Ảnh hưởng của cỡ mẫu |
| S7 | Cỡ mẫu lớn ($T=5000$) | Kiểm tra tính nhất quán |
| S8 | Sai đặc tả (thêm tự tương quan trong trung bình) | Độ bền khi mô hình sai |

Ở S8, dữ liệu được sinh với $r_t^{\text{sai}} = r_t + \phi r_{t-1}$, $\phi=0{,}3$,
trong khi mô hình ước lượng vẫn giả định không có tự tương quan trong trung bình.

### 11.3 Tiêu chí đánh giá

Với $B$ lần lặp và tham số thật $\theta_0$:

$$\mathrm{Bias} = \frac{1}{B}\sum_{b=1}^{B}\left(\hat\theta^{(b)}-\theta_0\right),
\qquad
\mathrm{RMSE} = \sqrt{\frac{1}{B}\sum_{b=1}^{B}\left(\hat\theta^{(b)}-\theta_0\right)^2},$$

$$\mathrm{Coverage} = \frac{1}{B}\sum_{b=1}^{B}
\mathbf{1}\left\{\hat\theta^{(b)}_{2{,}5\%}\le\theta_0\le\hat\theta^{(b)}_{97{,}5\%}\right\}.$$

Coverage chỉ tính được cho VB và MCMC, vì chỉ hai phương pháp này cho khoảng tin
cậy trực tiếp từ phân phối hậu nghiệm. Giá trị lý tưởng là $0{,}95$; thấp hơn
đáng kể nghĩa là phương pháp đánh giá thấp bất định.

Độ chính xác nhận dạng trạng thái được đo bằng Brier score:

$$\mathrm{Brier} = \frac{1}{T}\sum_{t=1}^{T}\sum_{s=1}^{2}
\left[\widehat\xi_t(s)-\mathbf{1}\{S_t=s\}\right]^2.$$

Giá trị nhỏ hơn là tốt hơn; dự báo ngẫu nhiên $\widehat\xi=(0{,}5;0{,}5)$ cho
Brier $=0{,}5$.

---

## 12. Hệ thống mô hình

| Mã | Mô hình | Phân phối | Biến ngoại sinh | Chuyển trạng thái | Suy diễn |
|---|---|---|---|---|---|
| B1 | GARCH(1,1) | Chuẩn | Không | Không | MLE |
| B2 | GARCH(1,1) | Student-t | Không | Không | MLE |
| B3 | GARCH-X | Student-t | Có | Không | MLE |
| B4 | MS-GARCH | Chuẩn | Không | Có | MLE |
| **M1** | **MS-GARCH-X** | **Student-t** | **Có** | **Có** | **MLE** |
| M2 | MS-GARCH-X | Student-t | Có | Có | VB mean-field |
| M3 | MS-GARCH-X | Student-t | Có | Có | VB Gaussian đầy đủ |
| M4 | MS-GARCH-X | Student-t | Có | Có | MCMC (NUTS) |
| R1 | MS-GARCH-X, $\nu_s$ theo trạng thái | Student-t | Có | Có | VB mean-field |

Thiết kế này cho phép cô lập từng đóng góp: B1 với B2 tách riêng tác dụng của
Student-t (RQ1); B3 với M1 tách riêng tác dụng của chuyển trạng thái (RQ2); M1
với M2/M3/M4 tách riêng tác dụng của phương pháp suy diễn (RQ3, RQ4, RQ5).

Mô hình B4 được cài đặt bằng chính khung MS-GARCH-X với $\nu$ **cố định** ở giá
trị rất lớn ($10^6$), khi đó Student-t chuẩn hóa hội tụ về phân phối chuẩn. Cách
này tránh phải viết một mô hình riêng, đồng thời bảo đảm B4 dùng đúng bộ lọc như
M1. Gradient theo $\nu$ được che lại để tham số này không bị cập nhật, và $\nu$
không tính vào số tham số khi tính AIC/BIC.

---

## 13. Chi phí tính toán

### 13.1 Số đo thực tế

Đo trên AMD Ryzen 7 5700X3D (8 nhân, 16 luồng), 32 GB RAM, một lần
forward + backward qua bộ lọc tiến:

| $T$ | MS-GARCH-X | GARCH một trạng thái |
|---|---|---|
| 500 | 394 ms | — |
| 1.735 (tập huấn luyện) | **1.400 ms** | **224 ms** |
| 2.000 | 1.629 ms | — |
| 5.000 | 4.147 ms | — |

Hai kết luận quan trọng:

- MS-GARCH-X đắt hơn GARCH một trạng thái khoảng **6,25 lần**.
- Số luồng **không ảnh hưởng**: 1, 4 hay 8 luồng đều cho khoảng 1.400 ms. Vòng
  lặp bị chặn bởi chi phí thông dịch Python và ghi sổ đạo hàm tự động trên các
  tensor chỉ 2 phần tử, không phải bởi phép nhân ma trận. Một tiến trình chạy
  thực tế chỉ dùng khoảng 0,92 nhân.

### 13.2 Vì sao MCMC đắt hơn MLE khoảng 100 lần

| | MLE (M1) | MCMC (M4) |
|---|---|---|
| Mục tiêu | Tìm **một điểm** cực đại | Mô tả **cả phân phối** 13 chiều |
| Số lần chạy bộ lọc | $6\times 500 = 3.000$ | $2.600\times 127 \approx 330.000$ |
| Thời gian ước tính | ~1,4 giờ | ~128 giờ |

Con số $2.600$ là $4$ chuỗi $\times\,(250$ khởi động $+\ 400$ mẫu$)$. Con số $127$
là số bước leapfrog tối đa ở độ sâu cây $D=7$, tức $2^7-1$. Nói cách khác, **một
mẫu duy nhất** tốn khoảng 3 phút.

Nguyên nhân sâu xa: hậu nghiệm có hình học "cứng" — NUTS thích nghi bước tích
phân xuống khoảng $10^{-4}$ do tương quan mạnh giữa $\omega_s,\alpha_s,\beta_s$ và
do $p_{ss}$ nằm sát biên 1. Bước nhỏ nên cần rất nhiều bước mới đi được một quãng
có ý nghĩa.

### 13.3 Đánh đổi khi giảm cấu hình

| Cách giảm | Ảnh hưởng | Có làm sai kết quả không |
|---|---|---|
| Giảm độ sâu cây $D$ | ESS giảm gần tỷ lệ thuận | **Không** — chỉ giảm hiệu quả |
| Giảm số mẫu | ESS giảm tỷ lệ thuận | **Không** |
| Thu hẹp phạm vi mô phỏng | Kết luận hẹp hơn | **Không**, nếu khai báo rõ |
| **Cắt giai đoạn khởi động** | Thích nghi chưa xong | **Có — mẫu bị lệch** |
| **Giảm số chuỗi xuống dưới 4** | $\widehat R$ mất ý nghĩa | **Có — mất chẩn đoán** |

Điểm hay bị hiểu nhầm: giảm độ sâu cây **không phải** là tăng tốc miễn phí. Trong
chế độ chạm trần, cả ESS lẫn chi phí mỗi vòng lặp đều tỷ lệ với độ dài quỹ đạo,
nên ESS trên mỗi giờ tính toán gần như không đổi. Cách tăng tốc thật sự không mất
gì là **chạy các chuỗi song song trên nhiều tiến trình**, vì các chuỗi độc lập
hoàn toàn về mặt thống kê.

---

## 14. Ánh xạ công thức sang mã nguồn

| Nội dung | Công thức | Tệp |
|---|---|---|
| Mật độ Student-t chuẩn hóa | Mục 4.1 | `msgarchx/distributions.py` |
| Biến đổi tham số | Mục 6.1 | `msgarchx/transforms.py` |
| Liên kết dương của phương sai | Mục 6.2 | `msgarchx/transforms.py` |
| GARCH, GARCH-X | Mục 4.2 | `msgarchx/models/garch.py` |
| Bộ lọc tiến, thu gọn Gray | Mục 5 | `msgarchx/models/ms_garch.py` |
| Làm trơn Kim | Mục 5.4 | `msgarchx/models/ms_garch.py` |
| Nhận dạng nhãn trạng thái | Mục 4.6 | `msgarchx/models/ms_garch.py` |
| Vector tham số, tiền nghiệm | Mục 6.1, 7.2 | `msgarchx/inference/param_spec.py` |
| MLE hai giai đoạn, AIC/BIC, sai số chuẩn | Mục 7.1 | `msgarchx/inference/mle.py` |
| Mô hình xác suất chung cho VB và MCMC | Mục 7.5 | `msgarchx/inference/bayes_model.py` |
| Variational Bayes | Mục 7.3 | `msgarchx/inference/variational.py` |
| MCMC bằng NUTS | Mục 7.4 | `msgarchx/inference/mcmc.py` |
| Dự báo | Mục 8 | `msgarchx/inference/forecast.py` |
| MAE, RMSE, QLIKE | Mục 9.1 | `msgarchx/evaluation/volatility_loss.py` |
| Điểm log, CRPS, PIT, phân vị hỗn hợp | Mục 9.2, 8.3 | `msgarchx/evaluation/density_scores.py` |
| Kupiec, Christoffersen | Mục 9.3 | `msgarchx/evaluation/var_backtests.py` |
| Diebold–Mariano | Mục 9.4 | `msgarchx/evaluation/dm_test.py` |
| Kiểm định ban đầu | Mục 10 | `msgarchx/stats_tests/preliminary_tests.py` |
| Cơ chế sinh dữ liệu | Mục 11.1 | `msgarchx/simulation/dgp.py` |
| Tám kịch bản | Mục 11.2 | `msgarchx/simulation/scenarios.py` |
| Bias, RMSE, Coverage, Brier | Mục 11.3 | `msgarchx/simulation/metrics.py` |
| Nạp dữ liệu, lợi suất, $RV_t$ | Mục 3.1, 3.2 | `msgarchx/data/loader.py` |
| Biến ngoại sinh, chuẩn hóa | Mục 3.3, 3.4 | `msgarchx/data/exogenous.py` |
| Điều phối toàn bộ quy trình | — | `scripts/run_full_study.py` |
| Báo cáo tổng hợp dạng HTML | — | `scripts/build_html_report.py` |

---

## 15. Quy trình chạy

Toàn bộ quy trình gồm tám bước, mỗi bước được lưu điểm kiểm tra riêng vào thư mục
`results/`:

| Thứ tự | Bước | Kết quả lưu |
|---|---|---|
| 0 | Nạp dữ liệu, xây biến ngoại sinh, chia mẫu | `00_data.pt` |
| 1 | Kiểm định thống kê ban đầu | `01_preliminary_tests.pt` |
| 2 | MLE cho B1, B2, B3, B4, M1 và sai số chuẩn | `02_mle.pt` |
| 3 | Variational Bayes cho M2, M3 | `03_vb.pt` |
| 4 | MCMC cho M4 | `04_mcmc.pt` |
| 5 | Mô hình mở rộng R1 và $\Pr(\nu_2<\nu_1\mid\mathcal{D}_T)$ | `05_r1.pt` |
| 6 | Đánh giá ngoài mẫu cho toàn bộ mô hình | `06_oos.pt` |
| 7 | Mô phỏng S1–S8 | `07_simulation_S1.pt` … `S8.pt` |

Chi tiết lệnh chạy, cài đặt thư viện và hai tệp `.bat` cho Windows nằm ở
[`code thuật toán/README.md`](code%20thuật%20toán/README.md).

**Lưu ý quan trọng về điểm kiểm tra**: kết quả chỉ được ghi ra đĩa **sau khi một
bước hoàn tất trọn vẹn**. Không có cơ chế lưu giữa chừng. Dừng chương trình khi
bước 2 đang chạy sẽ mất toàn bộ kết quả của B1 đến M1; dừng khi bước 4 đang chạy
sẽ mất cả bốn chuỗi MCMC. Thời điểm dừng an toàn là ngay sau khi nhật ký in dòng
báo hoàn tất một bước.

---

## 16. Những điểm chưa hoàn tất

Danh sách này ghi lại trung thực khoảng cách giữa cài đặt hiện tại và yêu cầu đầy
đủ của bản thảo.

**Dữ liệu**

- Ba biến ngoại sinh $BASIS_{t-1}$, $EXP_t$, $ROLL_t$ chưa dùng được do nguồn dữ
  liệu không có chỉ số VN30 cơ sở và không có lịch đáo hạn hợp đồng.

**Quy trình ước lượng**

- Điểm khởi tạo MLE hiện chỉ là nhiễu ngẫu nhiên quanh một tâm cố định. Bản thảo
  yêu cầu thêm **khởi tạo từ mô hình đơn giản hơn** (ví dụ lấy nghiệm của B3 làm
  điểm xuất phát cho M1).
- Chưa có bootstrap tham số hoặc bootstrap phần dư làm phương án thay thế khi ma
  trận Hessian không ổn định.
- Ngưỡng hội tụ hiện đặt ở $\lVert\nabla\ell_T\rVert<1$ trên thang không ràng
  buộc — khá lỏng so với độ lớn của hàm hợp lý.

**Chẩn đoán MCMC**

- Chỉ tính ESS phần trung tâm; bản thảo yêu cầu cả ESS phần đuôi.
- Chưa có biểu đồ chuỗi, thống kê độ sâu cây, chẩn đoán năng lượng, hàm tự tương
  quan và phân tích độ nhạy với số vòng khởi động.

**Đánh giá**

- CRPS đã cài đặt nhưng chưa được gọi trong quy trình chính.
- Biểu đồ PIT hiện chỉ vẽ cho M1; cần vẽ cho từng phương pháp suy diễn.
- Chưa tính độ rộng khoảng dự báo, chưa đo bộ nhớ cực đại, và tỷ lệ không hội tụ
  mới có cho MLE.

**Mô phỏng**

- Giá trị tham số của tám kịch bản hiện là giá trị ví dụ hợp lý về mặt định tính,
  cần chốt giá trị chính thức trước khi báo cáo.
- Brier score trong mô phỏng đang dùng xác suất **lọc**, trong khi công thức của
  bản thảo dùng xác suất **làm trơn**.

**Thiết kế ngoài mẫu**

- Chưa có tái ước lượng định kỳ trong tập kiểm định; tham số được ước lượng một
  lần duy nhất. Đây là giới hạn do chi phí tính toán và đã được nêu rõ trong bản
  thảo.
