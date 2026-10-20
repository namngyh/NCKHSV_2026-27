# Dự báo rủi ro và xây dựng cơ chế ký quỹ động cho hợp đồng tương lai VN30F1M bằng mô hình EGARCH-X

**Đề cương nghiên cứu chi tiết (phiên bản sửa đổi)**

Nguyễn Hoài Nam, Nguyễn Thuận Hiếu, Đặng Chí Hiếu — Khoa Toán – Thống kê, Trường Đại học Tôn Đức Thắng

Tháng 10 năm 2026

> Tệp này được sinh tự động từ mã nguồn LaTeX (`main.tex`). Số phương trình, số bảng, số mục và số tài liệu tham khảo trùng với bản PDF.

## Tóm tắt

Nghiên cứu xây dựng một khung định lượng liên kết dự báo rủi ro thị trường với mức ký quỹ cho hợp đồng tương lai chỉ số VN30 kỳ hạn gần nhất (VN30F1M). Dự án gồm hai giai đoạn. Giai đoạn 1 ước lượng lại mô hình EGARCH-X, tức mô hình GARCH dạng mũ có bổ sung biến giải thích ngoại sinh vào phương trình logarit phương sai có điều kiện, với ba phân phối sai số chuẩn hóa: phân phối chuẩn, Student-$t$ chuẩn hóa và Student-$t$ lệch. Dữ liệu là chuỗi ngày gồm giá mở cửa, cao nhất, thấp nhất, đóng cửa, khối lượng, khối lượng mua chủ động và khối lượng bán chủ động của VN30F1M. Do không có dữ liệu trong ngày, phương sai thực tế được đại diện bằng các ước lượng theo biên độ giá, có hiệu chỉnh thang đo. Hiệu ứng chuyển hợp đồng và chu kỳ đáo hạn được xử lý tường minh. Mô hình được ước lượng bằng hợp lý cực đại, sau đó suy diễn Bayes (biến phân mean-field, biến phân full-rank và Monte Carlo Hamilton với thuật toán NUTS) được áp dụng cho cấu hình được chọn. Giai đoạn 2 chuyển dự báo phân phối có điều kiện thành giá trị chịu rủi ro (VaR) và tổn thất kỳ vọng ở phần đuôi (ES) cho khoảng thanh lý $H$ phiên bằng mô phỏng, rồi xây dựng quy tắc ký quỹ động gồm mức đệm, mức sàn chống thuận chu kỳ, làm trơn bất đối xứng và giới hạn tốc độ điều chỉnh. Hiệu quả được đánh giá ngoài mẫu theo thiết kế ba khối thời gian, bằng các hàm mất mát nhất quán, các kiểm định độ phủ và đường biên đánh đổi giữa mức bảo vệ và lượng vốn ký quỹ.  

**Từ khóa:** EGARCH-X, biến động có điều kiện, giá trị chịu rủi ro, tổn thất kỳ vọng ở phần đuôi, ký quỹ động, ước lượng theo biên độ giá, suy diễn Bayes, VN30F1M.

## Mục lục

- [Bảng ký hiệu](#bảng-ký-hiệu)
- [1. Đặt vấn đề và mục tiêu nghiên cứu](#1-đặt-vấn-đề-và-mục-tiêu-nghiên-cứu)
  - [1.1. Bối cảnh và động cơ](#11-bối-cảnh-và-động-cơ)
  - [1.2. Đối tượng và phạm vi](#12-đối-tượng-và-phạm-vi)
  - [1.3. Câu hỏi nghiên cứu và giả thuyết](#13-câu-hỏi-nghiên-cứu-và-giả-thuyết)
  - [1.4. Đóng góp dự kiến](#14-đóng-góp-dự-kiến)
  - [1.5. Những thay đổi so với đề cương đã đăng ký](#15-những-thay-đổi-so-với-đề-cương-đã-đăng-ký)
  - [1.6. Tổng quan tài liệu liên quan](#16-tổng-quan-tài-liệu-liên-quan)
- [2. Dữ liệu và xây dựng biến](#2-dữ-liệu-và-xây-dựng-biến)
  - [2.1. Nguồn và cấu trúc dữ liệu](#21-nguồn-và-cấu-trúc-dữ-liệu)
  - [2.2. Thông số thể chế cần xác minh](#22-thông-số-thể-chế-cần-xác-minh)
  - [2.3. Lợi suất và xử lý ngày chuyển hợp đồng](#23-lợi-suất-và-xử-lý-ngày-chuyển-hợp-đồng)
  - [2.4. Đại lượng đại diện cho phương sai](#24-đại-lượng-đại-diện-cho-phương-sai)
  - [2.5. Biến lịch](#25-biến-lịch)
  - [2.6. Biến ngoại sinh ứng viên](#26-biến-ngoại-sinh-ứng-viên)
  - [2.7. Chuẩn hóa không rò rỉ thông tin](#27-chuẩn-hóa-không-rò-rỉ-thông-tin)
  - [2.8. Kiểm tra chất lượng dữ liệu](#28-kiểm-tra-chất-lượng-dữ-liệu)
  - [2.9. Giới hạn của dữ liệu](#29-giới-hạn-của-dữ-liệu)
- [3. Đặc tả mô hình EGARCH-X](#3-đặc-tả-mô-hình-egarch-x)
  - [3.1. Phương trình trung bình và phương sai](#31-phương-trình-trung-bình-và-phương-sai)
  - [3.2. Phân phối sai số chuẩn hóa](#32-phân-phối-sai-số-chuẩn-hóa)
  - [3.3. Điều kiện chính quy và các lựa chọn cài đặt](#33-điều-kiện-chính-quy-và-các-lựa-chọn-cài-đặt)
  - [3.4. Quan sát khuyết tại phiên chuyển hợp đồng](#34-quan-sát-khuyết-tại-phiên-chuyển-hợp-đồng)
  - [3.5. Hàm hợp lý](#35-hàm-hợp-lý)
  - [3.6. Mô hình lồng nhau và kiểm định CH1](#36-mô-hình-lồng-nhau-và-kiểm-định-ch1)
- [4. Ước lượng và suy diễn tham số](#4-ước-lượng-và-suy-diễn-tham-số)
  - [4.1. Tham số hóa không ràng buộc](#41-tham-số-hóa-không-ràng-buộc)
  - [4.2. Ước lượng hợp lý cực đại](#42-ước-lượng-hợp-lý-cực-đại)
  - [4.3. Suy diễn Bayes](#43-suy-diễn-bayes)
  - [4.4. Nghiên cứu mô phỏng khôi phục tham số](#44-nghiên-cứu-mô-phỏng-khôi-phục-tham-số)
- [5. Thiết kế đánh giá ngoài mẫu](#5-thiết-kế-đánh-giá-ngoài-mẫu)
  - [5.1. Chia mẫu theo thời gian](#51-chia-mẫu-theo-thời-gian)
  - [5.2. Dự báo phương sai](#52-dự-báo-phương-sai)
  - [5.3. Dự báo mật độ](#53-dự-báo-mật-độ)
  - [5.4. Dự báo VaR và ES một phiên](#54-dự-báo-var-và-es-một-phiên)
  - [5.5. So sánh dự báo có kiểm định](#55-so-sánh-dự-báo-có-kiểm-định)
  - [5.6. Phân tích theo nhóm phiên](#56-phân-tích-theo-nhóm-phiên)
- [6. Quy trình chọn biến ngoại sinh](#6-quy-trình-chọn-biến-ngoại-sinh)
- [7. Giai đoạn 2: Thiết kế cơ chế ký quỹ động](#7-giai-đoạn-2-thiết-kế-cơ-chế-ký-quỹ-động)
  - [7.1. Tổn thất, phân vị và mục tiêu ký quỹ](#71-tổn-thất-phân-vị-và-mục-tiêu-ký-quỹ)
  - [7.2. Dự báo rủi ro $H$ phiên bằng mô phỏng](#72-dự-báo-rủi-ro-h-phiên-bằng-mô-phỏng)
  - [7.3. Quy tắc ký quỹ năm bước](#73-quy-tắc-ký-quỹ-năm-bước)
  - [7.4. Hiệu chỉnh siêu tham số](#74-hiệu-chỉnh-siêu-tham-số)
  - [7.5. Quy tắc đối chứng](#75-quy-tắc-đối-chứng)
  - [7.6. Đánh giá ký quỹ](#76-đánh-giá-ký-quỹ)
- [8. Kế hoạch thực hiện](#8-kế-hoạch-thực-hiện)
  - [8.1. Quy trình tổng thể](#81-quy-trình-tổng-thể)
  - [8.2. Tiến độ](#82-tiến-độ)
  - [8.3. Sản phẩm](#83-sản-phẩm)
  - [8.4. Rủi ro và phương án dự phòng](#84-rủi-ro-và-phương-án-dự-phòng)
  - [8.5. Tái lập](#85-tái-lập)
  - [8.6. Dự trù kinh phí](#86-dự-trù-kinh-phí)
  - [8.7. Giới hạn của nghiên cứu](#87-giới-hạn-của-nghiên-cứu)
- [Phụ lục A. Công thức VaR và ES một phiên theo phân phối](#phụ-lục-a-công-thức-var-và-es-một-phiên-theo-phân-phối)
- [Phụ lục B. Phân phối sai số tổng quát (phương án thay thế)](#phụ-lục-b-phân-phối-sai-số-tổng-quát-phương-án-thay-thế)
- [Phụ lục C. Danh mục kiểm tra trước khi chạy ước lượng chính thức](#phụ-lục-c-danh-mục-kiểm-tra-trước-khi-chạy-ước-lượng-chính-thức)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

# Bảng ký hiệu

Toàn bộ tài liệu dùng thống nhất các ký hiệu dưới đây. Lợi suất và phương sai được tính theo đơn vị phần trăm (lợi suất) và phần trăm bình phương (phương sai), trừ khi nói khác. Tỷ lệ ký quỹ và tổn thất trong Mục 7 tính theo tỷ lệ của giá trị hợp đồng.

| **Ký hiệu** | **Ý nghĩa** |
|---|---|
| *Thời gian, thông tin và dữ liệu* |  |
| $t$ | Chỉ số phiên giao dịch. |
| $\mathcal{F}_t$ | Tập thông tin có được đến cuối phiên $t$. |
| $P^{\mathrm o}_t, P^{\mathrm h}_t, P^{\mathrm l}_t, P^{\mathrm c}_t$ | Giá mở cửa, cao nhất, thấp nhất, đóng cửa của VN30F1M tại phiên $t$; viết gọn $P_t\equiv P^{\mathrm c}_t$. |
| $V_t,\ V^{+}_t,\ V^{-}_t$ | Khối lượng giao dịch, khối lượng mua chủ động, khối lượng bán chủ động tại phiên $t$. |
| $J_t$ | Chỉ thị ngày chuyển hợp đồng: $J_t=1$ nếu phiên $t$ là phiên đầu tiên chuỗi F1M dùng hợp đồng mới. |
| $\tau_t$ | Số phiên còn lại đến ngày đáo hạn của hợp đồng đang là F1M tại phiên $t$. |
| $\mathbf{D}_t$ | Vectơ biến giả lịch: $D^{\text{đh}}_t$ (phiên đáo hạn), $D^{\text{sau}}_t$ (phiên sau chuyển hợp đồng), $D^{\text{nghỉ}}_t$ (phiên đầu sau kỳ nghỉ dài). |
| *Lợi suất và đại lượng đại diện cho phương sai* |  |
| $r_t$ | Lợi suất logarit đóng cửa – đóng cửa (%), $r_t=100\ln(P_t/P_{t-1})$, chỉ xác định khi $J_t=0$. |
| $r^{\text{đêm}}_t$ | Lợi suất qua đêm (%), $r^{\text{đêm}}_t=100\ln(P^{\mathrm o}_t/P^{\mathrm c}_{t-1})$, chỉ xác định khi $J_t=0$. |
| $\rho^{\mathrm P}_t,\rho^{\mathrm{GK}}_t,\rho^{\mathrm{RS}}_t$ | Ước lượng phương sai trong phiên theo biên độ giá: Parkinson, Garman–Klass, Rogers–Satchell. |
| $\rho_t,\ \rho^{\star}_t$ | Đại lượng đại diện cho phương sai ngày; $\rho^{\star}_t=\hat c_\rho\,\rho_t$ là bản đã hiệu chỉnh thang đo. |
| *Mô hình EGARCH-X* |  |
| $\mu$ | Kỳ vọng có điều kiện của lợi suất (hằng số). |
| $\varepsilon_t,\ z_t,\ h_t$ | Phần dư, sai số chuẩn hóa ($\mathbb{E}z_t=0$, $\operatorname{Var} z_t=1$), phương sai có điều kiện: $\varepsilon_t=\sqrt{h_t}\,z_t$. |
| $\mathcal{G}$ | Phân phối của $z_t$: $\mathcal{G}\in\{\mathcal N,\ t^{*}_{\nu},\ \mathrm{SkT}_{\nu,\lambda}\}$. |
| $f_{\mathcal{G}},\ F_{\mathcal{G}},\ Q_{\mathcal{G}}$ | Hàm mật độ, hàm phân phối, hàm phân vị của $\mathcal{G}$. |
| $\nu,\ \lambda$ | Bậc tự do ($\nu>2$) và tham số lệch ($-1<\lambda<1$). |
| $\mu_{\|z\|}$ | $\mathbb{E}\|z_t\|$ dưới phân phối $\mathcal{G}$. |
| $\omega,\kappa,\theta,\beta$ | Hệ số chặn, hiệu ứng độ lớn, hiệu ứng dấu (đòn bẩy), độ dai dẳng của $\ln h_t$. |
| $\mathbf{X}_{t-1}\in\mathbb R^{d_X}$ | Vectơ biến ngoại sinh quan sát được đến cuối phiên $t-1$; $\boldsymbol{\gamma}$ là vectơ hệ số. |
| $\boldsymbol{\delta}$ | Vectơ hệ số của biến giả lịch $\mathbf{D}_t$. |
| $\boldsymbol{\psi}\in\Psi\subset\mathbb R^{d}$ | Vectơ toàn bộ tham số của mô hình; $d=\dim\boldsymbol{\psi}$. |
| $\boldsymbol{\xi}=g^{-1}(\boldsymbol{\psi})$ | Ảnh của $\boldsymbol{\psi}$ trên không gian không ràng buộc $\mathbb R^{d}$ qua song ánh trơn $g$. |
| $\mathcal L,\ \ell_T$ | Hàm hợp lý và logarit hàm hợp lý trên mẫu ước lượng. |
| $\pi(\boldsymbol{\psi}),\ \pi(\boldsymbol{\psi}\mid\mathcal D)$ | Mật độ tiên nghiệm và hậu nghiệm; $\mathcal D$ là dữ liệu dùng để suy diễn. |
| $q(\boldsymbol{\xi})$ | Phân phối xấp xỉ hậu nghiệm trong suy diễn biến phân. |
| *Rủi ro và ký quỹ* |  |
| $\alpha$ | Mức đuôi (xác suất tổn thất vượt ngưỡng), ví dụ $\alpha=1\%$. |
| $p$ | Mức của phân vị; đối với tổn thất, $p=1-\alpha$. |
| $Q_Y(p)$ | Hàm phân vị của biến ngẫu nhiên $Y$: $Q_Y(p)=\inf\{x: F_Y(x)\ge p\}$. |
| $H$ | Khoảng thanh lý (số phiên). |
| $R_{t,H}$ | Lợi suất đơn $H$ phiên: $R_{t,H}=P_{t+H}/P_t-1$. |
| $L^{\mathrm M}_{t,H},\ L^{\mathrm B}_{t,H}$ | Tổn thất theo tỷ lệ của vị thế mua và vị thế bán: $L^{\mathrm M}_{t,H}=-R_{t,H}$, $L^{\mathrm B}_{t,H}=R_{t,H}$. |
| $\mathrm{VaR}_{t,H,\alpha},\ \mathrm{ES}_{t,H,\alpha}$ | Giá trị chịu rủi ro và tổn thất kỳ vọng ở phần đuôi, có điều kiện theo $\mathcal{F}_t$. |
| $m_t,\ \mathcal M_t$ | Tỷ lệ ký quỹ và ký quỹ bằng tiền trên một hợp đồng: $\mathcal M_t=m_t K P_t$. |
| $K$ | Hệ số nhân của hợp đồng (đồng trên một điểm chỉ số). |
| $\boldsymbol{\Lambda}$ | Vectơ siêu tham số của quy tắc ký quỹ: $(H,\alpha,w,b,\varphi,\eta,\Delta^{\downarrow},g)$. |
| $c_{\mathrm v},\ c_{\mathrm{th}}$ | Chi phí vốn trên một đơn vị ký quỹ mỗi phiên; chi phí trên một đơn vị tổn thất vượt ký quỹ. |
| *Đánh giá* |  |
| $\mathcal T_{\mathrm{UL}},\mathcal T_{\mathrm{XT}},\mathcal T_{\mathrm{KT}}$ | Khối ước lượng ban đầu, khối xác thực, khối kiểm tra. |
| $N$ | Số phiên dự báo trong một khối đánh giá. |
| $I_t$ | Chỉ thị vi phạm: $I_t=\mathbf{1}\{L_t>\widehat{\mathrm{VaR}}_{t}\}$. |
| $S$ | Số đường mô phỏng Monte Carlo. |

# 1. Đặt vấn đề và mục tiêu nghiên cứu

## 1.1. Bối cảnh và động cơ

Ký quỹ là tuyến phòng vệ đầu tiên của hệ thống thanh toán bù trừ phái sinh. Mức ký quỹ quá thấp làm tổn thất có thể vượt tài sản bảo đảm khi giá biến động mạnh; mức ký quỹ quá cao làm tăng chi phí vốn của nhà đầu tư và giảm thanh khoản thị trường. Các nguyên tắc quốc tế về hạ tầng thị trường tài chính yêu cầu ký quỹ được xác định dựa trên rủi ro, đồng thời hạn chế tính thuận chu kỳ, tức xu hướng ký quỹ tăng đột ngột đúng vào lúc thị trường căng thẳng \[11\].

Biến động của lợi suất tài sản tài chính có tính tụ cụm: các giai đoạn biến động mạnh và yếu nối tiếp nhau. Họ mô hình phương sai có điều kiện thay đổi tự hồi quy (ARCH) và mở rộng tổng quát (GARCH) mô tả hiện tượng này \[15, 6\]. Mô hình GARCH dạng mũ (EGARCH) mô hình hóa logarit của phương sai, nhờ đó vừa bảo đảm tính dương của phương sai mà không cần ràng buộc dấu, vừa cho phép phản ứng bất đối xứng trước cú sốc dương và âm \[38\]. Khi bổ sung biến giải thích ngoại sinh vào phương trình phương sai, ta được lớp mô hình GARCH-X, mà các mô hình như Realized GARCH và HEAVY là những trường hợp phát triển dùng thước đo biến động thực hiện \[25, 45\].

Với VN30F1M, một mức ký quỹ cố định bỏ qua tính tụ cụm của biến động: thừa vốn trong giai đoạn yên tĩnh và thiếu bảo vệ trong giai đoạn căng thẳng. Nghiên cứu này đề xuất một quy tắc ký quỹ thay đổi theo dự báo rủi ro có điều kiện từ mô hình EGARCH-X, và đánh giá quy tắc đó trên dữ liệu lịch sử.

## 1.2. Đối tượng và phạm vi

Đối tượng nghiên cứu gồm: (i) biến động có điều kiện và rủi ro đuôi của chuỗi lợi suất ngày VN30F1M; (ii) đóng góp của biến ngoại sinh vào dự báo; (iii) ảnh hưởng của phân phối sai số và phương pháp suy diễn đến dự báo phân phối; (iv) quy tắc chuyển dự báo VaR/ES thành tỷ lệ ký quỹ.

Phạm vi: dữ liệu ngày của VN30F1M từ 06/11/2017 đến 17/07/2026 (khoảng $2.170$ phiên). Ký quỹ được mô phỏng theo tần suất ngày. Nghiên cứu không mô phỏng toàn bộ quy trình thanh toán bù trừ, quản trị tài sản bảo đảm hay các yêu cầu pháp lý của tổ chức bù trừ; kết quả mang tính thực nghiệm và không nhằm thay thế cơ chế hiện hành.

## 1.3. Câu hỏi nghiên cứu và giả thuyết

- **CH1.** Việc bổ sung biến ngoại sinh vào phương trình logarit phương sai của EGARCH có cải thiện dự báo phân phối ngoài mẫu của lợi suất VN30F1M hay không?

- **CH2.** Phân phối sai số chuẩn hóa (chuẩn, Student-$t$, Student-$t$ lệch) và phương pháp suy diễn (hợp lý cực đại, biến phân mean-field, biến phân full-rank, NUTS) ảnh hưởng thế nào đến dự báo rủi ro đuôi và mức bất định của tham số?

- **CH3.** Quy tắc ký quỹ động dựa trên VaR/ES có cải thiện sự đánh đổi giữa mức bảo vệ trước tổn thất và lượng vốn ký quỹ so với các quy tắc đối chứng hay không?

Các câu hỏi được chuyển thành giả thuyết kiểm định được:

**Giả thuyết 1** (CH1). $H_0:\boldsymbol{\gamma}=\mathbf 0$ trong mô hình EGARCH-X (kiểm định trong mẫu bằng tỷ số hợp lý); và $H_0:\mathbb{E}[\Delta\ell_t]=0$, với $\Delta\ell_t$ là chênh lệch điểm mất mát ngoài mẫu giữa EGARCH-X và EGARCH ($\boldsymbol{\gamma}=\mathbf 0$), kiểm định bằng thống kê Diebold–Mariano.

**Giả thuyết 2** (CH2). Ba phân phối sai số cho cùng điểm logarit mật độ dự báo và cùng điểm VaR–ES ngoài mẫu; phân phối hậu nghiệm xấp xỉ bằng biến phân trùng với phân phối hậu nghiệm tham chiếu từ NUTS về độ lệch chuẩn biên và cấu trúc tương quan.

**Giả thuyết 3** (CH3). Tại cùng mức ký quỹ trung bình, quy tắc ký quỹ động không làm giảm thiếu hụt bình quân so với quy tắc đối chứng tốt nhất.

## 1.4. Đóng góp dự kiến

1.  Một quy trình dữ liệu tái lập cho chuỗi F1M ngày, xử lý tường minh ngày chuyển hợp đồng, chu kỳ đáo hạn và ngày chạm biên độ giá, với đại lượng đại diện cho phương sai theo biên độ giá có hiệu chỉnh thang đo.

2.  Đánh giá thực nghiệm đóng góp của biến ngoại sinh (khối lượng bất thường, biên độ giá, mất cân bằng lệnh) trong EGARCH-X cho VN30F1M, với quy trình chọn biến có kiểm soát đa kiểm định.

3.  So sánh có kiểm soát giữa ba phân phối sai số và bốn phương pháp suy diễn trên cùng hàm hợp lý.

4.  Một quy tắc ký quỹ động có các thành phần chống thuận chu kỳ, cùng khung hiệu chỉnh siêu tham số dựa trên chi phí kỳ vọng và đường biên đánh đổi.

## 1.5. Những thay đổi so với đề cương đã đăng ký

Bảng 2 liệt kê các điều chỉnh và lý do. Các điều chỉnh xuất phát từ việc rà soát kết quả đợt ước lượng trước và từ giới hạn của dữ liệu hiện có.

**Bảng 2.** Các điều chỉnh so với đề cương đã đăng ký

| **Nội dung** | **Phiên bản cũ** | **Phiên bản sửa đổi và lý do** |
|---|---|---|
| Danh mục mô hình | GARCH, EGARCH, EGARCH-X | Chỉ EGARCH-X; EGARCH được giữ dưới dạng ràng buộc lồng $\boldsymbol{\gamma}=\mathbf 0$ để trả lời CH1 trên dữ liệu đã làm sạch. |
| Phân phối sai số | Chuẩn, Student-$t$, GED, Student-$t$ lệch | Ba phân phối: chuẩn, Student-$t$, Student-$t$ lệch. GED được trình bày ở Phụ lục B làm phương án thay thế. |
| Chia mẫu | $80\%/20\%$ | Ba khối: ước lượng ban đầu, xác thực, kiểm tra; cửa sổ mở rộng. Lý do: cần khối riêng để chọn mô hình và siêu tham số ký quỹ, và khối kiểm tra đủ lớn cho kiểm định độ phủ (Mục 5.1). |
| Đại lượng đại diện cho phương sai | Bình phương phần dư | Ước lượng theo biên độ giá cộng phần qua đêm, có hiệu chỉnh thang đo (Mục 2.4). |
| Ngày chuyển hợp đồng | Chưa xử lý | Loại lợi suất ngày chuyển khỏi hàm hợp lý; điều chỉnh đệ quy phương sai (Mục 2.3). |
| Biến khối lượng | $\ln(1+V_t)$ chuẩn hóa | Khối lượng bất thường đã khử chu kỳ đáo hạn, do $\ln(1+V_t)$ không dừng (Mục 2.6). |
| Chu kỳ đáo hạn | Chưa xử lý | Biến giả lịch trong phương trình phương sai. |
| Kiểm định so sánh | AIC/BIC, QLIKE | Bổ sung Diebold–Mariano, tập mô hình tin cậy, kiểm định SPA, điểm nhất quán VaR–ES. |
| Ký quỹ | VaR/ES một phiên | Khoảng thanh lý $H$ phiên, mô phỏng đường đi, quy tắc năm bước, hiệu chỉnh theo chi phí kỳ vọng. |
| Quy ước VaR | Phân vị của lợi suất | Phân vị của tổn thất (VaR dương), thống nhất trong toàn tài liệu. |

## 1.6. Tổng quan tài liệu liên quan

#### Mô hình biến động.

GARCH \[6\] và phân phối Student-$t$ cho sai số \[7\] là chuẩn mực. EGARCH \[38\] và GJR-GARCH \[21\] mô tả hiệu ứng đòn bẩy. Lý thuyết ước lượng hợp lý cực đại cho EGARCH phức tạp hơn GARCH do điều kiện khả nghịch của đệ quy \[46\]. Sai lệch cấu trúc trong mức phương sai làm phóng đại độ dai dẳng ước lượng \[36\]; các mô hình có thành phần chậm như Spline-GARCH \[17\] là một cách xử lý. Phân phối Student-$t$ lệch của Hansen \[23\] cho phép đuôi bất đối xứng.

#### Biến ngoại sinh và thước đo biến động.

Thước đo biến động thực hiện từ dữ liệu trong ngày cải thiện đáng kể dự báo \[25, 45, 10\]. Khi chỉ có dữ liệu ngày, các ước lượng theo biên độ giá của Parkinson \[40\], Garman–Klass \[19\] và Rogers–Satchell \[44\] hiệu quả hơn nhiều so với bình phương lợi suất. Phần qua đêm cần được xử lý riêng khi xây dựng thước đo phương sai cho cả ngày \[26\].

#### Đánh giá dự báo biến động và rủi ro.

Hàm mất mát QLIKE bền vững trước nhiễu của đại lượng đại diện \[41, 27\]. Phân vị là phiếm hàm có thể đánh giá được bằng hàm mất mát phân vị \[33, 22\]; ES không có tính chất này một mình nhưng cặp (VaR, ES) thì có \[18, 42\]. ES là thước đo rủi ro nhất quán \[2, 1\]. Kiểm định VaR gồm Kupiec \[35\], Christoffersen \[9\] và kiểm định phân vị động \[16\]. So sánh dự báo dùng Diebold–Mariano \[14\], Giacomini–White \[20\], tập mô hình tin cậy \[28\] và kiểm định SPA \[24, 50\].

#### Suy diễn Bayes.

Suy diễn biến phân \[5\] với biến phân vi phân tự động \[34\] và thủ thuật tái tham số hóa \[32\] cho xấp xỉ nhanh; NUTS \[30\] cho mẫu hậu nghiệm tiệm cận chính xác với các chẩn đoán hội tụ hiện đại \[48\]. Chất lượng của xấp xỉ biến phân được chẩn đoán bằng lấy mẫu quan trọng làm trơn Pareto \[51\] và hiệu chuẩn dựa trên mô phỏng \[47\].

#### Ký quỹ và quản trị rủi ro.

Khung vốn tối thiểu cho rủi ro thị trường đã chuyển từ VaR sang ES ở mức $97{,}5\%$ \[3\]. Mô phỏng lịch sử có lọc theo mô hình GARCH là phương pháp chuẩn để ước lượng rủi ro đuôi có điều kiện \[37\].

# 2. Dữ liệu và xây dựng biến

## 2.1. Nguồn và cấu trúc dữ liệu

Dữ liệu lịch sử VN30F1M được thu thập từ nền tảng DataPro \[12\], tần suất ngày, giai đoạn 06/11/2017 – 17/07/2026. Mỗi phiên gồm bảy trường: $P^{\mathrm o}_t, P^{\mathrm h}_t, P^{\mathrm l}_t, P^{\mathrm c}_t, V_t, V^{+}_t, V^{-}_t$. Chuỗi F1M là chuỗi nối của các hợp đồng kỳ hạn gần nhất; dữ liệu *không* gồm giá của hợp đồng kỳ hạn thứ hai, giá thanh toán hằng ngày, dữ liệu trong ngày hay chuỗi tỷ lệ ký quỹ lịch sử. Các giới hạn này quyết định cách xây dựng biến ở các mục dưới và được tổng hợp ở Mục 2.9.

## 2.2. Thông số thể chế cần xác minh

Bảng 3 liệt kê các thông số thể chế được dùng trong tính toán. Các giá trị cần được đối chiếu với quy chế hiện hành và quy chế áp dụng trong từng thời kỳ của mẫu, từ nguồn chính thức của Sở Giao dịch Chứng khoán Hà Nội \[29\] và Tổng công ty Lưu ký và Bù trừ Chứng khoán Việt Nam \[49\], trước khi chạy ước lượng chính thức.

**Bảng 3.** Thông số thể chế dùng trong nghiên cứu (cần xác minh)

| **Thông số** | **Giá trị làm việc** | **Vai trò** |
|---|---|---|
| Ngày giao dịch cuối cùng | Thứ Năm thứ ba của tháng đáo hạn | Xác định $J_t$, $\tau_t$, $D^{\text{đh}}_t$ |
| Biên độ dao động giá ngày | $\pm 7\%$ so với giá tham chiếu | Xác định ngày chạm biên |
| Hệ số nhân $K$ | $100.000$ đồng/điểm | Quy đổi ký quỹ ra tiền |
| Bước điều chỉnh tỷ lệ ký quỹ $g$ | Theo thông lệ của tổ chức bù trừ | Làm tròn ở Mục 7.3 |

## 2.3. Lợi suất và xử lý ngày chuyển hợp đồng

Gọi $\mathcal R$ là tập các phiên chuyển hợp đồng, xác định theo lịch đáo hạn ở Bảng 3 chứ không dò bằng khoảng nhảy giá; $J_t=\mathbf{1}\{t\in\mathcal R\}$. Tại phiên chuyển, $P_{t-1}$ là giá của hợp đồng cũ còn $P_t$ là giá của hợp đồng mới, nên $\ln(P_t/P_{t-1})$ chứa chênh lệch cơ sở giữa hai kỳ hạn và không phải lợi suất của một vị thế thực. Vì dữ liệu không có giá của hợp đồng mới tại phiên $t-1$, lợi suất cùng hợp đồng tại phiên chuyển không tính được. Do đó: $$\tag{1}
r_t=100\ln\frac{P_t}{P_{t-1}},\qquad r^{\text{đêm}}_t=100\ln\frac{P^{\mathrm o}_t}{P^{\mathrm c}_{t-1}},\qquad\text{chỉ xác định khi } J_t=0.$$ Các phiên $J_t=1$ được coi là quan sát khuyết của $r_t$. Cách đưa chúng vào đệ quy phương sai được trình bày ở Mục 3.4. Với khoảng $105$ lần chuyển trong mẫu, tỷ lệ quan sát khuyết xấp xỉ $5\%$.

## 2.4. Đại lượng đại diện cho phương sai

Phương sai có điều kiện $h_t$ không quan sát được, nên việc đánh giá dự báo cần một đại lượng đại diện không chệch (hoặc chệch đã biết) và ít nhiễu. Bình phương phần dư $\hat\varepsilon_t^2$ không chệch nhưng rất nhiễu, làm giảm công suất của các kiểm định so sánh \[41, 27\]. Khi không có dữ liệu trong ngày, ta dùng ước lượng theo biên độ giá. Gọi $$u_t=\ln\frac{P^{\mathrm h}_t}{P^{\mathrm o}_t},\qquad d_t=\ln\frac{P^{\mathrm l}_t}{P^{\mathrm o}_t},\qquad c_t=\ln\frac{P^{\mathrm c}_t}{P^{\mathrm o}_t}$$ là biên độ lên, biên độ xuống và lợi suất trong phiên (theo logarit, chưa nhân $100$). Ba ước lượng phương sai trong phiên, nhân $100^2$ để về thang phần trăm bình phương, là $$\begin{aligned}
\rho^{\mathrm P}_t&=100^2\,\frac{(u_t-d_t)^2}{4\ln 2}, \qquad(3)\\
\rho^{\mathrm{GK}}_t&=100^2\Big[\tfrac12(u_t-d_t)^2-(2\ln2-1)\,c_t^2\Big], \qquad(4)\\
\rho^{\mathrm{RS}}_t&=100^2\big[u_t(u_t-c_t)+d_t(d_t-c_t)\big] \qquad(5)
\end{aligned}$$ \[40, 19, 44\]. Khi dữ liệu thỏa $P^{\mathrm h}_t\ge\max(P^{\mathrm o}_t,P^{\mathrm c}_t)$ và $P^{\mathrm l}_t\le\min(P^{\mathrm o}_t,P^{\mathrm c}_t)$, ta có $|c_t|\le u_t-d_t$, nên $\rho^{\mathrm{GK}}_t\ge\big(\tfrac12-(2\ln2-1)\big)100^2(u_t-d_t)^2\ge 0$; tương tự $\rho^{\mathrm{RS}}_t\ge0$ vì $u_t\ge\max(0,c_t)$ và $d_t\le\min(0,c_t)$. Ước lượng Rogers–Satchell không chệch khi giá có thành phần xu hướng.

Các ước lượng (3)–(5) chỉ phản ánh phương sai trong phiên. Đại lượng đại diện cho phương sai đóng cửa – đóng cửa được xây dựng bằng cách cộng phần qua đêm và hiệu chỉnh thang đo theo tinh thần của \[26\]: $$\tag{6}
\rho_t=\rho^{\mathrm{GK}}_t+\big(r^{\text{đêm}}_t\big)^2,\qquad
\hat c_\rho=\frac{\sum_{t\in\mathcal T_{\mathrm{UL}},\,J_t=0}r_t^2}{\sum_{t\in\mathcal T_{\mathrm{UL}},\,J_t=0}\rho_t},\qquad
\rho^{\star}_t=\hat c_\rho\,\rho_t .$$ Hệ số $\hat c_\rho$ chỉ được tính trên khối ước lượng ban đầu (và cập nhật theo cửa sổ mở rộng khi ước lượng lại), để không dùng thông tin tương lai. Hệ số này bù cho hai nguồn chệch xuống: biên độ quan sát rời rạc nhỏ hơn biên độ liên tục, và các khoảng không giao dịch trong phiên. $\rho^{\star}_t$ không xác định khi $J_t=1$.

*Nhận xét 1* (Ngày chạm biên). Khi giá khóa ở biên độ trần hoặc sàn trong cả phiên, $P^{\mathrm o}_t=P^{\mathrm h}_t=P^{\mathrm l}_t=P^{\mathrm c}_t$ và mọi ước lượng theo biên độ bằng không, trong khi biến động thực tế rất lớn. Các phiên này được đánh dấu; $\rho^{\star}_t$ của chúng bị loại khỏi đánh giá QLIKE, và trong biến ngoại sinh được thay bằng $r_t^2$.

## 2.5. Biến lịch

Ba biến giả, đều biết trước tại thời điểm dự báo: $$\begin{gathered}
D^{\text{đh}}_t=\mathbf{1}\{\tau_t=0\},\qquad D^{\text{sau}}_t=J_t,\\
D^{\text{nghỉ}}_t=\mathbf{1}\{\text{phiên } t \text{ là phiên đầu sau kỳ nghỉ liên tiếp từ } 3 \text{ ngày làm việc}\}.
\end{gathered}$$ Biến $D^{\text{đh}}_t$ và $D^{\text{sau}}_t$ mô tả khác biệt biến động trước và sau đáo hạn; $D^{\text{nghỉ}}_t$ mô tả lượng thông tin tích lũy trong kỳ nghỉ dài (ví dụ Tết Nguyên đán).

## 2.6. Biến ngoại sinh ứng viên

Mọi biến ngoại sinh vào phương trình phương sai của phiên $t$ được đo đến cuối phiên $t-1$. Chuỗi khối lượng F1M có dạng răng cưa theo chu kỳ đáo hạn (thanh khoản chuyển dần sang kỳ hạn sau) và có xu hướng tăng dài hạn, nên $\ln(1+V_t)$ không dừng. Gọi $n_V=60$, $\overline V_{t-1}=\frac1{n_V}\sum_{j=1}^{n_V}V_{t-j}$ và hệ số mùa theo kỳ hạn $$\hat\varsigma(k)=\frac{1}{|\{t\in\mathcal T_{\mathrm{UL}}:\tau_t=k\}|}\sum_{t\in\mathcal T_{\mathrm{UL}}:\ \tau_t=k}\frac{V_t}{\overline V_{t-1}} .$$ Bảng 4 định nghĩa các ứng viên. Hằng số $\epsilon>0$ nhỏ tránh logarit của không.

**Bảng 4.** Biến ngoại sinh ứng viên

| **Ký hiệu** | **Định nghĩa** | **Diễn giải** |
|---|---|---|
| $A_t$ | $\ln V_t-\ln\!\big(\overline V_{t-1}\,\hat\varsigma(\tau_t)\big)$ | Khối lượng bất thường đã khử xu hướng và chu kỳ đáo hạn |
| $G_t$ | $\ln\big(\rho^{\mathrm{GK}}_t+\epsilon\big)$ | Biến động trong phiên theo biên độ giá; xác định cả ở phiên chuyển |
| $F_t$ | $\ln\!\big(\frac{1}{60}\sum_{j=0}^{59}\rho^{\mathrm{GK}}_{t-j}\big)$ | Thành phần chậm của biến động |
| $B_t$ | $\big\|\mathrm{OFI}_t\big\|$,  $\mathrm{OFI}_t=\dfrac{V^{+}_t-V^{-}_t}{V^{+}_t+V^{-}_t}$ | Độ lớn mất cân bằng lệnh |
| $O_t$ | $\ln\big((r^{\text{đêm}}_t)^2+\epsilon\big)$ | Độ lớn khoảng nhảy qua đêm; khuyết khi $J_t=1$ |

Cấu hình chính có $d_X=2$ biến, $\mathbf{X}_{t-1}=(A_{t-1},G_{t-1})^\top$, tương ứng với hai biến khối lượng và biến động của đề cương đã đăng ký. Các ứng viên còn lại được đánh giá theo quy trình ở Mục 6. Các biến $G_t$ và $F_t$ tương quan mạnh nên không cùng xuất hiện trong một cấu hình nếu chưa kiểm tra đa cộng tuyến.

## 2.7. Chuẩn hóa không rò rỉ thông tin

Mỗi thành phần của $\mathbf{X}_{t-1}$ được chuẩn hóa $z$-score bằng trung bình và độ lệch chuẩn tính trên dữ liệu có đến thời điểm ước lượng lại gần nhất. Cũng như $\hat c_\rho$ và $\hat\varsigma(\cdot)$, các thống kê này không bao giờ dùng quan sát thuộc khối đang được dự báo. Giá trị khuyết của $O_t$ tại phiên chuyển được gán bằng $0$ sau chuẩn hóa, tức bằng trung bình huấn luyện.

## 2.8. Kiểm tra chất lượng dữ liệu

Trước khi ước lượng, các kiểm tra sau được thực hiện và ghi nhận trong báo cáo:

1.  Tính toàn vẹn OHLC: $P^{\mathrm l}_t\le\min(P^{\mathrm o}_t,P^{\mathrm c}_t)\le\max(P^{\mathrm o}_t,P^{\mathrm c}_t)\le P^{\mathrm h}_t$.

2.  Tính nhất quán khối lượng: $V_t=V^{+}_t+V^{-}_t$ (hoặc chênh lệch được giải thích bằng lệnh không phân loại).

3.  Phiên trùng lặp, phiên khuyết so với lịch giao dịch.

4.  Mọi phiên có $|r_t|>7\%$ sau khi đã loại phiên chuyển: phân loại thành lỗi dữ liệu hay sự kiện thực (đối chiếu biên độ giá).

5.  Danh sách phiên chạm biên và phiên chuyển hợp đồng.

## 2.9. Giới hạn của dữ liệu

- Không có giá kỳ hạn thứ hai: mất lợi suất tại khoảng $5\%$ số phiên; các cửa sổ tổn thất $H$ phiên chứa phiên chuyển bị loại khỏi đánh giá ký quỹ (Mục 7.6).

- Không có giá thanh toán hằng ngày: giá đóng cửa được dùng thay; sai khác (nếu có theo quy chế) là một giới hạn của kết quả ký quỹ.

- Không có dữ liệu trong ngày: đại lượng đại diện cho phương sai là ước lượng theo biên độ giá thay vì phương sai thực hiện.

- Không có chuỗi ký quỹ lịch sử: đối chứng thực tế cần được thu thập thủ công từ thông báo công khai của tổ chức bù trừ; nếu không thu thập được thì được báo cáo như giới hạn.

- Khối lượng chỉ của F1M: khối lượng bất thường được khử chu kỳ đáo hạn thay vì dùng tổng khối lượng mọi kỳ hạn.

# 3. Đặc tả mô hình EGARCH-X

## 3.1. Phương trình trung bình và phương sai

Mô hình được đặc tả cho các phiên có $r_t$ xác định: $$\begin{aligned}
r_t&=\mu+\varepsilon_t,\qquad \varepsilon_t=\sqrt{h_t}\,z_t,\qquad z_t\overset{\text{iid}}{\sim}\mathcal{G},\quad \mathbb{E}z_t=0,\ \operatorname{Var}z_t=1, \qquad(10)\\
\ln h_t&=\omega+s_{t-1}+\beta\ln h_{t-1}+\boldsymbol{\gamma}^\top\mathbf{X}_{t-1}+\boldsymbol{\delta}^\top\mathbf{D}_t, \qquad(11)\\
s_{t-1}&=(1-J_{t-1})\big[\kappa\big(|z_{t-1}|-\mu_{|z|}\big)+\theta z_{t-1}\big]. \qquad(12)
\end{aligned}$$ Số hạng $s_{t-1}$ là tác động của cú sốc chuẩn hóa phiên trước. Hiệu ứng độ lớn $\kappa(|z_{t-1}|-\mu_{|z|})$ có kỳ vọng bằng không và phản ứng với biên độ của cú sốc; hiệu ứng dấu $\theta z_{t-1}$ với $\theta<0$ biểu thị đòn bẩy: cú sốc âm làm tăng $\ln h_t$ nhiều hơn cú sốc dương cùng độ lớn \[38\]. Hệ số $\beta$ đo độ dai dẳng của logarit phương sai; $\boldsymbol{\gamma}$ là kênh thông tin ngoại sinh; $\boldsymbol{\delta}$ đo khác biệt biến động có tính lịch.

Dưới góc nhìn thống kê, (11) là một hồi quy tuyến tính của logarit phương sai có điều kiện lên trễ của chính nó, một hàm của phần dư chuẩn hóa và các biến giải thích. Hệ số $\gamma_j$ là mức thay đổi của $\ln h_t$ khi $X_{j,t-1}$ tăng một độ lệch chuẩn huấn luyện, các yếu tố khác cố định.

## 3.2. Phân phối sai số chuẩn hóa

Ba phân phối được so sánh, đều chuẩn hóa về kỳ vọng $0$ và phương sai $1$ để $h_t$ giữ nguyên ý nghĩa phương sai có điều kiện.

#### (i) Phân phối chuẩn.

$f_{\mathcal N}(z)=(2\pi)^{-1/2}\exp(-z^2/2)$ và $\mu_{|z|}=\sqrt{2/\pi}$.

#### (ii) Student-$t$ chuẩn hóa.

Với $\nu>2$, $$\tag{13}
f_{t^*_\nu}(z)=\frac{\Gamma\!\big(\frac{\nu+1}{2}\big)}{\sqrt{\pi(\nu-2)}\,\Gamma\!\big(\frac{\nu}{2}\big)}\Big(1+\frac{z^2}{\nu-2}\Big)^{-\frac{\nu+1}{2}},\qquad
\mu_{|z|}=\frac{2\sqrt{\nu-2}\,\Gamma\!\big(\frac{\nu+1}{2}\big)}{(\nu-1)\sqrt{\pi}\,\Gamma\!\big(\frac{\nu}{2}\big)} .$$ Thừa số $\nu-2$ thay cho $\nu$ là phép co giãn bảo đảm $\operatorname{Var}z_t=1$; điều kiện $\nu>2$ là điều kiện tồn tại của phép chuẩn hóa \[7\]. Khi $\nu\to\infty$, $\mu_{|z|}\to\sqrt{2/\pi}$, nhất quán với trường hợp chuẩn; đây là một phép kiểm tra bắt buộc khi cài đặt.

#### (iii) Student-$t$ lệch của Hansen.

Với $\nu>2$ và $-1<\lambda<1$, gọi $$\mathcal K_\nu=\frac{\Gamma\!\big(\frac{\nu+1}{2}\big)}{\sqrt{\pi(\nu-2)}\,\Gamma\!\big(\frac{\nu}{2}\big)},\qquad
A=4\lambda\,\mathcal K_\nu\,\frac{\nu-2}{\nu-1},\qquad
B=\sqrt{1+3\lambda^2-A^2}.$$ Mật độ chuẩn hóa là \[23\] $$\tag{15}
f_{\mathrm{SkT}}(z)=
\begin{cases}
B\,\mathcal K_\nu\Big(1+\dfrac{1}{\nu-2}\Big(\dfrac{Bz+A}{1-\lambda}\Big)^2\Big)^{-\frac{\nu+1}{2}}, & z<-A/B,\\[10pt]
B\,\mathcal K_\nu\Big(1+\dfrac{1}{\nu-2}\Big(\dfrac{Bz+A}{1+\lambda}\Big)^2\Big)^{-\frac{\nu+1}{2}}, & z\ge -A/B.
\end{cases}$$ Với $\lambda<0$ đuôi trái dày hơn đuôi phải; $\lambda=0$ cho lại (13). Hằng số $\mu_{|z|}$ được tính bằng cầu phương số một chiều tại mỗi giá trị $(\nu,\lambda)$ và kiểm tra lại với trường hợp $\lambda=0$.

Phân phối sai số tổng quát (GED), phương án thay thế cho một trong ba phân phối trên, được trình bày ở Phụ lục B.

## 3.3. Điều kiện chính quy và các lựa chọn cài đặt

1.  **Tính dương của phương sai.** Vì vế trái của (11) là $\ln h_t$, ta có $h_t>0$ với mọi giá trị tham số và mọi giá trị của $\mathbf{X}_{t-1}$; không cần ràng buộc dấu trên $\omega,\kappa,\theta,\boldsymbol{\gamma},\boldsymbol{\delta}$.

2.  **Tính dừng.** Quá trình $\ln h_t$ dừng khi $|\beta|<1$. Cài đặt dùng miền hẹp hơn $\beta\in(0,1)$: độ dai dẳng âm không có ý nghĩa kinh tế trong bối cảnh này, và miền tham số phải trùng nhau giữa hợp lý cực đại và suy diễn Bayes (tiên nghiệm của $\beta$ có giá trên $(0,1)$) để so sánh các phương pháp có nghĩa. Ràng buộc này được công bố cùng kết quả.

3.  **Bảo vệ số học.** $\ln h_t$ được chặn trong một khoảng hữu hạn rộng để $\exp(\cdot)$ không tràn số; số phiên chạm chặn được báo cáo và phải bằng không tại nghiệm.

4.  **Khởi tạo đệ quy.** $\ln h_1$ được đặt bằng logarit phương sai mẫu của $20$ lợi suất đầu tiên; độ nhạy với lựa chọn này được kiểm tra.

5.  **Khả nghịch.** Lý thuyết tiệm cận của ước lượng hợp lý cực đại cho EGARCH đòi hỏi đệ quy lọc khả nghịch, tức ảnh hưởng của giá trị khởi tạo tắt dần \[46\]. Điều kiện này được kiểm tra thực nghiệm bằng cách so sánh chuỗi $\ln h_t$ lọc từ hai giá trị khởi tạo khác nhau.

*Nhận xét 2* (Moment vô điều kiện dưới sai số Student-$t$). Từ (11), $\mathbb{E}[h_{t+1}\mid\mathcal{F}_{t-1}]$ chứa thừa số $\mathbb{E}\big[\exp(\kappa|z|+\theta z)\big]$, tức hàm sinh moment của $z$ tại các điểm khác không. Với $z$ có đuôi đa thức (Student-$t$, Student-$t$ lệch), đại lượng này vô hạn khi $\kappa+|\theta|>0$; do đó kỳ vọng phương sai từ hai bước trở đi và phương sai vô điều kiện không tồn tại theo nghĩa chặt. Nelson \[38\] dùng GED vì lý do này. Mô hình *có điều kiện* vẫn xác định đầy đủ: hàm hợp lý, dự báo một bước và mọi tiêu chí ở Mục 5 đều là đại lượng có điều kiện. Hệ quả thực hành là dự báo nhiều bước phải dùng phân vị và mô phỏng (Mục 7.2) chứ không dùng kỳ vọng phương sai, và mức sàn ký quỹ không được dựa trên phương sai vô điều kiện.

## 3.4. Quan sát khuyết tại phiên chuyển hợp đồng

Tại phiên $t$ với $J_t=1$, $r_t$ không xác định nên $z_t$ không xác định. Phiên này không đóng góp vào hàm hợp lý, và trong (12) số hạng cú sốc của nó được thay bằng kỳ vọng có điều kiện, bằng không, vì $\mathbb{E}\big[\kappa(|z|-\mu_{|z|})+\theta z\big]=0$. Đệ quy $\ln h_{t+1}$ vẫn được cập nhật qua $\beta\ln h_t$, $\mathbf{X}_t$ và $\mathbf{D}_{t+1}$. Cách xử lý này tương đương với việc lấy kỳ vọng của đóng góp cú sốc trên phân phối dự báo của quan sát khuyết, và giữ cho chuỗi $h_t$ liên tục qua các lần chuyển hợp đồng.

## 3.5. Hàm hợp lý

Gọi $\mathcal T$ là tập phiên dùng để ước lượng và $\mathcal T^{\circ}=\{t\in\mathcal T: J_t=0\}$. Gọi $$\boldsymbol{\psi}=(\mu,\omega,\kappa,\theta,\beta,\boldsymbol{\gamma}^\top,\boldsymbol{\delta}^\top,\boldsymbol{\psi}_{\mathcal{G}}^\top)^\top,$$ trong đó $\boldsymbol{\psi}_{\mathcal{G}}$ là tham số hình dạng ($\varnothing$, $\nu$ hoặc $(\nu,\lambda)$). Logarit hàm hợp lý là $$\tag{16}
\ell_T(\boldsymbol{\psi})=\sum_{t\in\mathcal T^{\circ}}\Big[-\tfrac12\ln h_t(\boldsymbol{\psi})+\ln f_{\mathcal{G}}\Big(\frac{r_t-\mu}{\sqrt{h_t(\boldsymbol{\psi})}};\boldsymbol{\psi}_{\mathcal{G}}\Big)\Big].$$ Số chiều tham số là $d=5+d_X+3+\dim\boldsymbol{\psi}_{\mathcal{G}}$; với cấu hình chính $d_X=2$, ta có $d=10$, $11$ và $12$ cho ba phân phối.

## 3.6. Mô hình lồng nhau và kiểm định CH1

EGARCH là trường hợp riêng $\boldsymbol{\gamma}=\mathbf 0$ của (11). Vì $\mathbf 0$ là điểm trong của không gian tham số của $\boldsymbol{\gamma}$, thống kê tỷ số hợp lý $$\mathrm{LR}=2\big[\ell_T(\hat{\boldsymbol{\psi}})-\ell_T(\tilde{\boldsymbol{\psi}})\big]\xrightarrow{d}\chi^2_{d_X}$$ dưới $H_0:\boldsymbol{\gamma}=\mathbf 0$, với $\tilde{\boldsymbol{\psi}}$ là nghiệm có ràng buộc. Kiểm định trong mẫu này bổ sung, chứ không thay thế, cho so sánh ngoài mẫu ở Mục 5.

# 4. Ước lượng và suy diễn tham số

## 4.1. Tham số hóa không ràng buộc

Mọi phương pháp làm việc trên cùng không gian không ràng buộc $\boldsymbol{\xi}\in\mathbb R^{d}$ qua song ánh trơn $\boldsymbol{\psi}=g(\boldsymbol{\xi})$ tác động theo từng tọa độ: $$\tag{18}
\beta=\operatorname{sigmoid}(\xi_\beta),\qquad \nu=2+\operatorname{softplus}(\xi_\nu),\qquad \lambda=\tanh(\xi_\lambda),$$ các tham số còn lại ($\mu,\omega,\kappa,\theta,\boldsymbol{\gamma},\boldsymbol{\delta}$) giữ nguyên. Ở đây $\operatorname{sigmoid}(x)=(1+e^{-x})^{-1}$ và $\operatorname{softplus}(x)=\ln(1+e^{x})$. Phép biến đổi bảo đảm $\beta\in(0,1)$, $\nu>2$ và $|\lambda|<1$ theo cấu trúc.

## 4.2. Ước lượng hợp lý cực đại

#### Tối ưu hóa.

$\hat{\boldsymbol{\psi}}=g(\hat{\boldsymbol{\xi}})$ với $\hat{\boldsymbol{\xi}}=\arg\max_{\boldsymbol{\xi}}\ell_T(g(\boldsymbol{\xi}))$, giải bằng L-BFGS với gradient tính bằng vi phân tự động. Vì bề mặt hợp lý có thể có nhiều cực trị địa phương, tối ưu được chạy từ ít nhất $20$ điểm khởi tạo: một phần từ nghiệm của mô hình đơn giản hơn ($\boldsymbol{\gamma}=\mathbf 0$, phân phối chuẩn), phần còn lại ngẫu nhiên quanh đó. Độ tán của giá trị $\ell_T$ tại các nghiệm được báo cáo như một chẩn đoán độc lập.

#### Sai số chuẩn.

Vì phân phối sai số có thể bị đặc tả sai, sai số chuẩn được tính theo dạng bánh kẹp của ước lượng hợp lý cực đại giả \[8\]: $$\widehat{\operatorname{Var}}(\hat{\boldsymbol{\psi}})=\hat{\mathbf A}^{-1}\hat{\mathbf B}\hat{\mathbf A}^{-1},\qquad
\hat{\mathbf A}=-\sum_{t}\frac{\partial^2\ell_t}{\partial\boldsymbol{\psi}\,\partial\boldsymbol{\psi}^\top}\Big|_{\hat{\boldsymbol{\psi}}},\qquad
\hat{\mathbf B}=\sum_{t}\frac{\partial\ell_t}{\partial\boldsymbol{\psi}}\frac{\partial\ell_t}{\partial\boldsymbol{\psi}^\top}\Big|_{\hat{\boldsymbol{\psi}}},$$ với $\ell_t$ là đóng góp của phiên $t$ vào (16). Sai số chuẩn dạng Hessian $\hat{\mathbf A}^{-1}$ được báo cáo song song; chênh lệch lớn giữa hai dạng là dấu hiệu đặc tả sai phân phối. Khi nghiệm nằm sát biên (ví dụ $\hat\beta$ sát $1$), khoảng tin cậy được bổ sung bằng bootstrap tham số.

#### Chẩn đoán.

\(i\) Kiểm định Ljung–Box trên $\hat z_t$ và $\hat z_t^2$; (ii) hồi quy $\hat z_t^2-1$ lên $\mathbf{D}_t$ và lên chỉ thị nhóm $\tau_t$, kèm kiểm định $F$ cho các hệ số bằng không, để xác nhận chu kỳ đáo hạn đã được hấp thụ; (iii) biến đổi tích phân xác suất $u_t=F_{\mathcal{G}}(\hat z_t)$ với kiểm định Berkowitz \[4\]; (iv) kiểm định điểm gãy của phương sai trên $\hat z_t$ \[31\]; (v) độ ổn định của $\hat\beta$ và $\hat{\boldsymbol{\gamma}}$ giữa các lần ước lượng lại.

## 4.3. Suy diễn Bayes

### 4.3.1. Phân phối tiên nghiệm

Giả định độc lập tiên nghiệm giữa các khối tham số. Bảng 5 là đặc tả ban đầu, có tính chất thông tin yếu trên thang của dữ liệu (lợi suất theo phần trăm, biến ngoại sinh đã chuẩn hóa). Đặc tả cuối cùng được xác nhận bằng kiểm tra dự báo tiên nghiệm (mô phỏng lợi suất từ tiên nghiệm và so sánh với thang của dữ liệu) và phân tích độ nhạy.

**Bảng 5.** Phân phối tiên nghiệm đề xuất

| **Tham số** | **Tiên nghiệm** | **Ghi chú** |
|---|---|---|
| $\mu$ | $\mathcal N(0,\,0{,}5^2)$ | Lợi suất ngày theo % |
| $\omega,\ \kappa,\ \theta$ | $\mathcal N(0,\,0{,}5^2)$ | Không ràng buộc dấu |
| $\beta$ | $\mathrm{Beta}(20;\,1{,}5)$ | Kỳ vọng $\approx 0{,}93$; giá trên $(0,1)$ |
| $\gamma_j$ | $\mathcal N(0,\,0{,}5^2)$ | Biến đã chuẩn hóa |
| $\delta_j$ | $\mathcal N(0,\,1^2)$ | Biến giả lịch |
| $\nu-2$ | $\mathrm{Gamma}(2;\,0{,}1)$ | Kỳ vọng $\mathbb{E}\nu=22$; cho phép đuôi rất dày |
| $(\lambda+1)/2$ | $\mathrm{Beta}(2;\,2)$ | Đối xứng quanh $\lambda=0$ |

### 4.3.2. Phân phối hậu nghiệm trên không gian không ràng buộc

Theo định lý Bayes, $\pi(\boldsymbol{\psi}\mid\mathcal D)\propto\pi(\boldsymbol{\psi})\,\mathcal L(\boldsymbol{\psi};\mathcal D)$. Suy diễn được thực hiện trên $\boldsymbol{\xi}$, nên theo công thức đổi biến $$\tag{20}
\ln\pi(\boldsymbol{\xi}\mid\mathcal D)=\ell_T\big(g(\boldsymbol{\xi})\big)+\ln\pi\big(g(\boldsymbol{\xi})\big)+\sum_{i=1}^{d}\ln\Big|\frac{\partial g_i(\xi_i)}{\partial\xi_i}\Big|+\text{hằng số},$$ với các đạo hàm $\partial\beta/\partial\xi_\beta=\beta(1-\beta)$, $\partial\nu/\partial\xi_\nu=\operatorname{sigmoid}(\xi_\nu)$, $\partial\lambda/\partial\xi_\lambda=1-\lambda^2$ và bằng $1$ cho các tham số không biến đổi. Thiếu số hạng Jacobian, mật độ trên $\boldsymbol{\xi}$ không còn là ảnh đúng của hậu nghiệm và mọi moment tính từ đó bị chệch.

### 4.3.3. Suy diễn biến phân

Suy diễn biến phân tìm $q^\star=\arg\min_{q\in\mathcal Q}\mathrm{KL}\big(q(\boldsymbol{\xi})\,\|\,\pi(\boldsymbol{\xi}\mid\mathcal D)\big)$, tương đương cực đại hóa cận dưới $$\mathrm{ELBO}(q)=\mathbb{E}_{q}\big[\ln\pi(\boldsymbol{\xi}\mid\mathcal D)+\ln Z\big]+\mathcal H(q),\qquad \mathcal H(q)=-\mathbb{E}_q[\ln q(\boldsymbol{\xi})],$$ với $Z$ là hằng số chuẩn hóa chưa biết \[5, 34\]. Hai họ phân phối được so sánh:

- **Mean-field:** $q_{\mathrm{MF}}(\boldsymbol{\xi})=\mathcal N\big(\mathbf m,\operatorname{diag}(s_1^2,\dots,s_d^2)\big)$, $2d$ tham số biến phân.

- **Full-rank:** $q_{\mathrm{FR}}(\boldsymbol{\xi})=\mathcal N(\mathbf m,\boldsymbol\Sigma)$, $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$ với $\mathbf L$ tam giác dưới có đường chéo dương, $d+d(d+1)/2$ tham số biến phân.

Gradient của ELBO được ước lượng không chệch bằng tái tham số hóa $\boldsymbol{\xi}=\mathbf m+\mathbf L\boldsymbol\zeta$, $\boldsymbol\zeta\sim\mathcal N(\mathbf 0,\mathbf I_d)$ \[32\]. Vì $\mathrm{KL}(q\|\pi)$ phạt nặng khối lượng của $q$ đặt ở vùng hậu nghiệm thấp, mean-field có xu hướng đánh giá thấp phương sai hậu nghiệm, đặc biệt khi các tham số tương quan mạnh như $\beta$ với $\omega$ và $\boldsymbol{\gamma}$. Chất lượng xấp xỉ được chẩn đoán bằng chỉ số $\hat k$ của lấy mẫu quan trọng làm trơn Pareto ($\hat k<0{,}7$ được coi là chấp nhận được) \[51\] và bằng độ ổn định của ELBO qua nhiều lần khởi tạo.

### 4.3.4. Monte Carlo Hamilton với NUTS

NUTS \[30\] sinh mẫu từ (20) và được dùng làm *hậu nghiệm tham chiếu*. Thiết lập: $4$ chuỗi, $1.000$ vòng khởi động và $1.000$ mẫu mỗi chuỗi, ma trận khối lượng đầy đủ thích nghi. Kết quả chỉ được sử dụng khi thỏa đồng thời: $\hat R<1{,}01$ (dạng chuẩn hóa thứ hạng), cỡ mẫu hiệu dụng phần thân và phần đuôi tối thiểu $400$, không có bước chuyển phân kỳ sau khởi động \[48\].

### 4.3.5. So sánh bốn phương pháp

Với NUTS làm tham chiếu, sai lệch của mỗi xấp xỉ được đo bằng: (i) tỷ số độ lệch chuẩn hậu nghiệm biên $\mathrm{sd}_q(\psi_i)/\mathrm{sd}_{\mathrm{NUTS}}(\psi_i)$; (ii) chênh lệch trung bình hậu nghiệm tính theo đơn vị độ lệch chuẩn tham chiếu; (iii) chuẩn Frobenius của hiệu hai ma trận tương quan hậu nghiệm; (iv) thời gian tính toán. Với hợp lý cực đại, độ lệch chuẩn được lấy từ ma trận bánh kẹp. Dự báo phân phối được so sánh theo Mục 5. Lưu ý rằng mật độ dự báo cắm tham số của hợp lý cực đại và mật độ dự báo hậu nghiệm của phương pháp Bayes là hai đối tượng khác nhau; sự khác biệt giữa chúng chính là ảnh hưởng của bất định tham số lên dự báo.

Để giới hạn khối lượng tính toán, suy diễn Bayes được thực hiện trên cấu hình được chọn ở Giai đoạn 1 (biến ngoại sinh và phân phối thắng), không trên toàn bộ lưới.

## 4.4. Nghiên cứu mô phỏng khôi phục tham số

Trước khi áp dụng cho dữ liệu thật, bộ ước lượng được kiểm tra trên dữ liệu mô phỏng từ tham số $\boldsymbol{\psi}_0$ đã biết, với $T=2.000$ quan sát (sau $200$ quan sát khởi động) và $B=100$ lần lặp cho mỗi kịch bản. Các tham số chung: $\mu_0=0{,}05$, $\omega_0=0{,}04$, $\kappa_0=0{,}15$, $\beta_0=0{,}95$, $\boldsymbol{\delta}_0=\mathbf 0$. Mức $\omega_0/(1-\beta_0)=0{,}8$ cho $\exp(0{,}8)\approx2{,}2$, cùng bậc với phương sai mẫu của lợi suất VN30F1M ($\approx2{,}27$).

**Bảng 6.** Kịch bản mô phỏng khôi phục tham số

| **KB** | $\theta_0$ | $\boldsymbol{\gamma}_0$ | $\nu_0$ | $\lambda_0$ | **Câu hỏi** |
|---|---|---|---|---|---|
| $E_1$ | $-0{,}10$ | $(0{,}05;\,0{,}05)$ | $8$ | $-0{,}1$ | Khôi phục đúng khi có đòn bẩy? |
| $E_2$ | $0$ | $(0{,}05;\,0{,}05)$ | $8$ | $-0{,}1$ | Có tạo ra đòn bẩy giả khi không có? |
| $E_3$ | $-0{,}10$ | $(0{,}05;\,0{,}05)$ | $4$ | $-0{,}1$ | Còn đúng khi đuôi sát biên tồn tại moment bậc bốn? |
| $E_4$ | $-0{,}10$ | $(0;\,0)$ | $8$ | $-0{,}1$ | Có tạo ra hiệu ứng ngoại sinh giả khi không có? |

Với phân phối chuẩn, bỏ $\nu_0,\lambda_0$; với Student-$t$, bỏ $\lambda_0$. Các tiêu chí: độ chệch $\frac1B\sum_b(\hat\psi^{(b)}-\psi_0)$; căn trung bình bình phương sai số; độ phủ của khoảng $95\%$ (khoảng Wald cho hợp lý cực đại, khoảng tin cậy Bayes cho các phương pháp còn lại); và trong $E_2$, $E_4$, tỷ lệ bác bỏ sai $H_0:\theta=0$ và $H_0:\boldsymbol{\gamma}=\mathbf 0$ ở mức $5\%$. Kịch bản $E_2$ và $E_4$ kiểm tra độ đặc hiệu của bộ ước lượng: nếu quy trình tạo ra hiệu ứng khi không có, mọi kết luận thực nghiệm về hiệu ứng đó mất căn cứ. Phương pháp hiệu chuẩn dựa trên mô phỏng \[47\] được áp dụng bổ sung cho suy diễn Bayes.

# 5. Thiết kế đánh giá ngoài mẫu

## 5.1. Chia mẫu theo thời gian

Dữ liệu được chia theo thứ tự thời gian thành ba khối không giao nhau (số phiên là dự kiến, ngày cụ thể được chốt sau bước làm sạch):

**Bảng 7.** Ba khối thời gian

| **Khối** | **Số phiên** | **Vai trò** |
|---|---|---|
| $\mathcal T_{\mathrm{UL}}$ | $\approx 750$ | Ước lượng ban đầu; tính $\hat c_\rho$, $\hat\varsigma(\cdot)$, thống kê chuẩn hóa. |
| $\mathcal T_{\mathrm{XT}}$ | $\approx 420$ | Xác thực: chọn biến ngoại sinh, phân phối sai số, siêu tham số ký quỹ. |
| $\mathcal T_{\mathrm{KT}}$ | $\approx 1.000$ | Kiểm tra: chỉ dùng để báo cáo; không có lựa chọn nào được thực hiện trên khối này. |

Trong cả $\mathcal T_{\mathrm{XT}}$ và $\mathcal T_{\mathrm{KT}}$, dự báo được tạo theo cửa sổ mở rộng: tham số được ước lượng lại mỗi $21$ phiên (khoảng một tháng) trên toàn bộ dữ liệu trước đó; giữa hai lần ước lượng lại, đệ quy (11) được cập nhật hằng ngày với tham số cố định. Như vậy mọi dự báo cho phiên $t+1$ chỉ dùng thông tin trong $\mathcal{F}_t$.

#### Lý do chọn cỡ khối kiểm tra.

Bảng 8 cho công suất chính xác (theo phân phối nhị thức) của kiểm định Kupiec ở mức ý nghĩa $5\%$, khi tỷ lệ vi phạm thực gấp khoảng $1{,}5$–$2$ lần mức danh nghĩa. Với khối $420$ phiên, công suất chỉ khoảng $0{,}4$–$0{,}5$; với $1.000$ phiên, công suất khoảng $0{,}75$–$0{,}78$. Vì vậy việc lựa chọn trên $\mathcal T_{\mathrm{XT}}$ dựa vào các điểm số liên tục (điểm logarit mật độ, điểm VaR–ES), vốn có công suất cao hơn nhiều so với kiểm định dựa trên số lần vi phạm.

**Bảng 8.** Công suất của kiểm định Kupiec (mức ý nghĩa $5\%$)

| $\alpha$ danh nghĩa | Tỷ lệ thực | $N\alpha$ khi $N=1.000$ | Công suất, $N=420$ | Công suất, $N=1.000$ |
|---|---|---|---|---|
| $1\%$ | $2\%$ | $10$ | $0{,}46$ | $0{,}78$ |
| $2{,}5\%$ | $4\%$ | $25$ | $0{,}42$ | $0{,}76$ |
| $5\%$ | $7\%$ | $50$ | $0{,}41$ | $0{,}75$ |

## 5.2. Dự báo phương sai

Gọi $\hat h_t$ là dự báo phương sai một bước tạo tại cuối phiên $t-1$ và $\rho^{\star}_t$ là đại lượng đại diện (6). Tiêu chí chính là $$\mathrm{QLIKE}=\frac1N\sum_{t}\Big[\ln\hat h_t+\frac{\rho^{\star}_t}{\hat h_t}\Big],$$ tổng lấy trên các phiên có $\rho^{\star}_t$ xác định và không chạm biên. QLIKE thuộc lớp hàm mất mát cho thứ hạng nhất quán khi phương sai thật được thay bằng đại lượng đại diện có kỳ vọng có điều kiện bằng phương sai thật \[41\]. Điều kiện này là lý do phải hiệu chỉnh thang đo ở (6): nếu $\mathbb{E}[\rho_t\mid\mathcal{F}_{t-1}]=c\,h_t$ với $c\ne1$, kỳ vọng của QLIKE đạt cực tiểu tại $\hat h_t=c\,h_t$ thay vì $h_t$. Sai số tuyệt đối trung bình và căn trung bình bình phương sai số được báo cáo bổ sung nhưng không dùng làm căn cứ kết luận.

## 5.3. Dự báo mật độ

Gọi $\hat f_t$ là mật độ dự báo của $r_t$. Điểm logarit mật độ trung bình $$\mathrm{LS}=\frac1N\sum_t\ln\hat f_t(r_t),\qquad
\hat f_t(r)=\frac{1}{\sqrt{\hat h_t}}\,f_{\mathcal{G}}\Big(\frac{r-\hat\mu}{\sqrt{\hat h_t}};\hat{\boldsymbol{\psi}}_{\mathcal{G}}\Big)$$ là quy tắc chấm điểm chặt; giá trị lớn hơn là tốt hơn. Với suy diễn Bayes, $\hat f_t$ là mật độ dự báo hậu nghiệm $\hat f_t(r)=\frac1{S}\sum_{s=1}^{S}f\big(r\mid\mathcal{F}_{t-1};\boldsymbol{\psi}^{(s)}\big)$ với $\boldsymbol{\psi}^{(s)}$ rút từ hậu nghiệm. Hiệu chuẩn của dự báo mật độ được kiểm tra bằng biến đổi tích phân xác suất \[13, 4\].

## 5.4. Dự báo VaR và ES một phiên

Gọi $L_t=-r_t$ là tổn thất (theo %) của vị thế mua trong một phiên, $\alpha$ là mức đuôi và $p=1-\alpha$. Theo quy ước của tài liệu, $\mathrm{VaR}$ là phân vị mức $p$ của tổn thất (một số dương): $$\mathrm{VaR}_{t,\alpha}=Q_{L_t\mid\mathcal{F}_{t-1}}(1-\alpha),\qquad
\mathrm{ES}_{t,\alpha}=\frac1\alpha\int_{1-\alpha}^{1}Q_{L_t\mid\mathcal{F}_{t-1}}(u)\,du .$$ Với phân phối liên tục, $\mathrm{ES}_{t,\alpha}=\mathbb{E}[L_t\mid L_t\ge\mathrm{VaR}_{t,\alpha},\mathcal{F}_{t-1}]$, và có biểu diễn tối ưu hóa $\mathrm{ES}_\alpha=\min_v\{v+\alpha^{-1}\mathbb{E}[(L-v)^+]\}$ với cực tiểu đạt tại $v=\mathrm{VaR}_\alpha$ \[43\]. Vì $L_t=-\mu-\sqrt{h_t}\,z_t$, tính chất vị trí – thang đo của hàm phân vị cho $$\tag{25}
\widehat{\mathrm{VaR}}_{t,\alpha}=-\hat\mu-\sqrt{\hat h_t}\,Q_{\mathcal{G}}(\alpha),\qquad
\widehat{\mathrm{ES}}_{t,\alpha}=-\hat\mu-\sqrt{\hat h_t}\,\mathbb{E}_{\mathcal{G}}\big[z\mid z\le Q_{\mathcal{G}}(\alpha)\big].$$ Các biểu thức đóng của $Q_{\mathcal{G}}$ và kỳ vọng đuôi cho từng phân phối ở Phụ lục A. Với vị thế bán, $L_t=r_t$ và đuôi phải của $\mathcal{G}$ được dùng.

#### Kiểm định độ phủ.

Gọi $I_t=\mathbf{1}\{L_t>\widehat{\mathrm{VaR}}_{t,\alpha}\}$, $N_1=\sum_tI_t$ và $\hat\pi=N_1/N$. Dự báo phân vị đúng khi và chỉ khi $\{I_t\}$ là dãy Bernoulli$(\alpha)$ độc lập. Thống kê Kupiec cho độ phủ vô điều kiện \[35\]: $$LR_{uc}=-2\ln\frac{(1-\alpha)^{N-N_1}\alpha^{N_1}}{(1-\hat\pi)^{N-N_1}\hat\pi^{N_1}}\xrightarrow{d}\chi^2_1 .$$ Gọi $n_{ij}$ là số lần $I_t$ chuyển từ trạng thái $i$ sang $j$, $\pi_{01}=n_{01}/(n_{00}+n_{01})$, $\pi_{11}=n_{11}/(n_{10}+n_{11})$, $\pi=(n_{01}+n_{11})/(N-1)$. Thống kê độc lập và độ phủ có điều kiện của Christoffersen \[9\]: $$LR_{ind}=-2\ln\frac{(1-\pi)^{n_{00}+n_{10}}\pi^{n_{01}+n_{11}}}{(1-\pi_{01})^{n_{00}}\pi_{01}^{n_{01}}(1-\pi_{11})^{n_{10}}\pi_{11}^{n_{11}}}\xrightarrow{d}\chi^2_1,\qquad
LR_{cc}=LR_{uc}+LR_{ind}\xrightarrow{d}\chi^2_2 .$$ Kiểm định phân vị động \[16\] được bổ sung để phát hiện phụ thuộc của $I_t$ vào thông tin trễ.

#### Điểm nhất quán VaR–ES.

Với tổn thất dương và $0<v\le e$, hàm điểm FZ0 \[18, 42\] viết theo quy ước tổn thất là $$\tag{28}
S_\alpha(v,e;L)=\frac{1}{\alpha e}\,\mathbf{1}\{L>v\}\,(L-v)+\frac{v}{e}+\ln e-1 .$$ Kỳ vọng của $S_\alpha$ đạt cực tiểu tại cặp $(\mathrm{VaR}_\alpha,\mathrm{ES}_\alpha)$ thật: lấy đạo hàm theo $v$ cho điều kiện $\mathbb{P}(L>v)=\alpha$, và theo $e$ cho $e=\mathbb{E}[L\mid L>v]$. Điểm trung bình $\bar S_\alpha$ càng nhỏ càng tốt. Kiểm định ES dựa trên phần dư vượt ngưỡng \[37\] được báo cáo bổ sung.

#### Quan hệ với thước đo rủi ro.

VaR là phân vị và có thể đánh giá được bằng hàm mất mát phân vị \[33, 22\]; ES là thước đo rủi ro nhất quán \[2, 1\] nhưng chỉ có thể đánh giá được cùng với VaR. Đây là lý do dùng (28) làm tiêu chí chính cho dự báo rủi ro đuôi.

## 5.5. So sánh dự báo có kiểm định

Gọi $\ell^{(a)}_t,\ell^{(b)}_t$ là mất mát (QLIKE, $-\ln\hat f_t(r_t)$ hoặc $S_\alpha$) của hai cấu hình $a,b$ tại phiên $t$, $\Delta\ell_t=\ell^{(a)}_t-\ell^{(b)}_t$, $\overline{\Delta\ell}=\frac1N\sum_t\Delta\ell_t$ và $\hat\gamma_j$ là hiệp phương sai mẫu bậc $j$ của $\Delta\ell_t$. Thống kê Diebold–Mariano \[14\] với phương sai HAC dạng Newey–West \[39\]: $$DM=\frac{\overline{\Delta\ell}}{\sqrt{\widehat V}},\qquad
\widehat V=\frac1N\Big[\hat\gamma_0+2\sum_{j=1}^{J}\Big(1-\frac{j}{J+1}\Big)\hat\gamma_j\Big]\xrightarrow{d}\mathcal N(0,1)\ \text{dưới } H_0:\mathbb{E}[\Delta\ell_t]=0 .$$ Vì dự báo được tạo từ tham số ước lượng lại theo cửa sổ, cách diễn giải theo khả năng dự báo có điều kiện của Giacomini–White \[20\] được áp dụng. Khi so sánh đồng thời nhiều cấu hình, tập mô hình tin cậy \[28\] xác định tập cấu hình không thể phân biệt với cấu hình tốt nhất ở mức tin cậy cho trước.

## 5.6. Phân tích theo nhóm phiên

Mọi tiêu chí được báo cáo cho toàn khối và cho các nhóm: (i) cửa sổ quanh đáo hạn $\tau_t\le2$ và phiên sau chuyển hợp đồng; (ii) phiên sau kỳ nghỉ dài; (iii) các giai đoạn căng thẳng, xác định khách quan bằng các phiên có $\rho^{\star}_t$ thuộc $5\%$ lớn nhất của phân phối trượt $250$ phiên. Phân tích nhóm (i) kiểm tra trực tiếp liệu biến động trước và sau đáo hạn đã được mô hình hấp thụ hay chưa.

# 6. Quy trình chọn biến ngoại sinh

Năm ứng viên ở Bảng 4 tạo ra nhiều tổ hợp. Việc chọn tổ hợp tốt nhất rồi báo cáo giá trị $p$ của riêng nó dẫn tới thiên lệch do tìm kiếm. Quy trình sau được cố định trước khi nhìn vào $\mathcal T_{\mathrm{KT}}$:

1.  **Chỉ dùng $\mathcal T_{\mathrm{UL}}\cup\mathcal T_{\mathrm{XT}}$.** Khối kiểm tra không tham gia bất kỳ lựa chọn nào.

2.  **Chọn tiến từng biến.** Bắt đầu từ EGARCH ($\boldsymbol{\gamma}=\mathbf 0$, giữ $\mathbf{D}_t$). Ở mỗi bước, thêm ứng viên làm giảm nhiều nhất $-\mathrm{LS}$ trung bình trên $\mathcal T_{\mathrm{XT}}$; giữ biến nếu cải thiện có ý nghĩa theo kiểm định Diebold–Mariano một phía ở mức $5\%$ và không làm xấu điểm $\bar S_\alpha$. Dừng khi không còn cải thiện có ý nghĩa.

3.  **Kiểm soát đa kiểm định.** Câu hỏi “có ít nhất một ứng viên thực sự cải thiện dự báo hay không” được kiểm định bằng kiểm định SPA \[24\] (cải tiến của Reality Check \[50\]); các kiểm định tỷ số hợp lý $H_0:\gamma_j=0$ trong mẫu được hiệu chỉnh theo Holm.

4.  **Đa cộng tuyến.** Không đưa đồng thời các biến có tương quan mẫu vượt $0{,}8$; báo cáo hệ số phóng đại phương sai.

5.  **Độ ổn định.** $\hat\gamma_j$ phải giữ dấu và cùng bậc độ lớn qua các lần ước lượng lại; biến có hệ số đổi dấu theo thời gian bị loại.

6.  **Minh bạch.** Mọi ứng viên đã thử, kể cả biến bị loại, được báo cáo trong phụ lục kết quả.

Cấu hình chính $\mathbf{X}_{t-1}=(A_{t-1},G_{t-1})^\top$ luôn được báo cáo, bất kể kết quả chọn biến, để bảo đảm tính so sánh với đề cương đã đăng ký. Nếu cấu hình chọn được có $d_X\ne2$, đề cương được điều chỉnh tương ứng và thay đổi được ghi nhận.

# 7. Giai đoạn 2: Thiết kế cơ chế ký quỹ động

## 7.1. Tổn thất, phân vị và mục tiêu ký quỹ

Gọi $m_t$ là tỷ lệ ký quỹ (theo tỷ lệ giá trị hợp đồng) được ấn định sau khi đóng cửa phiên $t$ và áp dụng từ phiên $t+1$; ký quỹ bằng tiền trên một hợp đồng là $\mathcal M_t=m_tKP_t$. Gọi $H$ là số phiên cần để thanh lý vị thế khi thành viên không bổ sung ký quỹ. Với $R_{t,H}=P_{t+H}/P_t-1$, tổn thất theo tỷ lệ của hai phía là $$L^{\mathrm M}_{t,H}=-R_{t,H},\qquad L^{\mathrm B}_{t,H}=R_{t,H},$$ nên $L^{\mathrm M}_{t,H}=-L^{\mathrm B}_{t,H}$: tại mỗi thời điểm chỉ một phía chịu lỗ. VaR mức $\alpha$ của một tổn thất $L$ là phân vị mức $1-\alpha$ của chính nó; do $Q_{-Y}(1-\alpha)=-Q_Y(\alpha)$ với phân phối liên tục, $$\mathrm{VaR}^{\mathrm M}_{t,H,\alpha}=-Q_{R_{t,H}\mid\mathcal{F}_t}(\alpha),\qquad
\mathrm{VaR}^{\mathrm B}_{t,H,\alpha}=Q_{R_{t,H}\mid\mathcal{F}_t}(1-\alpha).$$ Vị thế mua chịu rủi ro ở đuôi trái của $R_{t,H}$, vị thế bán ở đuôi phải. Hai VaR bằng nhau khi phân phối có điều kiện của $R_{t,H}$ đối xứng; chúng khác nhau khi $\mathcal{G}$ lệch, và cả khi $H\ge2$ với $\theta\neq0$, vì cú sốc âm ở phiên $t+1$ làm tăng $h_{t+2}$ nhiều hơn cú sốc dương cùng độ lớn.

Yêu cầu “tổn thất vượt ký quỹ với xác suất $\alpha$” tương đương $$\mathbb{P}\big(L_{t,H}>m_t\mid\mathcal{F}_t\big)=\alpha\iff F_{L_{t,H}\mid\mathcal{F}_t}(m_t)=1-\alpha\iff m_t=Q_{L_{t,H}\mid\mathcal{F}_t}(1-\alpha),$$ tức ký quỹ cơ sở là một phân vị có điều kiện của tổn thất. Mức đuôi $\alpha$ và khoảng thanh lý $H$ là *tham số thiết kế* do người nghiên cứu ấn định; mô hình cung cấp phân phối có điều kiện, từ đó tính ra phân vị.

## 7.2. Dự báo rủi ro $H$ phiên bằng mô phỏng

Với $H=1$, (25) cho công thức đóng. Với $H\ge2$, $\ln h_{t+2}$ phụ thuộc $z_{t+1}$ qua hàm mũ, nên lợi suất tích lũy là tổng của các biến phụ thuộc, khác phân phối và đuôi dày. Chẳng hạn với $H=2$, $$\mathbb{P}\big(r_{t+1}+r_{t+2}\le x\mid\mathcal{F}_t\big)=\int_{\mathbb R}F_{\mathcal{G}}\!\left(\frac{x-2\mu-\sqrt{h_{t+1}}\,u}{\sqrt{h_{t+2}(u)}}\right)f_{\mathcal{G}}(u)\,du,$$ với $h_{t+2}(u)=\exp\{\omega+\kappa(|u|-\mu_{|z|})+\theta u+\beta\ln h_{t+1}+\boldsymbol{\gamma}^\top\mathbf{X}_{t+1}+\boldsymbol{\delta}^\top\mathbf{D}_{t+2}\}$. Tích phân này không có nguyên hàm sơ cấp; với $H$ tổng quát cần tích phân $H-1$ chiều, và $\mathbf{X}_{t+1},\dots$ chưa quan sát được. Do đó phân phối được xấp xỉ bằng mô phỏng $S=10^4$ đường:

1.  Với $s=1,\dots,S$: lấy $\boldsymbol{\psi}^{(s)}$ từ hậu nghiệm (suy diễn Bayes) hoặc đặt $\boldsymbol{\psi}^{(s)}=\hat{\boldsymbol{\psi}}$ (hợp lý cực đại).

2.  Đặt $\ln h^{(s)}_{t+1}$ bằng giá trị lọc tại cuối phiên $t$ (xác định theo $\mathcal{F}_t$).

3.  Với $j=1,\dots,H$: sinh $z^{(s)}_{t+j}\sim\mathcal{G}$; đặt $r^{(s)}_{t+j}=\sqrt{h^{(s)}_{t+j}}\,z^{(s)}_{t+j}$ (với $\mu=0$, lựa chọn thận trọng cho ký quỹ); sinh $\mathbf{X}^{(s)}_{t+j}$ từ mô hình tự hồi quy bậc một ước lượng trên cùng cửa sổ, với phần dư lấy mẫu lại; cập nhật $\ln h^{(s)}_{t+j+1}$ theo (11) với $\mathbf{D}_{t+j+1}$ theo lịch.

4.  $R^{(s)}=\exp\!\big(\sum_{j=1}^{H}r^{(s)}_{t+j}/100\big)-1$; $L^{\mathrm M,(s)}=-R^{(s)}$, $L^{\mathrm B,(s)}=R^{(s)}$.

5.  Với các tổn thất xếp giảm dần $L^{(1)}\ge\dots\ge L^{(S)}$ và $j^\ast=\lceil\alpha S\rceil$: $\widehat{\mathrm{VaR}}=L^{(j^\ast)}$, $\widehat{\mathrm{ES}}=\frac1{j^\ast}\sum_{i=1}^{j^\ast}L^{(i)}$.

Mô phỏng theo hậu nghiệm cho phân phối dự báo là hỗn hợp theo tham số, có đuôi dày hơn phân phối cắm tham số; đây là kênh qua đó CH2 (phương pháp suy diễn) ảnh hưởng tới CH3 (ký quỹ).

## 7.3. Quy tắc ký quỹ năm bước

Ký quỹ thường áp dụng cùng một tỷ lệ cho hai phía, nên gọi $\widehat{\mathrm{VaR}}_t=\max\{\widehat{\mathrm{VaR}}{}^{\mathrm M}_{t,H,\alpha},\widehat{\mathrm{VaR}}{}^{\mathrm B}_{t,H,\alpha}\}$ và $\widehat{\mathrm{ES}}_t$ tương tự.

#### Bước 1 – Mức cơ sở.

Với trọng số $w\in[0,1]$, $$m^{(1)}_t=(1-w)\,\widehat{\mathrm{VaR}}_t+w\,\widehat{\mathrm{ES}}_t .$$

#### Bước 2 – Hệ số đệm.

Với $b\ge0$, $m^{(2)}_t=(1+b)\,m^{(1)}_t$.

#### Bước 3 – Mức sàn chống thuận chu kỳ.

Gọi $\widehat{\mathrm{VaR}}{}^{\mathrm{LS}}_t$ là VaR $H$ phiên theo mô phỏng lịch sử trên toàn bộ dữ liệu đến $t$ (cửa sổ mở rộng, gồm các giai đoạn căng thẳng) và $\varphi\in(0,1]$: $$\underline{m}_t=\varphi\,\widehat{\mathrm{VaR}}{}^{\mathrm{LS}}_t,\qquad m^{(3)}_t=\max\{m^{(2)}_t,\underline{m}_t\}.$$

#### Bước 4 – Làm trơn bất đối xứng.

Với $\eta\in[0,1)$, $$m^{(4)}_t=\max\big\{m^{(3)}_t,\ \eta\,m^{(4)}_{t-1}+(1-\eta)\,m^{(3)}_t\big\}.$$ Khi mức mục tiêu tăng, $m^{(4)}_t=m^{(3)}_t$ (tăng ngay); khi giảm, $m^{(4)}_t$ giảm dần theo trung bình mũ. Về thống kê, đây là đánh đổi độ chệch lấy phương sai của một ước lượng phân vị nhiễu, theo hướng có lợi cho an toàn.

#### Bước 5 – Giới hạn tốc độ giảm và làm tròn.

Với $\Delta^{\downarrow}>0$ và bước làm tròn $g$, $$m_t=\max\Big\{\underline{m}_t,\ g\Big\lceil\tfrac{1}{g}\max\{m^{(4)}_t,\ m_{t-1}-\Delta^{\downarrow}\}\Big\rceil\Big\}.$$ Quy tắc không giới hạn tốc độ tăng; nếu thêm giới hạn tăng thì phải bù bằng hệ số đệm lớn hơn để tránh thiếu hụt khi biến động bùng nổ.

Vectơ siêu tham số là $\boldsymbol{\Lambda}=(H,\alpha,w,b,\varphi,\eta,\Delta^{\downarrow},g)$.

## 7.4. Hiệu chỉnh siêu tham số

Gọi $L_{t,H}=\max\{L^{\mathrm M}_{t,H},L^{\mathrm B}_{t,H}\}=|R_{t,H}|$. Chi phí kỳ vọng thực nghiệm của một quy tắc là $$\mathcal C(\boldsymbol{\Lambda})=\frac1N\sum_{t}\Big[c_{\mathrm v}\,m_t+c_{\mathrm{th}}\,\big(L_{t,H}-m_t\big)^{+}\Big].$$ Với $m$ cố định, đạo hàm của $\mathbb{E}[c_{\mathrm v}m+c_{\mathrm{th}}(L-m)^+]$ theo $m$ bằng $c_{\mathrm v}-c_{\mathrm{th}}\mathbb{P}(L>m)$, nên mức tối ưu thỏa $\mathbb{P}(L>m^\ast)=c_{\mathrm v}/c_{\mathrm{th}}$. Mức đuôi tối ưu là $\alpha^\ast=c_{\mathrm v}/c_{\mathrm{th}}$; bài toán tương đương cực tiểu hóa hàm mất mát phân vị \[33\]. Vì $c_{\mathrm{th}}$ (gộp xác suất thành viên mất khả năng thanh toán và tổn thất khi đó) không quan sát được, nghiên cứu không cố định một tỷ số chi phí mà báo cáo *đường biên*: với mỗi $\boldsymbol{\Lambda}$ trong lưới, tính cặp (ký quỹ trung bình $\bar m$, thiếu hụt bình quân) trên $\mathcal T_{\mathrm{XT}}$, và so sánh các quy tắc tại cùng $\bar m$. Lưới gồm $H\in\{1,2,3\}$, $\alpha\in\{5\%;2{,}5\%;1\%\}$, $w\in\{0;0{,}5;1\}$, $b\in\{0;0{,}1;0{,}25\}$, $\varphi\in\{0{,}5;0{,}75;1\}$, $\eta\in\{0;0{,}8;0{,}95\}$. Cấu hình được chọn giữ cố định trên $\mathcal T_{\mathrm{KT}}$.

## 7.5. Quy tắc đối chứng

1.  Ký quỹ cố định, mức được chọn trên $\mathcal T_{\mathrm{XT}}$ sao cho tỷ lệ vượt bằng của quy tắc đề xuất.

2.  VaR theo trung bình trượt hàm mũ với hệ số $0{,}94$.

3.  Mô phỏng lịch sử $250$ phiên.

4.  Mô phỏng lịch sử có lọc theo EGARCH-X \[37\]: lấy mẫu lại phần dư chuẩn hóa thay cho $\mathcal{G}$ tham số.

5.  Chuỗi tỷ lệ ký quỹ lịch sử của tổ chức bù trừ, nếu thu thập được.

6.  Mốc trên $m=H\times$ biên độ giá ngày, nếu biên độ được xác minh: bảo đảm đủ cho $H$ phiên chạm biên liên tiếp.

## 7.6. Đánh giá ký quỹ

Gọi $B_t=\mathbf{1}\{L_{t,H}>m_t\}$. Các chỉ tiêu trên $\mathcal T_{\mathrm{KT}}$: $$\text{Tỷ lệ vượt}=\frac1N\sum_tB_t,\qquad
\text{Thiếu hụt bình quân}=\frac{\sum_t(L_{t,H}-m_t)^{+}}{\sum_tB_t},\qquad
\bar m=\frac1N\sum_tm_t,$$ cùng thiếu hụt lớn nhất, ký quỹ lớn nhất, độ lệch chuẩn của $m_t-m_{t-1}$, số lần điều chỉnh và tỷ số thuận chu kỳ $\max_t m_{t+20}/m_t$.

*Nhận xét 3* (Phụ thuộc do cửa sổ chồng lấn). Với $H\ge2$, các tổn thất $L_{t,H}$ chồng lấn nên $B_t$ có tương quan chuỗi đến bậc $H-1$; kiểm định Kupiec và Christoffersen giả định độc lập nên sai cỡ. Kiểm định được thực hiện trên dãy lấy thưa mỗi $H$ phiên, và kiểm định phân vị động với phương sai HAC được báo cáo bổ sung.

*Nhận xét 4* (Cửa sổ chứa phiên chuyển hợp đồng). Một cửa sổ $(t,t+H]$ chứa phiên chuyển không cho tổn thất đúng của một vị thế, do thiếu lợi suất cùng hợp đồng tại phiên chuyển. Các cửa sổ này bị loại khỏi đánh giá; với $H=2$ hoặc $3$, tỷ lệ bị loại khoảng $10$–$14\%$ và được báo cáo.

#### Kiểm tra độ bền.

Thay $H$, $\alpha$, $w$, $\eta$; đánh giá riêng các giai đoạn căng thẳng, cửa sổ quanh đáo hạn và sau kỳ nghỉ dài; so sánh ký quỹ từ hợp lý cực đại và từ phân phối dự báo hậu nghiệm.

# 8. Kế hoạch thực hiện

## 8.1. Quy trình tổng thể

1.  **Giai đoạn 1 – Mô hình rủi ro.** Làm sạch dữ liệu và xây dựng biến (Mục 2); mô phỏng khôi phục tham số (Mục 4.4); ước lượng hợp lý cực đại EGARCH-X với ba phân phối; chọn biến ngoại sinh và phân phối trên $\mathcal T_{\mathrm{XT}}$; suy diễn Bayes trên cấu hình được chọn; đánh giá trên $\mathcal T_{\mathrm{KT}}$.

2.  **Điểm bàn giao.** Lưu theo từng ngày ước lượng lại: $\hat{\boldsymbol{\psi}}$ hoặc mẫu hậu nghiệm; chuỗi $\ln\hat h_{t+1}$ lọc; mô hình tự hồi quy của $\mathbf{X}$; phần dư chuẩn hóa; lịch $\tau_t$, $\mathbf{D}_t$; lịch ước lượng lại.

3.  **Giai đoạn 2 – Ký quỹ.** Bộ mô phỏng $H$ phiên; quy tắc năm bước; hiệu chỉnh $\boldsymbol{\Lambda}$ trên $\mathcal T_{\mathrm{XT}}$; đánh giá và so sánh đối chứng trên $\mathcal T_{\mathrm{KT}}$; kiểm tra độ bền.

Điều kiện chuyển từ Giai đoạn 1 sang Giai đoạn 2: (i) mô phỏng khôi phục đạt độ phủ gần $95\%$ và tỷ lệ bác bỏ sai gần $5\%$ trong $E_2$, $E_4$; (ii) trung bình $\hat z_t^2$ theo nhóm $\tau_t$ không khác $1$ có ý nghĩa; (iii) dự báo VaR một phiên của cấu hình được chọn không bị bác bỏ trên $\mathcal T_{\mathrm{XT}}$.

## 8.2. Tiến độ

**Bảng 9.** Tiến độ dự kiến

| **Tháng** | **Công việc** | **Kết quả** | **Lũy kế** |
|---|---|---|---|
| 10/2026 | Làm sạch dữ liệu, xác minh thông số thể chế, xây dựng $\rho^{\star}_t$, biến lịch, biến ngoại sinh; mô phỏng khôi phục tham số với hợp lý cực đại. | Bộ dữ liệu phân tích; báo cáo kiểm tra dữ liệu; kết quả $E_1$–$E_4$. | $15\%$ |
| 11/2026 | Ước lượng hợp lý cực đại EGARCH-X với ba phân phối trên $\mathcal T_{\mathrm{UL}}$; chẩn đoán; kiểm định $\boldsymbol{\gamma}=\mathbf 0$. | Bảng tham số và chẩn đoán. | $30\%$ |
| 12/2026 | Dự báo tiến lên trên $\mathcal T_{\mathrm{XT}}$; chọn biến ngoại sinh và phân phối theo Mục 6. | Cấu hình được chọn; nhật ký chọn biến. | $45\%$ |
| 01/2027 | Suy diễn biến phân mean-field, full-rank và NUTS trên cấu hình được chọn; so sánh với hợp lý cực đại; hiệu chuẩn dựa trên mô phỏng. | Bảng so sánh bốn phương pháp. | $58\%$ |
| 02/2027 | Đánh giá Giai đoạn 1 trên $\mathcal T_{\mathrm{KT}}$: QLIKE, điểm logarit mật độ, kiểm định VaR, điểm VaR–ES, Diebold–Mariano, tập mô hình tin cậy. | Kết quả trả lời CH1, CH2. | $70\%$ |
| 03/2027 | Bộ mô phỏng $H$ phiên; quy tắc ký quỹ; hiệu chỉnh $\boldsymbol{\Lambda}$ trên $\mathcal T_{\mathrm{XT}}$; thu thập chuỗi ký quỹ lịch sử. | Thuật toán và chuỗi ký quỹ trên khối xác thực. | $82\%$ |
| 04/2027 | Đánh giá ký quỹ trên $\mathcal T_{\mathrm{KT}}$; so sánh đối chứng; kiểm tra độ bền; hoàn thiện báo cáo. | Kết quả trả lời CH3; bản thảo báo cáo. | $100\%$ |
| 05/2027 | Báo cáo cấp Trường; chỉnh sửa; chuẩn bị hồ sơ dự giải (nếu có). | Báo cáo thuyết trình, phiên bản cuối. | – |

## 8.3. Sản phẩm

1.  Bộ mã phân tích có khả năng tái lập, kèm tệp cấu hình và nhật ký chạy.

2.  Bộ dữ liệu phân tích và báo cáo kiểm tra chất lượng dữ liệu.

3.  Báo cáo mô phỏng khôi phục tham số.

4.  Bảng tham số, chẩn đoán và kết quả dự báo ngoài mẫu của EGARCH-X với ba phân phối và bốn phương pháp suy diễn.

5.  Công thức, thuật toán và chuỗi tỷ lệ ký quỹ động theo ngày; báo cáo so sánh với các quy tắc đối chứng.

6.  Báo cáo tổng kết công trình nghiên cứu khoa học sinh viên.

## 8.4. Rủi ro và phương án dự phòng

| **Rủi ro** | **Phương án** |
|---|---|
| Ước lượng hợp lý cực đại không hội tụ hoặc nghiệm sát biên | Tăng số điểm khởi tạo; khởi tạo từ mô hình lồng; báo cáo khoảng tin cậy bootstrap tham số. |
| NUTS có bước chuyển phân kỳ | Tăng hệ số chấp nhận đích; tham số hóa lại; nếu vẫn thất bại, báo cáo như kết quả âm có chẩn đoán đi kèm. |
| Xấp xỉ biến phân kém ($\hat k\ge0{,}7$) | Báo cáo như một phát hiện về giới hạn của phương pháp; dùng NUTS cho Giai đoạn 2. |
| Không thu thập được chuỗi ký quỹ lịch sử | Dùng các đối chứng còn lại; nêu rõ giới hạn. |
| Không biến ngoại sinh nào cải thiện có ý nghĩa | Báo cáo kết quả âm cho CH1; Giai đoạn 2 dùng EGARCH với biến giả lịch. |
| Thông số thể chế thay đổi trong mẫu | Mã hóa thông số theo thời kỳ; phân tích độ bền theo thời kỳ. |
| Thời gian tính toán vượt dự kiến | Giảm tần suất ước lượng lại NUTS (ví dụ $42$ phiên) và ghi nhận; thuê tài nguyên tính toán theo dự toán kinh phí. |

## 8.5. Tái lập

Toàn bộ phân tích được viết bằng Python, dùng NumPy, SciPy, pandas cho xử lý dữ liệu và tối ưu, JAX với NumPyro (hoặc CmdStanPy) cho vi phân tự động, suy diễn biến phân và NUTS. Mọi lần chạy có hạt giống ngẫu nhiên cố định, tệp cấu hình ghi lại toàn bộ lựa chọn (khối thời gian, biến, phân phối, siêu tham số), và môi trường phần mềm được khóa phiên bản. Mọi dự báo ngoài mẫu được lưu theo ngày để các bảng kết quả có thể tính lại mà không cần ước lượng lại. Mã nguồn được quản lý bằng hệ thống kiểm soát phiên bản.

## 8.6. Dự trù kinh phí

| **STT** | **Diễn giải** | **Số tiền (đồng)** |
|---|---|---|
| 1 | Chi phí điện toán, thuê tài nguyên tính toán phục vụ mô phỏng và MCMC | $2.500.000$ |
| 2 | In ấn, đóng quyển và văn phòng phẩm | $500.000$ |
|  | **Tổng cộng** | $\mathbf{3.000.000}$ |

Bằng chữ: Ba triệu đồng chẵn. Đề nghị nhà trường hỗ trợ: $3.000.000$ đồng.

## 8.7. Giới hạn của nghiên cứu

1.  Đại lượng đại diện cho phương sai dựa trên biên độ giá ngày, kém chính xác hơn phương sai thực hiện từ dữ liệu trong ngày.

2.  Lợi suất tại phiên chuyển hợp đồng bị loại do thiếu giá kỳ hạn thứ hai.

3.  Giá đóng cửa được dùng thay giá thanh toán hằng ngày.

4.  Biên độ giá ngày làm lợi suất bị kiểm duyệt tại biên; đuôi phân phối quan sát được có thể mỏng hơn đuôi thực của giá cân bằng. Nghiên cứu ghi nhận và phân tích riêng các phiên chạm biên nhưng không mô hình hóa kiểm duyệt.

5.  Quy tắc ký quỹ được đánh giá trên dữ liệu lịch sử của một hợp đồng; không mô phỏng hành vi của thành viên bù trừ, tài sản bảo đảm hay phản ứng của thị trường đối với thay đổi ký quỹ.

# Phụ lục A. Công thức VaR và ES một phiên theo phân phối

Gọi $q_\alpha=Q_{\mathcal{G}}(\alpha)$ là phân vị mức $\alpha$ của $z$ và $e_\alpha=\mathbb{E}_{\mathcal{G}}[z\mid z\le q_\alpha]$ là kỳ vọng đuôi trái. Theo (25), với vị thế mua $\widehat{\mathrm{VaR}}_{t,\alpha}=-\hat\mu-\sqrt{\hat h_t}\,q_\alpha$ và $\widehat{\mathrm{ES}}_{t,\alpha}=-\hat\mu-\sqrt{\hat h_t}\,e_\alpha$. Gọi $t_\nu^{-1}$ và $f_\nu$ là hàm phân vị và hàm mật độ của phân phối Student-$t$ chưa chuẩn hóa với $\nu$ bậc tự do, và $c_\nu=\sqrt{(\nu-2)/\nu}$.

#### Phân phối chuẩn.

Với $\Phi$ và $f_{\mathcal N}$ là hàm phân phối và hàm mật độ chuẩn tắc, $$q_\alpha=\Phi^{-1}(\alpha),\qquad e_\alpha=-\frac{f_{\mathcal N}(q_\alpha)}{\alpha}.$$

#### Student-$t$ chuẩn hóa.

Với $t_\alpha=t_\nu^{-1}(\alpha)$, $$q_\alpha=c_\nu\,t_\alpha,\qquad e_\alpha=-c_\nu\,\frac{f_\nu(t_\alpha)}{\alpha}\,\frac{\nu+t_\alpha^2}{\nu-1},$$ suy từ $\int_{-\infty}^{x}u\,f_\nu(u)\,du=-f_\nu(x)\,(\nu+x^2)/(\nu-1)$.

#### Student-$t$ lệch của Hansen.

Với $\mathcal K_\nu,A,B$ như ở Mục 3.2, hàm phân phối là $F(z)=(1-\lambda)F_{t^*_\nu}\big(\tfrac{Bz+A}{1-\lambda}\big)$ khi $z<-A/B$ và $F(z)=(1+\lambda)F_{t^*_\nu}\big(\tfrac{Bz+A}{1+\lambda}\big)-\lambda$ khi $z\ge-A/B$. Nghịch đảo cho $$q_\alpha=
\begin{cases}
\dfrac{(1-\lambda)\,c_\nu\,t_\nu^{-1}\!\big(\frac{\alpha}{1-\lambda}\big)-A}{B}, & \alpha<\frac{1-\lambda}{2},\\[12pt]
\dfrac{(1+\lambda)\,c_\nu\,t_\nu^{-1}\!\big(\frac{\alpha+\lambda}{1+\lambda}\big)-A}{B}, & \alpha\ge\frac{1-\lambda}{2}.
\end{cases}$$ Khi $q_\alpha<-A/B$ (trường hợp của các mức đuôi nhỏ), gọi $x_\alpha=t_\nu^{-1}\big(\alpha/(1-\lambda)\big)$; phép đổi biến $z=\big((1-\lambda)c_\nu x-A\big)/B$ cho $$e_\alpha=\frac{1-\lambda}{\alpha B}\Big[-(1-\lambda)\,c_\nu\,f_\nu(x_\alpha)\,\frac{\nu+x_\alpha^2}{\nu-1}-A\,F_\nu(x_\alpha)\Big],$$ với $F_\nu(x_\alpha)=\alpha/(1-\lambda)$. Kỳ vọng đuôi phải (cho vị thế bán) được tính tương tự trên nhánh $z\ge-A/B$ hoặc bằng cầu phương số.

#### Ví dụ số.

Bảng 11 cho $\mathrm{VaR}$ và $\mathrm{ES}$ một phiên với $\mu=0$ và $\sqrt{h_t}=1\%$; các giá trị đã được kiểm tra bằng tích phân số. Với $\sqrt{h_t}$ khác, nhân tuyến tính. Với Student-$t$ lệch, $\lambda=-0{,}1$ làm đuôi trái dày hơn: VaR của vị thế mua lớn hơn của vị thế bán.

**Bảng 11.** VaR và ES một phiên (%), $\mu=0$, $\sqrt{h_t}=1\%$

|  | $\alpha=1\%$ |  | $\alpha=2{,}5\%$ |  | $\alpha=5\%$ |  |
|---|---|---|---|---|---|---|
| Phân phối | VaR | ES | VaR | ES | VaR | ES |
| Chuẩn | $2{,}326$ | $2{,}665$ | $1{,}960$ | $2{,}338$ | $1{,}645$ | $2{,}063$ |
| Student-$t$, $\nu=5$ | $2{,}606$ | $3{,}449$ | $1{,}991$ | $2{,}728$ | $1{,}561$ | $2{,}239$ |
| SkT $\nu=5,\lambda=-0{,}1$, vị thế mua | $2{,}783$ | $3{,}720$ | $2{,}102$ | $2{,}919$ | $1{,}627$ | $2{,}377$ |
| SkT $\nu=5,\lambda=-0{,}1$, vị thế bán | $2{,}416$ | $3{,}159$ | $1{,}871$ | $2{,}522$ | $1{,}488$ | $2{,}089$ |

# Phụ lục B. Phân phối sai số tổng quát (phương án thay thế)

Nếu nhóm nghiên cứu chọn GED thay cho một trong ba phân phối ở Mục 3.2, mật độ chuẩn hóa với tham số hình dạng $\nu>0$ là \[38\] $$f_{\mathrm{GED}}(z)=\frac{\nu\exp\!\big(-\tfrac12|z/\lambda_\nu|^{\nu}\big)}{\lambda_\nu\,2^{1+1/\nu}\,\Gamma(1/\nu)},\qquad
\lambda_\nu=\Big[\frac{2^{-2/\nu}\,\Gamma(1/\nu)}{\Gamma(3/\nu)}\Big]^{1/2},\qquad
\mu_{|z|}=\lambda_\nu\,2^{1/\nu}\,\frac{\Gamma(2/\nu)}{\Gamma(1/\nu)} .$$ Trường hợp $\nu=2$ là phân phối chuẩn; $\nu<2$ cho đuôi dày hơn chuẩn. Với $\nu>1$, hàm sinh moment hữu hạn nên phương sai vô điều kiện của EGARCH tồn tại khi $|\beta|<1$, khác với Student-$t$. Tham số hóa không ràng buộc: $\nu=\operatorname{softplus}(\xi_\nu)$; tiên nghiệm đề xuất $\nu\sim\mathrm{Gamma}(4;\,2)$ (kỳ vọng $2$).

# Phụ lục C. Danh mục kiểm tra trước khi chạy ước lượng chính thức

1.  Thông số ở Bảng 3 đã được xác minh theo từng thời kỳ.

2.  Danh sách phiên chuyển hợp đồng, phiên đáo hạn, phiên chạm biên, phiên sau kỳ nghỉ dài đã được lập và đối chiếu lịch.

3.  Các kiểm tra chất lượng dữ liệu đã hoàn tất và được ghi nhận.

4.  $\hat c_\rho$, $\hat\varsigma(\cdot)$ và thống kê chuẩn hóa chỉ dùng dữ liệu trước điểm ước lượng lại.

5.  $\mu_{|z|}$ của từng phân phối khớp với giá trị tính bằng tích phân số; trường hợp giới hạn ($\nu\to\infty$, $\lambda=0$) cho lại phân phối đơn giản hơn.

6.  Mô phỏng khôi phục tham số đạt yêu cầu ở Mục 8.

7.  Khối $\mathcal T_{\mathrm{KT}}$ chưa được dùng cho bất kỳ lựa chọn nào.

8.  Tệp cấu hình và hạt giống ngẫu nhiên đã được cố định và lưu trữ.

# Tài liệu tham khảo

\[1\] Acerbi, C., & Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking & Finance*, 26(7), 1487–1503. <https://doi.org/10.1016/S0378-4266(02)00283-2>

\[2\] Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228. <https://doi.org/10.1111/1467-9965.00068>

\[3\] Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk*. Bank for International Settlements. <https://www.bis.org/bcbs/publ/d457.htm>

\[4\] Berkowitz, J. (2001). Testing density forecasts, with applications to risk management. *Journal of Business & Economic Statistics*, 19(4), 465–474. <https://doi.org/10.1198/07350010152596718>

\[5\] Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877. <https://doi.org/10.1080/01621459.2017.1285773>

\[6\] Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327. <https://doi.org/10.1016/0304-4076(86)90063-1>

\[7\] Bollerslev, T. (1987). A conditionally heteroskedastic time series model for speculative prices and rates of return. *The Review of Economics and Statistics*, 69(3), 542–547. <https://doi.org/10.2307/1925546>

\[8\] Bollerslev, T., & Wooldridge, J. M. (1992). Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances. *Econometric Reviews*, 11(2), 143–172. <https://doi.org/10.1080/07474939208800229>

\[9\] Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862. <https://doi.org/10.2307/2527341>

\[10\] Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196. <https://doi.org/10.1093/jjfinec/nbp001>

\[11\] Committee on Payment and Settlement Systems & Technical Committee of the International Organization of Securities Commissions (2012). *Principles for financial market infrastructures*. Bank for International Settlements. <https://www.bis.org/cpmi/publ/d101a.pdf>

\[12\] DataPro. Giải pháp phân tích và theo dõi thị trường đầu tư (nguồn dữ liệu VN30F1M). <https://datapro.vn/>

\[13\] Diebold, F. X., Gunther, T. A., & Tay, A. S. (1998). Evaluating density forecasts with applications to financial risk management. *International Economic Review*, 39(4), 863–883. <https://doi.org/10.2307/2527342>

\[14\] Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. <https://doi.org/10.1080/07350015.1995.10524599>

\[15\] Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007. <https://doi.org/10.2307/1912773>

\[16\] Engle, R. F., & Manganelli, S. (2004). CAViaR: Conditional autoregressive value at risk by regression quantiles. *Journal of Business & Economic Statistics*, 22(4), 367–381. <https://doi.org/10.1198/073500104000000370>

\[17\] Engle, R. F., & Rangel, J. G. (2008). The spline-GARCH model for low-frequency volatility and its global macroeconomic causes. *The Review of Financial Studies*, 21(3), 1187–1222. <https://doi.org/10.1093/rfs/hhn004>

\[18\] Fissler, T., & Ziegel, J. F. (2016). Higher order elicitability and Osband’s principle. *The Annals of Statistics*, 44(4), 1680–1707. <https://doi.org/10.1214/16-AOS1439>

\[19\] Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *The Journal of Business*, 53(1), 67–78. <https://doi.org/10.1086/296072>

\[20\] Giacomini, R., & White, H. (2006). Tests of conditional predictive ability. *Econometrica*, 74(6), 1545–1578. <https://doi.org/10.1111/j.1468-0262.2006.00718.x>

\[21\] Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *The Journal of Finance*, 48(5), 1779–1801. <https://doi.org/10.1111/j.1540-6261.1993.tb05128.x>

\[22\] Gneiting, T. (2011). Making and evaluating point forecasts. *Journal of the American Statistical Association*, 106(494), 746–762. <https://doi.org/10.1198/jasa.2011.r10138>

\[23\] Hansen, B. E. (1994). Autoregressive conditional density estimation. *International Economic Review*, 35(3), 705–730. <https://doi.org/10.2307/2527081>

\[24\] Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365–380. <https://doi.org/10.1198/073500105000000063>

\[25\] Hansen, P. R., Huang, Z., & Shek, H. H. (2012). Realized GARCH: A joint model for returns and realized measures of volatility. *Journal of Applied Econometrics*, 27(6), 877–906. <https://doi.org/10.1002/jae.1234>

\[26\] Hansen, P. R., & Lunde, A. (2005). A realized variance for the whole day based on intermittent high-frequency data. *Journal of Financial Econometrics*, 3(4), 525–554. <https://doi.org/10.1093/jjfinec/nbi028>

\[27\] Hansen, P. R., & Lunde, A. (2006). Consistent ranking of volatility models. *Journal of Econometrics*, 131(1–2), 97–121. <https://doi.org/10.1016/j.jeconom.2005.01.005>

\[28\] Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497. <https://doi.org/10.3982/ECTA5771>

\[29\] Sở Giao dịch Chứng khoán Hà Nội. Quy chế và thông số hợp đồng tương lai chỉ số VN30. <https://www.hnx.vn>

\[30\] Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(47), 1593–1623. <https://www.jmlr.org/papers/v15/hoffman14a.html>

\[31\] Inclán, C., & Tiao, G. C. (1994). Use of cumulative sums of squares for retrospective detection of changes of variance. *Journal of the American Statistical Association*, 89(427), 913–923. <https://doi.org/10.1080/01621459.1994.10476824>

\[32\] Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations*. <https://arxiv.org/abs/1312.6114>

\[33\] Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50. <https://doi.org/10.2307/1913643>

\[34\] Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. *Journal of Machine Learning Research*, 18(14), 1–45. <https://www.jmlr.org/papers/v18/16-107.html>

\[35\] Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *The Journal of Derivatives*, 3(2), 73–84. <https://doi.org/10.3905/jod.1995.407942>

\[36\] Lamoureux, C. G., & Lastrapes, W. D. (1990). Persistence in variance, structural change, and the GARCH model. *Journal of Business & Economic Statistics*, 8(2), 225–234. <https://doi.org/10.1080/07350015.1990.10509794>

\[37\] McNeil, A. J., & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: An extreme value approach. *Journal of Empirical Finance*, 7(3–4), 271–300. <https://doi.org/10.1016/S0927-5398(00)00012-8>

\[38\] Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347–370. <https://doi.org/10.2307/2938260>

\[39\] Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. <https://doi.org/10.2307/1913610>

\[40\] Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *The Journal of Business*, 53(1), 61–65. <https://doi.org/10.1086/296071>

\[41\] Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256. <https://doi.org/10.1016/j.jeconom.2010.03.034>

\[42\] Patton, A. J., Ziegel, J. F., & Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and Value-at-Risk). *Journal of Econometrics*, 211(2), 388–413. <https://doi.org/10.1016/j.jeconom.2018.10.008>

\[43\] Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41. <https://doi.org/10.21314/JOR.2000.038>

\[44\] Rogers, L. C. G., & Satchell, S. E. (1991). Estimating variance from high, low and closing prices. *The Annals of Applied Probability*, 1(4), 504–512. <https://doi.org/10.1214/aoap/1177005835>

\[45\] Shephard, N., & Sheppard, K. (2010). Realising the future: Forecasting with high-frequency-based volatility (HEAVY) models. *Journal of Applied Econometrics*, 25(2), 197–231. <https://doi.org/10.1002/jae.1158>

\[46\] Straumann, D., & Mikosch, T. (2006). Quasi-maximum-likelihood estimation in conditionally heteroscedastic time series: A stochastic recurrence equations approach. *The Annals of Statistics*, 34(5), 2449–2495. <https://doi.org/10.1214/009053606000000803>

\[47\] Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. *arXiv preprint*. <https://arxiv.org/abs/1804.06788>

\[48\] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved $\widehat R$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718. <https://doi.org/10.1214/20-BA1221>

\[49\] Tổng công ty Lưu ký và Bù trừ Chứng khoán Việt Nam. Quy chế bù trừ, thanh toán và ký quỹ chứng khoán phái sinh. <https://www.vsd.vn>

\[50\] White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126. <https://doi.org/10.1111/1468-0262.00152>

\[51\] Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Yes, but did it work?: Evaluating variational inference. *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 5581–5590. <https://proceedings.mlr.press/v80/yao18a.html>

