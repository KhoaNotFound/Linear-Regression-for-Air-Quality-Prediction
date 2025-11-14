# Phân tích Chất lượng Không khí bằng Hồi quy Tuyến tính & Gradient Descent

Dự án này là một bài tập lớn nhằm triển khai mô hình Hồi quy Tuyến tính Đơn biến từ đầu (from scratch) bằng thuật toán **Gradient Descent** để phân tích mối quan hệ giữa Tổng Bụi lơ lửng (TSP) và Bụi mịn (PM2.5) tại TP. Hồ Chí Minh.

Kết quả của mô hình thủ công được so sánh trực tiếp với thư viện `scikit-learn` để kiểm chứng tính chính xác.

## 🎯 Mục tiêu Dự án

1.  **Hiểu rõ Hồi quy Tuyến tính:** Áp dụng phương trình $y = wx + b$ vào một bài toán thực tế.
2.  **Triển khai Gradient Descent:** Tự tay viết thuật toán tối ưu hóa (Gradient Descent) để tìm ra các tham số `w` (trọng số) và `b` (hệ số chặn) nhằm tối thiểu hóa Hàm Mất mát (MSE).
3.  **Xử lý Dữ liệu:** Thực hành các kỹ thuật làm sạch dữ liệu (`dropna`, lọc lỗi) và chuẩn hóa dữ liệu (Standardization - Z-score).
4.  **So sánh & Đối chiếu:** Kiểm chứng kết quả của thuật toán thủ công với thư viện `sklearn.linear_model.LinearRegression`.

---

## 🔬 Các Khái niệm Toán học & Thống kê được sử dụng

Dự án này vận dụng các kiến thức nền tảng về Toán học và Thống kê:

### 1. Thống kê
* **Hàm Mất mát (Loss Function) - MSE:**
    $$J(w,b) = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2$$
* **Chuẩn hóa Z-Score (Standardization):**
    $$z = \frac{(x - \mu)}{\sigma}$$
    * Sử dụng để co giãn (scale) dữ liệu `X_Train` (TSP) nhằm tránh lỗi **Bùng nổ Gradient (Gradient Explosion)**.
* **Unscaling (Giải co giãn):**
    * Sử dụng phép chứng minh toán học để biến đổi `w_scaled` và `b_scaled` về thang đo gốc để so sánh.
    * $w_{\text{gốc}} = w_{\text{scaled}} / \sigma$
    * $b_{\text{gốc}} = b_{\text{scaled}} - (w_{\text{scaled}} \cdot \mu) / \sigma$

### 2. Giải tích
* **Đạo hàm riêng (Partial Derivatives):**
    * $\frac{\partial J}{\partial w} = \frac{2}{n} \sum x_i (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})$
    * $\frac{\partial J}{\partial b} = \frac{2}{n} \sum (y_{\text{pRed}}^{(i)} - y_{\text{true}}^{(i)})$
* **Thuật toán Gradient Descent:**
    * $w := w - \eta \cdot \frac{\partial J}{\partial w}$
    * $b := b - \eta \cdot \frac{\partial J}{\partial b}$

---

## dataset Dữ liệu (Dataset)

* **Tên:** Air Quality Ho Chi Minh City
* **Nguồn:** (https://data.mendeley.com/datasets/pk6tzrjks8/1?fbclid=IwY2xjawOEQ5ZleHRuA2FlbQIxMABicmlkETE5OGx4Q1FmbGZrWkozM3RWc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHvEkZXfUlvjYFc6QTs9smXjSBPa3H5UCQLTJvno-stcOf7qcihRDUo9ZQczC_aem_Z2mfiklCv4h65gSiW_2Ofg)
* **Feature (X) sử dụng:** `TSP` (Total Suspended Particulates)
* **Target (y) sử dụng:** `PM2.5`

---

## ⚙️ Hướng dẫn Chạy (How to Run)

1.  Đảm bảo bạn đã cài đặt các thư viện cần thiết.
2.  Tải tệp dữ liệu `Air-Quality-Ho-Chi-Minh-City.xlsx - Air Quality Ho Chi Minh City.csv` và đặt vào đúng đường dẫn trong file script.
3.  Chạy tệp script Python.

```bash
python your_script_name.py
