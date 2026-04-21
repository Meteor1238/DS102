# Pneumonia Detection using Support Vector Machine

## Cấu trúc thư mục

* `data_prep.py`: Chứa hàm đọc ảnh, chuyển sang ảnh xám, thay đổi kích thước (128x128), chuẩn hóa và làm phẳng.
* `svm.py`: Xây dựng lớp `SVM` với hàm mục tiêu là Hinge Loss và quá trình cập nhật gradient thủ công.
* `evaluate.py`: Chứa các hàm tính toán các độ đo (Precision, Recall, F1-Score) và vẽ đồ thị Loss.
* `main.py`: File thực thi chính, nạp dữ liệu, huấn luyện hai mô hình và so sánh kết quả.

## Yêu cầu môi trường

Cần cài đặt các thư viện sau:
* numpy
* opencv-python (cv2)
* matplotlib
* scikit-learn
* tqdm

## Cách sử dụng

1. Đặt dữ liệu vào thư mục `data/train` và `data/test` (với các thư mục con NORMAL và PNEUMONIA).
2. Chạy file main: python main.py