# Hướng dẫn code project nhận diện khuôn mặt bằng camera

Tên file code chính:

```text
nhan dien khuon mat bang camera.py
```

Tên file cài thư viện:

```text
tai thu vien.py
```

File này chỉ hướng dẫn phần code, không hướng dẫn báo cáo và không hướng dẫn GitHub.

---

## 1. Mục đích của code

Code dùng để xây dựng chương trình nhận diện khuôn mặt bằng camera laptop.

Chương trình có thể:

- Chụp ảnh khuôn mặt của từng người.
- Lưu ảnh vào dataset.
- Huấn luyện mô hình nhận diện khuôn mặt.
- Nhận diện khuôn mặt bằng webcam theo thời gian thực.
- Test nhận diện bằng một ảnh có sẵn.
- Hiển thị `Unknown` nếu gặp người lạ.

---

## 2. Các thư viện trong code

Trong code có các thư viện chính:

```python
import cv2
import numpy as np
import json
import os
from pathlib import Path
from collections import Counter, deque
```

Ý nghĩa:

- `cv2`: dùng để mở camera, đọc ảnh, xử lý ảnh, phát hiện khuôn mặt và huấn luyện mô hình LBPH.
- `numpy`: dùng để xử lý ảnh dưới dạng mảng số.
- `json`: dùng để lưu và đọc file `label_map.json`. File này giúp biết nhãn số nào tương ứng với tên người nào.
- `os`: dùng để thao tác với hệ điều hành, ví dụ mở thư mục sau khi chụp xong.
- `Path`: dùng để tạo và quản lý đường dẫn thư mục.
- `Counter`, `deque`: dùng để làm mượt kết quả nhận diện, tránh tên bị nhảy liên tục khi camera đang chạy.

---

## 3. Các thư mục được code tạo ra

Khi chạy code, chương trình sẽ tạo các thư mục:

```text
dataset_faces
models
outputs
```

Ý nghĩa:

- `dataset_faces`: lưu ảnh khuôn mặt đã chụp.
- `models`: lưu mô hình sau khi huấn luyện.
- `outputs`: lưu kết quả đánh giá mô hình.

Ví dụ cấu trúc thư mục:

```text
dataset_faces/
├── NguyenVanA/
│   ├── NguyenVanA_0001.jpg
│   ├── NguyenVanA_0002.jpg
│   └── NguyenVanA_0003.jpg
│
└── TranThiB/
    ├── TranThiB_0001.jpg
    ├── TranThiB_0002.jpg
    └── TranThiB_0003.jpg
```

Ví dụ thư mục model:

```text
models/
├── face_lbph_model.yml
└── label_map.json
```

Ví dụ thư mục kết quả:

```text
outputs/
├── evaluation_report.txt
└── confusion_matrix.png
```

---

## 4. Các tham số quan trọng trong code

Trong code có một số tham số nên hiểu rõ.

```python
FACE_SIZE = (160, 160)
```

Ý nghĩa:

Mỗi khuôn mặt sau khi được phát hiện sẽ được cắt ra và resize về kích thước `160x160`. Việc này giúp tất cả ảnh đầu vào có cùng kích thước trước khi đưa vào mô hình.

Ví dụ:

- Ảnh mặt của `NguyenVanA` ban đầu có thể là `300x300`.
- Ảnh mặt của `TranThiB` ban đầu có thể là `250x280`.
- Sau khi xử lý, cả hai đều được đưa về `160x160`.

```python
UNKNOWN_THRESHOLD = 60.0
```

Ý nghĩa:

Đây là ngưỡng để quyết định khuôn mặt là người quen hay người lạ.

Với mô hình LBPH:

- Số `confidence` càng nhỏ thì khuôn mặt càng giống dữ liệu đã học.
- Số `confidence` càng lớn thì khuôn mặt càng khác dữ liệu đã học.

Ví dụ:

```text
Model dự đoán: NguyenVanA
Confidence: 42.5
Ngưỡng: 60.0
Kết quả: nhận là NguyenVanA
```

Ví dụ khác:

```text
Model dự đoán: NguyenVanA
Confidence: 85.0
Ngưỡng: 60.0
Kết quả: không đủ chắc chắn, hiện Unknown
```

Nếu người lạ hay bị nhận nhầm thành `NguyenVanA` hoặc `TranThiB`, có thể giảm ngưỡng:

```python
UNKNOWN_THRESHOLD = 55.0
```

hoặc:

```python
UNKNOWN_THRESHOLD = 50.0
```

---

## 5. Hàm khởi tạo thư mục

Hàm thường có tên:

```python
init_folders()
```

Chức năng:

Hàm này tạo các thư mục cần thiết nếu chưa tồn tại.

Các thư mục được tạo:

- `dataset_faces`
- `models`
- `outputs`

Ví dụ:

Nếu mới tải code về và chưa có thư mục `dataset_faces`, khi chạy chương trình hàm này sẽ tự tạo.

---

## 6. Hàm mở camera

Hàm thường có tên:

```python
open_camera()
```

Chức năng:

Hàm này dùng để mở webcam.

Trong code thường có danh sách camera:

```python
CAMERA_IDS = [0, 1, 2]
```

Ý nghĩa:

- `0` thường là webcam mặc định của laptop.
- `1` hoặc `2` có thể là camera rời hoặc camera ảo.

Nếu camera không mở được, có thể sửa thành:

```python
CAMERA_IDS = [1, 0, 2]
```

Ví dụ:

Nếu laptop đang nhận webcam ở ID `1`, chương trình sẽ mở camera `1` trước.

---

## 7. Hàm phát hiện khuôn mặt

Hàm thường có tên:

```python
get_face_detector()
detect_faces(gray, detector)
```

Chức năng:

Dùng bộ phát hiện khuôn mặt Haar Cascade của OpenCV để tìm mặt trong ảnh.

Quy trình:

1. Camera lấy ảnh màu.
2. Ảnh màu được chuyển sang ảnh xám.
3. Bộ phát hiện khuôn mặt tìm vị trí mặt.
4. Kết quả trả về là tọa độ khuôn mặt.

Tọa độ khuôn mặt có dạng:

```text
x, y, w, h
```

Trong đó:

- `x, y`: vị trí góc trên bên trái của khuôn mặt.
- `w`: chiều rộng khuôn mặt.
- `h`: chiều cao khuôn mặt.

Ví dụ:

```text
x = 120
y = 80
w = 160
h = 160
```

Nghĩa là khuôn mặt nằm tại vùng ảnh bắt đầu từ điểm `(120, 80)`, rộng `160` pixel và cao `160` pixel.

---

## 8. Hàm tiền xử lý khuôn mặt

Hàm thường có tên:

```python
preprocess_face(gray, box)
```

Chức năng:

Hàm này xử lý khuôn mặt trước khi lưu hoặc đưa vào mô hình.

Quy trình:

1. Nhận ảnh xám.
2. Nhận tọa độ khuôn mặt.
3. Cắt riêng vùng khuôn mặt.
4. Resize khuôn mặt về kích thước `FACE_SIZE`.
5. Trả về ảnh khuôn mặt đã xử lý.

Ví dụ:

Camera đang nhìn thấy `NguyenVanA`. Code phát hiện khuôn mặt của `NguyenVanA`, cắt riêng phần mặt và đưa về kích thước `160x160`. Ảnh này sẽ được lưu vào dataset hoặc đưa vào model để dự đoán.

---

## 9. Hàm thu thập dữ liệu

Hàm thường có tên:

```python
collect_faces()
```

Chức năng:

Hàm này dùng để chụp ảnh khuôn mặt bằng webcam và lưu vào dataset.

Cách hoạt động:

1. Người dùng chọn mục `1` trong menu.
2. Nhập tên người cần chụp.
3. Camera mở lên.
4. Khi thấy mặt, chương trình vẽ khung quanh mặt.
5. Người dùng bấm phím `S` để bắt đầu lưu.
6. Code lưu từng ảnh khuôn mặt vào thư mục tương ứng.

Ví dụ:

Người dùng nhập tên:

```text
NguyenVanA
```

Ảnh sẽ được lưu vào:

```text
dataset_faces/NguyenVanA/
```

Tên ảnh có dạng:

```text
NguyenVanA_0001.jpg
NguyenVanA_0002.jpg
NguyenVanA_0003.jpg
```

Nếu chụp người thứ hai là `TranThiB`, nhập:

```text
TranThiB
```

Ảnh sẽ được lưu vào:

```text
dataset_faces/TranThiB/
```

Tên ảnh có dạng:

```text
TranThiB_0001.jpg
TranThiB_0002.jpg
TranThiB_0003.jpg
```

Lưu ý:

Nên nhập tên không dấu và không có khoảng trắng.

Nên dùng:

```text
NguyenVanA
```

Không nên dùng:

```text
Nguyễn Văn A
```

Lý do là tên không dấu giúp tránh lỗi đường dẫn và lỗi khi lưu file.

Mỗi người nên chụp khoảng `200` đến `500` ảnh.

Nên chụp nhiều kiểu:

- Mặt nhìn thẳng.
- Mặt hơi nghiêng trái.
- Mặt hơi nghiêng phải.
- Gần camera.
- Xa camera.
- Ánh sáng mạnh.
- Ánh sáng yếu.

---

## 10. Hàm đọc dataset

Hàm thường có tên:

```python
load_dataset()
```

Chức năng:

Hàm này đọc toàn bộ ảnh đã lưu trong thư mục `dataset_faces`.

Ví dụ dataset:

```text
dataset_faces/
├── NguyenVanA/
└── TranThiB/
```

Code sẽ hiểu:

- Thư mục `NguyenVanA` là một người.
- Thư mục `TranThiB` là một người.

Sau đó code gán nhãn số cho từng người.

Ví dụ:

```text
0 -> NguyenVanA
1 -> TranThiB
```

Bảng ánh xạ này được lưu vào file:

```text
models/label_map.json
```

Ví dụ nội dung `label_map.json`:

```json
{
    "0": "NguyenVanA",
    "1": "TranThiB"
}
```

Nhờ file này, khi model dự đoán ra nhãn `0` thì chương trình biết đó là `NguyenVanA`.

---

## 11. Hàm huấn luyện mô hình

Hàm thường có tên:

```python
train_model()
```

Chức năng:

Hàm này dùng để huấn luyện mô hình nhận diện khuôn mặt.

Quy trình:

1. Đọc ảnh trong `dataset_faces`.
2. Đọc nhãn tương ứng với từng người.
3. Chia dữ liệu thành tập train và test.
4. Tạo mô hình LBPH.
5. Huấn luyện mô hình.
6. Lưu mô hình.
7. Đánh giá mô hình.

Mô hình được tạo bằng lệnh:

```python
cv2.face.LBPHFaceRecognizer_create()
```

Sau khi huấn luyện xong, model được lưu vào:

```text
models/face_lbph_model.yml
```

Ví dụ:

Nếu dataset có `NguyenVanA` và `TranThiB`, mô hình sẽ học đặc điểm khuôn mặt của hai người này. Khi gặp ảnh mới, model sẽ so sánh và dự đoán ảnh đó giống `NguyenVanA` hay `TranThiB` hơn.

---

## 12. Mô hình LBPH là gì?

LBPH là viết tắt của `Local Binary Patterns Histograms`.

Hiểu đơn giản:

LBPH phân tích sự thay đổi sáng tối trên khuôn mặt.

Ví dụ:

Khuôn mặt của `NguyenVanA` có đặc điểm vùng mắt, mũi, miệng, ánh sáng và kết cấu khác với `TranThiB`. LBPH biến các đặc điểm đó thành dạng số. Khi có ảnh mới, model so sánh ảnh mới với dữ liệu đã học.

Ưu điểm:

- Dễ dùng.
- Chạy nhanh.
- Không cần GPU.
- Phù hợp với Spyder và laptop bình thường.
- Phù hợp với project môn học máy cơ bản.

Nhược điểm:

- Dễ bị ảnh hưởng bởi ánh sáng.
- Dễ bị ảnh hưởng bởi góc mặt.
- Nếu người lạ giống người trong dataset, có thể bị nhận nhầm.
- Vì vậy cần dùng thêm ngưỡng `Unknown`.

---

## 13. Hàm đánh giá mô hình

Trong hàm `train_model()` có phần đánh giá mô hình.

Chức năng:

- Dùng tập test để kiểm tra model.
- So sánh kết quả dự đoán với nhãn thật.
- Tính độ chính xác.
- Tạo ma trận nhầm lẫn.

Ví dụ:

```text
Tập test có 100 ảnh
Model nhận đúng 94 ảnh
Model nhận sai 6 ảnh
Độ chính xác = 94 / 100 = 94%
```

Kết quả đánh giá được lưu trong:

```text
outputs/evaluation_report.txt
```

Ma trận nhầm lẫn được lưu trong:

```text
outputs/confusion_matrix.png
```

Ma trận nhầm lẫn giúp biết model hay nhầm người nào với người nào.

Ví dụ:

Nếu ảnh của `NguyenVanA` bị nhầm thành `TranThiB` nhiều, cần chụp thêm dữ liệu cho `NguyenVanA` ở nhiều góc khác nhau.

---

## 14. Hàm load model

Hàm thường có tên:

```python
load_model_and_labels()
```

Chức năng:

Hàm này dùng để load lại model đã huấn luyện.

Code sẽ đọc:

```text
models/face_lbph_model.yml
models/label_map.json
```

Sau đó trả về:

- Model nhận diện khuôn mặt.
- Bảng tên tương ứng với nhãn số.

Ví dụ:

Model dự đoán nhãn `0`. File `label_map.json` cho biết `0` là `NguyenVanA`, nên chương trình sẽ hiển thị `NguyenVanA` lên màn hình.

---

## 15. Hàm demo nhận diện bằng camera

Hàm thường có tên:

```python
demo_camera()
```

Chức năng:

Dùng webcam để nhận diện khuôn mặt theo thời gian thực.

Quy trình:

1. Load model đã train.
2. Mở camera.
3. Đọc từng khung hình từ camera.
4. Phát hiện khuôn mặt.
5. Tiền xử lý khuôn mặt.
6. Model dự đoán khuôn mặt.
7. So sánh `confidence` với `UNKNOWN_THRESHOLD`.
8. Hiển thị tên hoặc `Unknown`.

Ví dụ:

Camera thấy mặt `NguyenVanA`.

```text
Label: 0
Confidence: 42.0
Label map: 0 -> NguyenVanA
Ngưỡng: 60.0
Kết quả: NguyenVanA
```

Ví dụ khác:

Camera thấy người lạ.

```text
Label gần nhất: 1
Confidence: 90.0
Label map: 1 -> TranThiB
Ngưỡng: 60.0
Kết quả: Unknown
```

---

## 16. Cách bỏ số confidence sau tên

Nếu khi demo chương trình hiện:

```text
NguyenVanA (42.0)
```

và muốn chỉ hiện:

```text
NguyenVanA
```

Thì tìm dòng:

```python
text = f"{name} ({confidence:.1f})"
```

Sửa thành:

```python
text = name
```

Nếu có dòng:

```python
text = f"Unknown ({confidence:.1f})"
```

Sửa thành:

```python
text = "Unknown"
```

Việc này chỉ thay đổi phần hiển thị, không ảnh hưởng đến mô hình.

---

## 17. Lọc kết quả nhiều frame liên tiếp

Trong code có thể dùng `deque` và `Counter` để làm kết quả ổn định hơn.

Mục đích:

Không để chương trình chỉ nhìn một frame rồi kết luận ngay.

Ví dụ camera nhận được 5 frame:

```text
Frame 1: NguyenVanA
Frame 2: NguyenVanA
Frame 3: Unknown
Frame 4: NguyenVanA
Frame 5: NguyenVanA
```

Kết quả xuất hiện nhiều nhất là `NguyenVanA`, vì vậy chương trình hiển thị `NguyenVanA`.

Cách này giúp giảm lỗi nhảy tên liên tục.

---

## 18. Hàm test bằng ảnh

Hàm thường có tên:

```python
predict_image()
```

Chức năng:

Dùng để nhận diện khuôn mặt từ một ảnh có sẵn.

Quy trình:

1. Người dùng nhập đường dẫn ảnh.
2. Code đọc ảnh bằng OpenCV.
3. Phát hiện khuôn mặt trong ảnh.
4. Tiền xử lý khuôn mặt.
5. Dùng model để dự đoán.
6. Hiển thị ảnh kết quả.

Ví dụ nhập đường dẫn:

```text
C:\Users\ADMIN\Pictures\anh_test.jpg
```

Nếu ảnh đó là `NguyenVanA` và model nhận đúng, màn hình sẽ hiện:

```text
NguyenVanA
```

Nếu ảnh đó là người lạ, màn hình sẽ hiện:

```text
Unknown
```

---

## 19. Hàm hiển thị ảnh vừa màn hình

Hàm thường có tên:

```python
hien_thi_anh_vua_man_hinh(window_name, image)
```

Chức năng:

Khi test ảnh, nếu ảnh quá lớn thì cửa sổ OpenCV có thể bị phóng to. Hàm này sẽ resize ảnh khi hiển thị để ảnh vừa màn hình.

Lưu ý:

Hàm này chỉ resize ảnh để hiển thị. Ảnh gốc và kết quả nhận diện không bị ảnh hưởng.

---

## 20. Hàm xem thông tin dataset

Hàm thường có tên:

```python
show_dataset_info()
```

Chức năng:

In ra thông tin dataset hiện tại.

Thông tin có thể gồm:

- Đường dẫn project.
- Đường dẫn `dataset_faces`.
- Danh sách người đã chụp.
- Số ảnh của từng người.
- Đường dẫn model.
- Đường dẫn outputs.

Ví dụ kết quả:

```text
DATASET_DIR: C:\Users\ADMIN\Desktop\nhan dien khuon mat bang camera\dataset_faces

NguyenVanA: 250 ảnh
TranThiB: 230 ảnh
```

Dùng hàm này để kiểm tra dữ liệu đã lưu đúng chưa.

---

## 21. Hàm menu chính

Hàm thường có tên:

```python
main()
```

Chức năng:

Hiển thị menu và gọi chức năng tương ứng theo lựa chọn của người dùng.

Menu:

```text
1. Thu thập dữ liệu khuôn mặt bằng camera
2. Huấn luyện model + đánh giá
3. Demo nhận diện bằng camera
4. Test model với 1 ảnh
5. Xem thông tin dataset
0. Thoát
```

Ví dụ:

- Người dùng nhập `1`: chương trình gọi `collect_faces()`.
- Người dùng nhập `2`: chương trình gọi `train_model()`.
- Người dùng nhập `3`: chương trình gọi `demo_camera()`.
- Người dùng nhập `4`: chương trình gọi `predict_image()`.
- Người dùng nhập `5`: chương trình gọi `show_dataset_info()`.

---

## 22. Luồng chạy của toàn bộ code

Luồng chạy tổng thể:

1. Chạy file `nhan dien khuon mat bang camera.py`.
2. Chọn mục `1` để chụp ảnh khuôn mặt.
3. Ảnh được lưu vào `dataset_faces`.
4. Chọn mục `2` để huấn luyện mô hình.
5. Model được lưu vào `models`.
6. Chọn mục `3` để demo nhận diện bằng camera.
7. Camera phát hiện khuôn mặt và hiển thị tên.

Ví dụ:

```text
Chụp NguyenVanA được 250 ảnh
Chụp TranThiB được 250 ảnh
Train model
Demo camera
Kết quả hiển thị: NguyenVanA / TranThiB / Unknown
```

---

## 23. Thứ tự chạy đúng

Thứ tự chạy đúng là:

1. Chạy `tai thu vien.py` để cài thư viện.
2. Restart kernel Spyder.
3. Chạy `nhan dien khuon mat bang camera.py`.
4. Chọn mục `1` để chụp dữ liệu cho `NguyenVanA`.
5. Chọn mục `1` tiếp để chụp dữ liệu cho `TranThiB`.
6. Chọn mục `2` để train model.
7. Chọn mục `3` để demo bằng camera.
8. Chọn mục `4` nếu muốn test bằng ảnh.
9. Chọn mục `5` nếu muốn kiểm tra dataset.

---

## 24. Những lỗi code thường gặp

### Lỗi 1: Không mở được camera

Cách sửa:

- Đóng ứng dụng đang dùng camera.
- Sửa `CAMERA_IDS = [1, 0, 2]`.

### Lỗi 2: Không có thư viện cv2

Cách sửa:

- Chạy `tai thu vien.py`.
- Restart kernel Spyder.

### Lỗi 3: Không có cv2.face

Cách sửa:

- Cài `opencv-contrib-python`.
- Không chỉ cài `opencv-python`.

### Lỗi 4: Train không được

Nguyên nhân:

- Chưa có ảnh trong `dataset_faces`.
- Chỉ có dữ liệu của một người.
- Ảnh không phát hiện được mặt.

Cách sửa:

- Chụp ít nhất hai người.
- Mỗi người nên có `200` đến `500` ảnh.
- Kiểm tra bằng mục `5`.

### Lỗi 5: Người lạ bị nhận nhầm

Ví dụ người lạ bị nhận nhầm thành `NguyenVanA` hoặc `TranThiB`.

Cách sửa:

- Giảm `UNKNOWN_THRESHOLD`.
- Tăng dữ liệu cho từng người.
- Chụp nhiều góc mặt.
- Chụp trong nhiều điều kiện ánh sáng.

---

## 25. Kết luận về code

Code được viết theo dạng một file chính để dễ chạy trên Spyder.

Các phần quan trọng nhất của code là:

- Thu thập dữ liệu khuôn mặt.
- Phát hiện khuôn mặt.
- Tiền xử lý ảnh mặt.
- Huấn luyện mô hình LBPH.
- Lưu model.
- Load model.
- Nhận diện realtime bằng camera.
- Dùng ngưỡng `Unknown` để hạn chế nhận nhầm người lạ.

Ví dụ với hai người `NguyenVanA` và `TranThiB`:

- Code chụp ảnh hai người.
- Model học đặc điểm khuôn mặt của hai người.
- Khi camera thấy `NguyenVanA`, chương trình hiện `NguyenVanA`.
- Khi camera thấy `TranThiB`, chương trình hiện `TranThiB`.
- Khi camera thấy người lạ, chương trình hiện `Unknown`.
