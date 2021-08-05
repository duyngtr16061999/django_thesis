# django_thesis

# Cài đặt môi trường và tải weights của model

## Môi trường

Cài đặt python version 3.x từ trang [web](https://www.python.org/)
Có 2 cách để cài đặt môi trường:

### Cài đặt từ đầu

Sử dụng pip (pip3) để cài các thư viện cần thiết, chạy tên Command Prompt '''pip install -r requirements.txt'''

### Sử dụng môi trường ảo - python virtual environment đã được thiết lập sẵn

Tải folder myenv chứa môi trường ảo tại [đây](https://drive.google.com/file/d/1q8LUVmM0sOLI9yOJBALEXHeKgPr99I2y/view?usp=sharing), giải nén tại folder chính ta được folder myenv.

Mở Command Prompt chạy '''myenv\Scripts\activate.bat'''

## Tải weights

Vì weights có dung lượng lớn nên sẽ download qua drive với tên file [django_thesis_weights.zip](https://drive.google.com/file/d/1NQXc6DqYL4PzThR5d7C-4_sUPmHB0O9h/view?usp=sharing)

Để file nén vào folder chính và giải nén trực tiếp 3 folder chứa weights: CoLA, MNLI và bert-base-uncased

# Các sử dụng

## Các trang web tham khảo về các sample cho demo

Link dataset [GLUE](https://huggingface.co/datasets/viewer/?dataset=glue), lựa chọn CoLA hoặc MNLI ở mục subset bên trái.

## Các bước sử dụng

Ở ngoài folder chính (django_thesis), chạy '''python manage.py runserver''', đợi một lúc cho app và model được khởi tạo như hình dưới đây.

![runserver](/images/runserver.png)

Mở trình duyệt, nhập vào '''http://127.0.0.1:8000/home/''' để vào trang home, ta được như hình dưới đây.

![home](/images/home.png)

Dùng trang web dataset như hình dưới để lấy ví dụ nhập input vào app. Với:

Task CoLA: ta chọn <strong>Your task *</strong> là CoLA, input câu cần classify vào <strong>Your first sentence *</strong> (Không cần quan tâm second sentence) và nhấn button đỏ <strong>Send Message and Classify</strong>.

Task MNLI: ta chọn <strong>Your task *</strong> là MNLI, input câu cần classify vào <strong>Your first sentence *</strong> và <strong>Your second sentence *</strong>, nhấn button đỏ <strong>Send Message and Classify</strong>.

![data](/images/datasets.png)

Kết quả sẽ hiện thị ở trang model:

![model](/images/model.png)
