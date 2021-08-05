# django_thesis

# Cài đặt môi trường và tải weights của model

## Môi trường

Cài đặt python version 3.x từ trang [web](https://www.python.org/)
Có 2 cách để cài đặt môi trường:

### Cài đặt từ đầu

Sử dụng pip (pip3) để cài các thư viện cần thiết '''pip install -r requirements.txt'''

### Sử dụng môi trường ảo - python virtual environment đã được thiết lập sẵn

myenv\Scripts\activate.bat

## Tải weights

Vì weights có dung lượng lớn nên sẽ download qua drive với tên file [django_thesis_weights.zip](https://drive.google.com/file/d/1NQXc6DqYL4PzThR5d7C-4_sUPmHB0O9h/view?usp=sharing)

Để file nén vào folder chính và giải nén trực tiếp 3 folder chứa weights: CoLA, MNLI và bert-base-uncased

# Các sử dụng

## Các trang web tham khảo về các sample cho demo

Link dataset [GLUE](https://huggingface.co/datasets/viewer/?dataset=glue), lựa chọn CoLA hoặc MNLI ở mục subset bên trái.

## Các bước sử dụng

Ở ngoài folder chính (django_thesis), chạy '''python manage.py runserver''', đợi một lúc cho app và model được khởi tạo.


