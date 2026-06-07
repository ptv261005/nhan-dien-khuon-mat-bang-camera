# -*- coding: utf-8 -*-
"""
File cai thu vien rieng cho project Nhan dien khuon mat bang camera.
Ban da sua loi: KHONG cai scikit-learn de tranh loi sparsefuncs_fast bi Windows chan.

Sau khi cai xong: restart kernel Spyder roi chay file nhan_dien_khuon_mat_1_file_spyder.py
"""

import sys
import subprocess

packages = [
    "opencv-contrib-python",
    "numpy",
]

print("Python dang dung:", sys.executable)
print("Bat dau cai dat thu vien...\n")

for pkg in packages:
    print("Dang cai:", pkg)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

print("\nDa cai xong. Hay restart kernel Spyder roi chay file nhan_dien_khuon_mat_1_file_spyder.py")
