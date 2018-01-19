import numpy as np
import io
import requests
from PIL import Image

def read_data(img_num, txt, idx):
    f = open(txt, 'r')
    IMG_URLs = []
    for i in range(img_num):
        line = f.readline()
        url = line.split()[idx]
        IMG_URLs = np.append(IMG_URLs,url)
    f.close()
    return IMG_URLs

def get_img(num, IMG_URL, root):
    response = requests.get(IMG_URL)
    img_pil = Image.open(io.BytesIO(response.content))
    #img_pil.save(str(idx) + '.jpg')
    img_pil.save(root + str(num) + '.jpg')
    return img_pil

"""
# 이미지넷 이미지 추출
IMG_URLs = read_data(1071, 'data/ear.txt', 0)

for i in range(len(IMG_URLs)):
    try:
        print(i)
        IMG_URL = IMG_URLs[i]
        img = get_img(i, IMG_URL, root='image/train/ear/')
        if img == 0: continue
        print()
    except:
        print('에러')
        print()
"""