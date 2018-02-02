# 检查同一张图片的不同的分割标签（软渗出/硬渗出/微血管流/出血斑）是否有重合区域

import cv2
import os
from glob import glob
import numpy as np

root = '../data/IDRID/IDRID 1/'
ex_root = os.path.join(root, 'EX')
he_root = os.path.join(root, 'HE')
ma_root = os.path.join(root, 'MA')
se_root = os.path.join(root, 'SE')
raw_root = os.path.join(root, 'Apparent Retinopathy')

raw_list = glob(os.path.join(raw_root, '*.jpg'))
raw_list = [os.path.basename(i).split('.')[0] for i in raw_list]

for index in raw_list:
    ex_file = os.path.join(ex_root, '{}_EX.tif'.format(index))
    he_file = os.path.join(ex_root, '{}_HE.tif'.format(index))
    ma_file = os.path.join(ex_root, '{}_MA.tif'.format(index))
    se_file = os.path.join(ex_root, '{}_SE.tif'.format(index))

    file_list = []
    file_list.append(ex_file)
    file_list.append(he_file)
    file_list.append(ma_file)
    file_list.append(se_file)

    raw_img = cv2.imread(os.path.join(raw_root, index+'.jpg'))

    cnt_array = np.zeros(raw_img.shape[0:2], dtype=np.uint8)
    bg_array = np.zeros(raw_img.shape[0:2], dtype=np.uint16)

    for file in file_list:
        if not os.path.exists(file):
            continue
        img = cv2.imread(file)
        bg_array += img[:,:,0]

    print(bg_array.max())
