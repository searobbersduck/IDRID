# 检查同一张图片的不同的分割标签（软渗出/硬渗出/微血管流/出血斑）是否有重合区域
# this file show the mask overlap.
# 生成混合标签，分别按如下关系对应：ex: b（蓝）， he: g（绿）， ma: r（红）， se: y（黄）

import cv2
import os
from glob import glob
import numpy as np
from PIL import Image

root = '../../data/IDRID/IDRID 1/'
ex_root = os.path.join(root, 'EX')
he_root = os.path.join(root, 'HE')
ma_root = os.path.join(root, 'MA')
se_root = os.path.join(root, 'SE')
raw_root = os.path.join(root, 'Apparent Retinopathy')
mix_root = os.path.join(root, 'mix')
os.makedirs(mix_root, exist_ok=True)

raw_list = glob(os.path.join(raw_root, '*.jpg'))
raw_list = [os.path.basename(i).split('.')[0] for i in raw_list]

for index in raw_list:
    ex_file = os.path.join(ex_root, '{}_EX.tif'.format(index))
    he_file = os.path.join(he_root, '{}_HE.tif'.format(index))
    ma_file = os.path.join(ma_root, '{}_MA.tif'.format(index))
    se_file = os.path.join(se_root, '{}_SE.tif'.format(index))

    file_list = []
    file_list.append(ex_file)
    file_list.append(he_file)
    file_list.append(ma_file)
    file_list.append(se_file)

    raw_img = cv2.imread(os.path.join(raw_root, index+'.jpg'))

    cnt_array = np.ones(raw_img.shape, dtype=np.uint8)
    bg_array = np.zeros(raw_img.shape, dtype=np.uint8)

    for index1, file in enumerate(file_list):
        if not os.path.exists(file):
            continue
        img = cv2.imread(file)
        '''
        ex: b
        he: g
        ma: r
        se: y
        '''
        if index1 < 3:
            bg_array[:,:,index1] = img[:,:,2]
        else:
            bg_array[:, :, 1] += img[:,:,2]
            bg_array[:, :, 2] += img[:, :, 2]

    # cv2.imshow('test', bg_array)
    # # cv2.waitKey(2000)
    # cv2.imwrite(os.path.join(mix_root, '{}_mix.jpg'.format(index)), bg_array)
    # # print(img[:,:,2].max())
    # print(bg_array.max())

mix_root_t = os.path.join(root, 'mix_thumbnail')
os.makedirs(mix_root_t, exist_ok=True)
mix_list = glob(os.path.join(mix_root, '*.jpg'))
for img_file in mix_list:
    img = Image.open(img_file)
    img.thumbnail((512,512), Image.ANTIALIAS)
    img.save(os.path.join(mix_root_t, os.path.basename(img_file).split('.')[0]+'.png'))