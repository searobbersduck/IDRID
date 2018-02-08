import os
from glob import glob
from PIL import Image
import cv2
import numpy as np

folder_list = ['EX', 'HE', 'MA', 'SE']
root = '../../data/IDRID/IDRID 1'
ex_root = os.path.join(root, 'EX')
he_root = os.path.join(root, 'HE')
ma_root = os.path.join(root, 'MA')
se_root = os.path.join(root, 'SE')
raw_root = os.path.join(root, 'Apparent Retinopathy')
mix_root = os.path.join(root, 'mix_s')
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
    bg_array = np.zeros(raw_img.shape[0:2], dtype=np.uint8)

    for index1, file in enumerate(file_list):
        if not os.path.exists(file):
            continue
        img = cv2.imread(file)
        '''
        ex: 1
        he: 2
        ma: 3
        se: 4
        '''
        # for i in range(raw_img.shape[0]):
        #     for j in range(raw_img.shape[1]):
        #         if img[i,j,2] == 255:
        #             bg_array[i,j] = img[i,j,2]
        bg_array[img[:,:,2]==255] = index1+1

    cv2.imshow('test', bg_array)
    # cv2.waitKey(2000)
    cv2.imwrite(os.path.join(mix_root, '{}_mix.jpg'.format(index)), bg_array)
    # print(img[:,:,2].max())
    print(bg_array.max())