'''
显示标注数据的中心位置
'''

import os
import argparse
import pandas as pd
from glob import glob
import cv2

root = '/home/weidong/code/kaggle/IDRID/data/IDRID/IDRID 3/Training a and b'
od_csv = '/home/weidong/code/kaggle/IDRID/data/IDRID/IDRID 3/IDRiD_OD_Center_Training_set.csv'
fovea_csv = '/home/weidong/code/kaggle/IDRID/data/IDRID/IDRID 3/IDRiD_Fovea_Center_Training_set.csv'
root_out = '/home/weidong/code/kaggle/IDRID/data/IDRID/IDRID 3/Show a and b'
os.makedirs(root_out, exist_ok=True)

df_od = pd.DataFrame.from_csv(od_csv)
df_fovea = pd.DataFrame.from_csv(fovea_csv)

image_list = glob(os.path.join(root, '*.jpg'))
for image_file in image_list:
    image_basename = os.path.basename(image_file).split('.')[0]
    # image_basename = 'IDRiD_0{}'.format(image_basename.split('.')[0].split('_')[1])
    od_x = int(df_od.loc[image_basename][0])
    od_y = int(df_od.loc[image_basename][1])
    fovea_x = int(df_fovea.loc[image_basename][0])
    fovea_y = int(df_fovea.loc[image_basename][1])
    image = cv2.imread(image_file)
    cv2.circle(image, (od_x, od_y), 10, (255, 255, 0), thickness=10)
    cv2.circle(image, (fovea_x, fovea_y), 10, (0, 255, 255), thickness=10)
    cv2.imshow('test', image)
    # cv2.waitKey(2000)
    out_file = os.path.join(root_out, os.path.basename(image_file))
    cv2.imwrite(out_file, image)
    print('output file: {}'.format(out_file))



