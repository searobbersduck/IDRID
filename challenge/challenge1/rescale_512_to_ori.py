'''
1. input ori image size, create
2. params
3. rescale 512 to params size
4. params size to ori size
'''

import cv2
from PIL import Image
import numpy as np
import pandas as pd
from glob import glob

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='generate segmentation metrics information')
    parser.add_argument('--cat', choices=['EX', 'HE', 'MA', 'SE'], default='EX')
    return parser.parse_args()

args = parse_args()

root = '../../data/IDRID/IDRID 1/preprocessed'
assert os.path.exists(root)
root_sub = os.path.join(root, args.cat)
params_file = os.path.join(root_sub, '{}.csv'.format(args.cat))
root_pred = os.path.join(root_sub, 'pred')
root_pred_ori = os.path.join(root_sub, 'pred_ori')
os.makedirs(root_pred_ori, exist_ok=True)

def rescale_512_to_params(img, params):
    width = params[1] - params[0]
    height = params[3] - params[2]
    img_resize = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img_resize

def rescale_prams_to_ori(img, params, oriwidth, oriheight):
    img_arr = np.zeros((oriheight, oriwidth, 3), dtype=np.uint8)
    img_arr[params[2]:params[3],params[0]:params[1],:] = img
    return img_arr


df = pd.DataFrame.from_csv(params_file)

mask_list = glob(os.path.join(root_pred, '*.png'))
mask_list = [os.path.basename(i).replace('_pred.png', '') for i in mask_list]

for mask_file in mask_list:
    params_str = df.loc[df['image'] == mask_file]['params'].iloc[0]
    ori_width = df.loc[df['image'] == mask_file]['width'].iloc[0]
    ori_height = df.loc[df['image'] == mask_file]['height'].iloc[0]
    params_str = params_str[1:-1]
    params = []
    vecs = params_str.split(',')
    for vec in vecs:
        params.append(int(vec))
    params = np.array(params)
    mask_pred_file = os.path.join(root_pred,'{}_pred.png'.format(mask_file))
    mask = cv2.imread(mask_pred_file)
    mask_param = rescale_512_to_params(mask, params)
    mask_ori = rescale_prams_to_ori(mask_param, params, ori_width, ori_height)
    out_file = os.path.join(root_pred_ori, '{}_pred.png'.format(mask_file))
    cv2.imwrite(out_file, mask_ori)
    print('save image to {}'.format(out_file))



