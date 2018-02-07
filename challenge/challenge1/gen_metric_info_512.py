import os
import sys
sys.path.append('../')
import argparse
from glob import glob

from common.idrid_metrics import metric_seg
import cv2

import pandas as pd

root = '../../data/IDRID/IDRID 1/preprocessed'
assert os.path.exists(root)

def parse_args():
    parser = argparse.ArgumentParser(description='generate segmentation metrics information')
    parser.add_argument('--cat', choices=['EX', 'HE', 'MA', 'SE'], default='EX')
    return parser.parse_args()

args = parse_args()

root_sub = os.path.join(root, args.cat)

root_pred = os.path.join(root_sub, 'pred')
root_gt = os.path.join(root_sub, 'ahe_mask')

gt_list = glob(os.path.join(root_gt, '*.png'))
gt_list = [os.path.basename(i).replace('_ahe_mask.png','') for i in gt_list]
pred_list = glob(os.path.join(root_pred, '*.png'))
pred_list = [os.path.basename(i).replace('_pred.png','') for i in pred_list]

exist_list = []

for i in gt_list:
    if i in pred_list:
        exist_list.append(i)

df = pd.DataFrame(columns=['image', 'sn', 'sp', 'f1'])

for i in exist_list:
    gt_file = os.path.join(root_gt, '{}_ahe_mask.png'.format(i))
    pred_file = os.path.join(root_pred, '{}_pred.png'.format(i))
    pred_img = cv2.imread(pred_file)
    gt_img = cv2.imread(gt_file)
    s1, s2, f = metric_seg(pred_img[:, :, 2], gt_img[:, :, 2])
    dict = {}
    dict['image'] = i
    dict['sn'] = round(s1, 4)
    dict['sp'] = round(s2, 4)
    dict['f1 score'] = round(f, 4)
    df = df.append(dict, ignore_index=True)
    print('sensitivity: {}\t specificity: {}\t f1 score: {}\t'.format(round(s1,4), round(s2,4), round(f,4)))

df.to_csv(os.path.join(root_sub, 'metric_info_512.csv'))