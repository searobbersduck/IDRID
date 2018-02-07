import os
from glob import glob
from PIL import Image

folder_list = ['EX', 'HE', 'MA', 'SE']

root_img = '../../data/IDRID/IDRID 1/Apparent Retinopathy'
root = '../../data/IDRID/IDRID 1'
os.makedirs(root, exist_ok=True)
for f in folder_list:
    root_sub_src = os.path.join(root, f)
    root_sub_dst = os.path.join(root, 'preprocessed', f, 'pred_gt')
    assert os.path.exists(root_sub_src)
    os.makedirs(root_sub_dst, exist_ok=True)
    mask_list = glob(os.path.join(root_sub_src, '*.tif'))
    for mask_file in mask_list:
        img = Image.open(mask_file).convert('RGB')
        dst_file = os.path.join(root_sub_dst, os.path.basename(mask_file).split('.')[0]+'.png')
        img.save(dst_file)