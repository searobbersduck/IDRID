import sys
sys.path.append('../')

from common.crop_and_rescale_img import tight_crop_params, channelwise_ahe
import numpy as np

from skimage.filters import threshold_otsu
from skimage import measure, exposure

import skimage

import scipy.misc
from PIL import Image
import cv2
from glob import glob
import os

import pandas as pd


# out_img = os.path.join(root, 'raw')
# out_ahe = os.path.join(root, 'ahe')
# out_mask = os.path.join(root, 'ahe_mask')
# os.makedirs(out_img, exist_ok=True)
# os.makedirs(out_ahe, exist_ok=True)
# os.makedirs(out_mask, exist_ok=True)
# root_img = '../../data/IDRID/IDRID 1/Apparent Retinopathy'
# root_mask = '../../data/IDRID/IDRID 1/HE'
#
#
# img_file = '../../data/IDRID/IDRID 3/Training c/IDRiD_01.jpg'
# mask_file ='../../data/IDRID/IDRID 3/OD Segmentation Training Set/IDRiD_01_OD.tif'
#
# img = scipy.misc.imread(img_file)
# img = img.astype(np.float32)
# img /= 255
# img_crop, params = tight_crop_params(img)


def scale_image(image, scale_size):
    w, h = image.size
    tw, th = (scale_size, scale_size)
    ratio_w = tw / w
    if ratio_w < 1:
        image = image.resize((tw, th), Image.CUBIC)
    elif ratio_w > 1:
        image = image.resize((tw, th), Image.CUBIC)
    return image

def scale_image_mask(pil_img, params, scale_size):
    mask = pil_img.convert('RGB')
    np_mask = np.array(mask)
    np_mask[:, :, 1] = np_mask[:, :, 0]
    np_mask[:, :, 2] = np_mask[:, :, 0]
    mask = Image.fromarray(np_mask)
    image = mask.crop((params[0], params[2], params[1], params[3]))
    w, h = image.size
    tw, th = (scale_size, scale_size)
    ratio_w = tw / w
    if ratio_w < 1:
        image = image.resize((tw, th), Image.NEAREST)
    elif ratio_w > 1:
        image = image.resize((tw, th), Image.CUBIC)
    image = image.point(lambda p:p>100 and 255)
    return image

def crop_and_scale_imageandmask(img_file, mask_file, scale_size):
    img = scipy.misc.imread(img_file)
    img = img.astype(np.float32)
    img /= 255
    img_crop, params = tight_crop_params(img)
    pilImage = Image.fromarray(skimage.util.img_as_ubyte(img_crop))
    pilMask = Image.open(mask_file)
    pilImage_rescaled = scale_image(pilImage, scale_size)
    pilMask_rescaled = scale_image_mask(pilImage, scale_size)
    return pilImage_rescaled, pilMask_rescaled

def gen_input_data(root_img, root_mask, suffix, out_dir):
    img_list = glob(os.path.join(root_img, '*.jpg'))
    mask_list = glob(os.path.join(root_mask, '*.tif'))
    mask_list = [os.path.basename(i) for i in mask_list]
    img_list_exist = []
    for i in img_list:
        if (os.path.basename(i).split('.')[0]+'_{}.tif'.format(suffix)) in mask_list:
            img_list_exist.append(i)
    df = pd.DataFrame(columns=['image', 'params', 'width', 'height'])
    for i in img_list_exist:
        img_file = i
        mask_file = os.path.join(root_mask, os.path.basename(i).split('.')[0]+'_{}.tif'.format(suffix))
        img = scipy.misc.imread(img_file)
        img = img.astype(np.float32)
        img /= 255
        img_crop, params = tight_crop_params(img)
        img_ahe = channelwise_ahe(img_crop)
        raw_img = Image.fromarray(skimage.util.img_as_ubyte(img_crop))
        ahe_img = Image.fromarray(skimage.util.img_as_ubyte(img_ahe))
        mask = Image.open(mask_file)
        scale_size = 512
        pilImage_rescaled = scale_image(raw_img, scale_size)
        pilAHE_rescaled = scale_image(ahe_img, scale_size)
        pilMask_rescaled = scale_image_mask(mask, params, scale_size)
        image_basename = os.path.basename(i).split('.')[0]

        out_img = os.path.join(out_dir, 'raw')
        out_ahe = os.path.join(out_dir, 'ahe')
        out_mask = os.path.join(out_dir, 'ahe_mask')
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_ahe, exist_ok=True)
        os.makedirs(out_mask, exist_ok=True)

        out_imagename = os.path.join(out_img, image_basename+'.png')
        out_ahename = os.path.join(out_ahe, image_basename+'_ahe.png')
        out_maskname = os.path.join(out_mask, image_basename+'_ahe_mask.png')
        pilImage_rescaled.save(out_imagename)
        print('save {}'.format(out_imagename))
        pilAHE_rescaled.save(out_ahename)
        print('save {}'.format(out_ahename))
        pilMask_rescaled.save(out_maskname)
        print('save {}'.format(out_maskname))
        print('\n')

        img_tmp = cv2.imread(img_file)
        dict = {}
        dict['image'] = image_basename
        dict['params'] = params
        dict['width'] = img_tmp.shape[1]
        dict['height'] = img_tmp.shape[0]
        df = df.append(dict, ignore_index=True)

    df.to_csv(os.path.join(out_dir, '{}.csv'.format(suffix)))
    print('save params info to {}.csv file'.format(suffix))


folder_list = ['EX', 'HE', 'MA', 'SE']

root_img = '../../data/IDRID/IDRID 1/Apparent Retinopathy'
root = '../../data/IDRID/IDRID 1/preprocessed'
os.makedirs(root, exist_ok=True)
for f in folder_list:
    sub_root = os.path.join(root, f)
    print('create folder: {}'.format(f))
    os.makedirs(sub_root, exist_ok=True)
    root_mask = '../../data/IDRID/IDRID 1/'+f
    assert os.path.exists(root_mask)
    gen_input_data(root_img, root_mask, f, sub_root)