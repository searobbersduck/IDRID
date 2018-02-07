import numpy as np
import cv2

'''


'''
def metric_seg(pred_label, gt_label):
    assert pred_label.dtype == np.uint8
    assert gt_label.dtype == np.uint8
    assert pred_label.shape == gt_label.shape
    tp_cnt = 0
    fn_cnt = 0
    fp_cnt = 0
    tn_cnt = 0
    for i in range(gt_label.shape[0]):
        for j in range(gt_label.shape[1]):
            if pred_label[i][j] == 255:
                if gt_label[i][j] == 255:
                    tp_cnt += 1
                else:
                    fn_cnt += 1
            else:
                if gt_label[i][j] == 255:
                    fp_cnt += 1
                else:
                    tn_cnt += 1
    sensitivity = 0
    if (tp_cnt + fn_cnt) == 0:
        sensitivity = -1
    else:
        sensitivity = tpr = tp_cnt / (tp_cnt + fn_cnt)

    specificity = tnr = tn_cnt / (tn_cnt + fp_cnt)

    f1 = 0
    if (2 * tp_cnt + fp_cnt + fn_cnt) == 0:
        f1 = -1
    else:
        f1 = 2 * tp_cnt / (2 * tp_cnt + fp_cnt + fn_cnt)
    return sensitivity, specificity, f1

def test_metric_seg():
    pred_img_file = '/home/weidong/code/kaggle/IDRID/data/IDRID/IDRID 1/preprocessed/EX/ahe_mask/IDRiD_02_ahe_mask.png'
    gt_img_file = '/home/weidong/code/kaggle/IDRID/data/IDRID/IDRID 1/preprocessed/EX/ahe_mask/IDRiD_03_ahe_mask.png'
    gt_img = cv2.imread(pred_img_file)
    pred_img = cv2.imread(gt_img_file)
    s1,s2,f = metric_seg(pred_img[:,:,2], gt_img[:,:,2])
    print('sensitivity: {}\t specificity: {}\t f1 score: {}\t'.format(s1, s2, f))
    print('hello')

if __name__ == '__main__':
    test_metric_seg()
