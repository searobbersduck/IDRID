# 文件功能记录：
* gen_gt_ori.py
    * 生成和原始金表准（*.tif）等大的，后缀名为.jpg的金表准图像
    * 位于数据子文件夹下的preprocessed子文件夹下，对应种类（EX/HE/MA/SE）下的pred_gt，文件夹下
* gen_metric_info.py
    * 得到原始图像的分割的各项指标，包括sn, sp, ppv, f1 score等
    * 支持各种分割结果的计算
    * 输入为gt和pred的分割结果
* gen_metric_info_512.py
    * 同gen_metric_info.py
    * 目前的整图分割是先做512图像的分割，再放大到原始尺寸
    * 该功能是计算在未放大之前，512尺寸上的各项指标
* preprocessing.py
    * 生成512大小的数据
    * 缩放成512的数据，crop图像区域（crop掉黑边，w和h可能不等大），缩放到512大小（不等比例缩放）
    * 生成对应的ahe数据，以及对应的mask数据
    * 位于数据子文件夹下的preprocessed子文件夹下，对应种类（EX/HE/MA/SE）下的raw（512图像）/ahe(512 ahe图像)/ahe_mask(512 mask图像)
    * 生成记录原始图像大小/crop位置信息的文件，位于数据子文件夹下的preprocessed子文件夹下的*.csv文件
* rescale_512_to_ori.py
    * 算法直接输出的整图分割图像是512大小的，该功能是将512大小的图像还原到原始大小
    * 需要用到记录原始图像大小/crop位置信息的*.csv文件
* show_mask.py
    * 将不同分割结果的mask，拼到一张图像上进行显示，看相互之间是否有重合区域
    * 若无重合区域，可以在一个网络中直接输出四种分割效果（EX/HE/MA/SE）
* mix_label.py
    * 将不同病灶的label，综合到一张mask上
    * label的对应关系分别为：ex: 1,he: 2,ma: 3,se: 4
