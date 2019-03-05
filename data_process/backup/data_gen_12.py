import numpy as np
import cv2
import os
from torch.utils.serialization import load_lua
import data_process.data_process_tool as date_pre

from tqdm import tqdm
from time import clock

img_list = np.load("../img.npy")
label_list = np.load('../label.npy')

stdsize = 12


prefix = "c:/dms_data/"

pos_save_dir = prefix + str(stdsize) + "/positive"
part_save_dir = prefix + str(stdsize) + "/part"
neg_save_dir = prefix + str(stdsize) + '/negative'

save_dir = prefix + str(stdsize)


def mkr(dr):
    if not os.path.exists(dr):
        print(dr)
        os.mkdir(dr)


mkr(save_dir)
mkr(pos_save_dir)
mkr(part_save_dir)
mkr(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_' + str(stdsize) + '.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_' + str(stdsize) + '.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_' + str(stdsize) + '.txt'), 'w')

data_len = len(img_list)

p_idx = 0  # positive
n_idx = 0  # negative
d_idx = 0  # dont care
idx = 0
box_idx = 0

count = 0
for data_index in tqdm(range(data_len)):

    pos_num = 0
    t1 = clock()
    label_data = load_lua(label_list[data_index], long_size=8)
    raw_img = cv2.imread(img_list[data_index])

    while pos_num < 5:

        img, bbox = date_pre.get_bbox(raw_img.copy(), label_data)

        out_img, iou, offset = date_pre.get_bbox_offset(img, bbox)

        if iou >= 0.65:
            save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
            f1.write(
                str(stdsize) + "/positive/%s" % p_idx + ' 1 ' + str(offset) + '\n')
            cv2.imwrite(save_file, out_img)
            p_idx += 1
            pos_num += 1
        elif iou >= 0.4:
            save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
            f3.write(str(stdsize) + "/part/%s" % d_idx + ' 0 ' + str(offset) + '\n')
            cv2.imwrite(save_file, out_img)
            d_idx += 1
            pos_num += 1

        # print("pos_num:", pos_num)
    neg_num = 0
    while neg_num < 10:

        img, bbox = date_pre.get_bbox(raw_img.copy(), label_data)

        out_img, iou, offset = date_pre.get_bbox_n(img, bbox)

        if iou < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(str(stdsize) + "/negative/%s" % n_idx + ' -1\n')
            cv2.imwrite(save_file, out_img)
            n_idx += 1
            neg_num += 1
            # print("neg_num:", neg_num)
    # print("---------------", data_index)


f1.close()
f2.close()
f3.close()
