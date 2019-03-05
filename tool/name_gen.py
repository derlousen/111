import torch
from torch.utils.serialization import load_lua
import os
import numpy as np

# a = load_lua('../LS3D-W/300VW-3D/CatA/114/0001.t7', long_size=8)
# print(a)

img_list = []
label_list = []

count = 0
for root, dirs, files in os.walk('../LS3D-W/'):
    # begin

    # print(dirs)
    root = root.replace('\\', '/')
    current_path = os.path.abspath('../').replace('\\', '/')

    root = root.replace('..', current_path)
    root = root + "/"
    # print(root)
    files.sort()

    for _ in files:
        if _[-2:] == "t7":
            label_list.append(root + _)

            if os.path.exists(root+_[:-3]+'.jpg'):
                img_list.append(root + _[:-3]+'.jpg')
                continue
            if os.path.exists(root+_[:-3]+'.JPG'):
                img_list.append(root + _[:-3]+'.JPG')
                continue
            if os.path.exists(root+_[:-3]+'.png'):
                img_list.append(root + _[:-3]+'.png')
                continue
            if os.path.exists(root+_[:-3]+'.PNG'):
                img_list.append(root + _[:-3]+'.PNG')
                continue
    print(count)
    count += 1

img_list = np.array(img_list)
label_list = np.array(label_list)

np.save('img.npy', img_list)
np.save('label.npy', label_list)
