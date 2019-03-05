import torch
import numpy as np
import cv2

from torch.utils.serialization import load_lua

img_list = np.load("../img.npy")
label = np.load('../label.npy')

print(len(img_list))
print(len(label))

count = 0
for data_index in range(len(img_list)):

    label_data = load_lua(label[data_index], long_size=8)
    img = cv2.imread(img_list[data_index])

    white = (255, 0, 0)
    for point in np.array(label_data):
        point = point.astype(np.int)
        cv2.circle(img, tuple(point), 1, white, 5)

    cv2.imshow('img', img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
    # count+=1
    # print(count)
