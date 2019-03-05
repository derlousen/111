import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import model.net_model.model as netmodel
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

model_1 = netmodel.ONetOld()
model_1.eval()
model_1.load_state_dict(torch.load('weight/model_015_0.0013178857_bad.pth'))

model_2 = netmodel.ONet()
model_2.eval()
model_2.load_state_dict(torch.load('weight/model_034_0.29928195.pth'))

plt.figure()

cv2.namedWindow("1", 0)
cv2.namedWindow("2", 0)


def run_para(modelO, pri=0):
    weights = []
    bias = []
    for n, p in modelO.named_parameters():

        if n[-6:] == 'weight':
            if len(p.data.shape) == 4:
                weights.append(p.data.numpy())
                # for we in p.data:
                #     weights.append(we.numpy())

                # print(torch.max(p.data))

                if pri:
                    print(p.data.numpy().shape, n)

        elif n[-4:] == 'bias':

            for bi in p.data:
                bias.append(bi.numpy())

    weights = np.array(weights)
    bias = np.array(bias)

    length = len(weights)

    max_list = []
    min_list = []

    # print(bias.shape, weights.shape)

    for i in weights:

        max_list.append(np.max(i))
        min_list.append(np.min(i))

    # for i in bias:
    #     min_list.append(np.median(i))

    max_list = np.array(max_list)
    min_list = np.array(min_list)

    max_list = max_list+min_list

    return max_list, min_list, length


max1, min1, length1 = run_para(model_1, 1)
max2, min2, length2 = run_para(model_2)


def get_features_hook1(self, input, output):

    data = output.data.numpy()[0]
    max_d = np.max(data)
    min_d = np.min(data)
    print('model_1: ', output.data.numpy().shape, max_d, min_d)
    out_list = []
    shape = data.shape[0]

    size = 3
    if shape%3 > 0:
        size = 8
    if shape % 8 == 0:
        size = 8

    for i in range(int(shape/size)):
        out_list.append(np.hstack(data[i*size:(i+1)*size, :, :]))
    out = np.vstack(out_list)

    out = (out+min_d)/(max_d-min_d)*255
    out = out.astype(np.uint8)

    cv2.imshow('1', out)



def get_features_hook2(self, input, output):
    data = output.data.numpy()[0]
    max_d = np.max(data)
    min_d = np.min(data)
    print('model_2: ', output.data.numpy().shape, max_d, min_d)
    out_list = []

    shape = data.shape[0]
    size = 3
    if shape%3 > 0:
        size = 8
    if shape % 8 == 0:
        size = 8
    for i in range(int(shape/size)):
        out_list.append(np.hstack(data[i * size:(i + 1) * size, :, :]))
    out = np.vstack(out_list)

    out = (out+min_d)/(max_d-min_d)*255
    out = out.astype(np.uint8)

    cv2.imshow('2', out)


handle = model_1.features.conv1.conv1.register_forward_hook(get_features_hook1)
handle2 = model_2.features.conv1.conv1.register_forward_hook(get_features_hook2)


current_img = cv2.imread('saved_img.jpg')

# current_img = (np.random.random((324, 324, 3))*255).astype(np.uint8)
# print(current_img.shape)

img_size = 96
transform1 = transforms.Compose([
    transforms.ToTensor(),
])
imCrop = cv2.resize(current_img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
# imCrop = origin_img
img = transform1(imCrop)
img = img[np.newaxis, :, :, :]
tensor_img = torch.FloatTensor(img) * 2 - 1  # GPU

# cls_pred, box_offset_pred, point_list = model_O(tensor_img)
point_list = model_1(tensor_img)
point_list = model_2(tensor_img)


cv2.waitKey(0)

x = np.linspace(0, length1, length1)

plt.subplot(211)
plt.scatter(x, min1, 1)
plt.scatter(x, max1, 1)

# my_x_ticks = np.arange(-0.5, 5, 1)
# my_y_ticks = np.arange(-5, 4, 1)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)

plt.subplot(212)
plt.scatter(x, min1, 1)
plt.scatter(x, max1, 1)

# my_x_ticks = np.arange(-0.5, 5, 1)
# my_y_ticks = np.arange(-5, 4, 1)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)

# plt.show()


