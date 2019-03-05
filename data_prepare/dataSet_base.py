from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
from torch.utils.serialization import load_lua
import data_process.data_process_tool as date_pre
import pickle
import torch
import random


class VcDataset(Dataset):

    def __init__(self, transform, pickle_data, data_len, is_train=0):
        # super(EvDataset, self).__init__(**kwargs)
        self.prefix = '/tmp/ramdisk/processed/'

        self.data_list = pickle_data
        self.data_len = data_len

        self.transform = transform
        # self.is_train = is_train
        self.show_count = 0

    def __getitem__(self, index):
        img, label = self.get_data(index)
        # label = np.array(label)
        label_bbox = label[1]
        if label[0] > 0.65:
            label_cls = torch.FloatTensor([1, 0])
        else:
            label_cls = torch.FloatTensor([0, 1])
            label_bbox = torch.FloatTensor([0, 0, 0, 0])

        label_bbox = torch.FloatTensor(label_bbox)
        label_cls = torch.FloatTensor(label_cls)

        # print(img)
        # print([label_cls, label_bbox])

        # k = cv2.waitKey(0)
        # if k == 27:
        #     cv2.destroyAllWindows()

        return img, [label_cls, label_bbox]

    def __len__(self):
        return self.data_len

    def get_data(self, data_index):
        # print(self.data_list[0])
        # print(self.prefix + str(self.data_list[data_index][0]))

        img = cv2.imread(self.prefix + str(self.data_list[data_index][0]) + '.jpg')
        # print(img)
        if img is None:
            # print('found None: ', self.prefix + str(self.data_list[data_index][0]) + '.jpg')
            img = cv2.imread(self.prefix + str(self.data_list[0][0]) + '.jpg')
            label = self.data_list[0][1:]

        label = self.data_list[data_index][1:]

        # self.show_count += 1
        # if self.show_count == 1280:
        #     self.show_count = 0

        # cv2.imshow('img', cv2.resize(img, (0, 0), fx=8, fy=8))
        # print(label)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = np.array(img).astype(np.float)
        # img = img[:, :, None]
        img = self.transform(img).float()/127.5-1

        return img, label


if __name__ == "__main__":
    transform1 = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    )
    f = open('/home/ev_ai/dms_data/pickle_gen/data_file.p', 'rb')
    print('openning file')
    data_list = pickle.load(f)
    data_list = data_list['img']
    random.shuffle(data_list)
    f.close()
    data_len = len(data_list)
    data_processor = VcDataset(transform=transform1, pickle_data=data_list, data_len=data_len)
    data_processor.__getitem__(1000)
